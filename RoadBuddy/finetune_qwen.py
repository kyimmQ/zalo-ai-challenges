"""
Fine-tune Qwen3-VL with LoRA on RoadBuddy training data
Optimized for 24GB GPU memory
Compatible with both Qwen2-VL and Qwen3-VL
"""

import json
import os
import torch
import yaml
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import random
from PIL import Image
import cv2

from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)

# Import QwenVL model (supports both Qwen2 and Qwen3)
try:
    from transformers import Qwen3VLForConditionalGeneration as QwenVLForConditionalGeneration
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as QwenVLForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from qwen_vl_utils import process_vision_info


@dataclass
class RoadBuddyDataCollator:
    """Custom data collator for RoadBuddy dataset"""
    processor: Any
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch of features"""
        # Extract messages and process
        messages_list = [f['messages'] for f in features]
        
        # Process all messages
        texts = []
        all_image_inputs = []
        all_video_inputs = []
        
        for messages in messages_list:
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)
            all_image_inputs.extend(image_inputs or [])
            all_video_inputs.extend(video_inputs or [])
        
        # Process batch
        batch = self.processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        
        # Prepare labels (same as input_ids for causal LM)
        batch['labels'] = batch['input_ids'].clone()
        
        return batch


class RoadBuddyFineTuner:
    """Fine-tune Qwen3-VL on RoadBuddy dataset (compatible with Qwen2-VL)"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize fine-tuner"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initializing RoadBuddy Fine-Tuner")
        print(f"   Device: {self.device}")
    
    def load_data(self) -> tuple:
        """Load and prepare training data"""
        print("\n📊 Loading training data...")
        
        data_config = self.config['data']
        json_path = data_config['train_json']
        base_dir = data_config['base_dir']
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)['data']
        
        print(f"   Total samples: {len(data)}")
        
        # Shuffle and split
        random.seed(data_config['random_seed'])
        random.shuffle(data)
        
        split_idx = int(len(data) * data_config['train_split'])
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        print(f"   Train samples: {len(train_data)}")
        print(f"   Validation samples: {len(val_data)}")
        
        return train_data, val_data, base_dir
    
    def extract_frames(self, video_path: str, support_frames: List[float] = None,
                       max_frames: int = 6) -> List[Image.Image]:
        """Extract frames from video for training"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        
        if support_frames:
            # Use support frames + some context
            timestamps = support_frames[:max_frames]
        else:
            # Uniform sampling
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            num_frames = min(max_frames, int(duration))
            timestamps = [duration * i / num_frames for i in range(num_frames)]
        
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize for efficiency
                frame_resized = cv2.resize(frame_rgb, (384, 384))
                pil_image = Image.fromarray(frame_resized)
                frames.append(pil_image)
        
        cap.release()
        return frames
    
    def prepare_sample(self, sample: Dict, base_dir: str) -> Dict:
        """Prepare a single sample for training"""
        video_path = os.path.join(base_dir, sample['video_path'])
        
        # Check if video exists
        if not os.path.exists(video_path):
            return None
        
        try:
            # Extract frames
            frames = self.extract_frames(
                video_path,
                support_frames=sample.get('support_frames'),
                max_frames=self.config['frames']['max_frames']
            )
            
            if not frames:
                return None
            
            # Create prompt
            question = sample['question']
            choices = sample['choices']
            choices_text = "\n".join(choices)
            
            prompt = f"""Bạn là trợ lý phân tích video giao thông theo luật giao thông Việt Nam.

Câu hỏi: {question}

Lựa chọn:
{choices_text}

Hãy phân tích video và trả lời chỉ bằng một chữ cái (A, B, C, hoặc D):"""
            
            # Get answer letter
            answer_full = sample['answer']  # e.g., "A. Đúng"
            answer_letter = answer_full.split('.')[0].strip()
            
            # Format as messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frames,
                            "fps": 1.0,
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer_letter}
                    ],
                }
            ]
            
            return {"messages": messages, "id": sample['id']}
            
        except Exception as e:
            print(f"⚠️ Error preparing sample {sample['id']}: {e}")
            return None
    
    def prepare_dataset(self, data: List[Dict], base_dir: str) -> Dataset:
        """Prepare dataset for training"""
        print(f"\n🔧 Preparing dataset...")
        
        processed_samples = []
        for i, sample in enumerate(data):
            if i % 100 == 0:
                print(f"   Processing: {i}/{len(data)}")
            
            prepared = self.prepare_sample(sample, base_dir)
            if prepared:
                processed_samples.append(prepared)
        
        print(f"   Successfully prepared: {len(processed_samples)}/{len(data)} samples")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(processed_samples)
        return dataset
    
    def setup_model_and_processor(self):
        """Setup model with LoRA and processor"""
        print("\n🤖 Setting up model...")
        
        model_config = self.config['model']
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load model
        model = QwenVLForConditionalGeneration.from_pretrained(
            model_config['name'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare for training
        model = prepare_model_for_kbit_training(model)
        
        # Setup LoRA
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            lora_dropout=self.config['lora']['lora_dropout'],
            target_modules=self.config['lora']['target_modules'],
            bias=self.config['lora']['bias'],
            task_type=self.config['lora']['task_type'],
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_config['name'],
            trust_remote_code=True
        )
        
        return model, processor
    
    def train(self):
        """Run fine-tuning"""
        # Load data
        train_data, val_data, base_dir = self.load_data()
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_data, base_dir)
        val_dataset = self.prepare_dataset(val_data, base_dir)
        
        # Setup model
        model, processor = self.setup_model_and_processor()
        
        # Data collator
        data_collator = RoadBuddyDataCollator(processor=processor)
        
        # Training arguments
        training_config = self.config['training']
        training_args = TrainingArguments(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            warmup_steps=training_config['warmup_steps'],
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            save_total_limit=training_config['save_total_limit'],
            fp16=training_config['fp16'],
            bf16=training_config['bf16'],
            gradient_checkpointing=training_config['gradient_checkpointing'],
            optim=training_config['optim'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            max_grad_norm=training_config['max_grad_norm'],
            group_by_length=training_config['group_by_length'],
            report_to="none",  # Disable wandb/tensorboard for now
            remove_unused_columns=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("\n🚀 Starting training...\n")
        trainer.train()
        
        # Save final model
        final_path = os.path.join(training_config['output_dir'], "final")
        trainer.save_model(final_path)
        print(f"\n✅ Training complete! Model saved to: {final_path}")


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL for RoadBuddy")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize and train
    finetuner = RoadBuddyFineTuner(config_path=args.config)
    finetuner.train()


if __name__ == "__main__":
    main()

