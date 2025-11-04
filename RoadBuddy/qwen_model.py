"""
Qwen3-VL Model Loading and Configuration
Handles model initialization with quantization for efficient inference
"""

import torch
try:
    from transformers import Qwen3VLForConditionalGeneration as QwenVLForConditionalGeneration
except ImportError:
    # Fallback to Qwen2VL if Qwen3VL not available
    from transformers import Qwen2VLForConditionalGeneration as QwenVLForConditionalGeneration
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


class QwenVLModel:
    """
    Wrapper class for Qwen3-VL model with quantization support
    Compatible with both Qwen2-VL and Qwen3-VL
    """
    
    def __init__(self, config_path: str = "config.yaml", use_lora: bool = False, 
                 lora_path: Optional[str] = None):
        """
        Initialize Qwen3-VL model
        
        Args:
            config_path: Path to configuration YAML file
            use_lora: Whether to load LoRA weights
            lora_path: Path to LoRA checkpoint
        """
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_name = self.config['model']['name']
        print(f"🚀 Initializing Qwen-VL Model: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   Quantization: {self.config['model']['quantization']}")
        
        # Load model and processor
        self.model = self._load_model()
        self.processor = self._load_processor()
        
        # Load LoRA weights if specified
        if use_lora and lora_path:
            self._load_lora_weights(lora_path)
        
        print(f"✅ Model loaded successfully")
        self._print_model_info()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self) -> QwenVLForConditionalGeneration:
        """Load Qwen-VL model with quantization (supports both Qwen2-VL and Qwen3-VL)"""
        model_config = self.config['model']
        
        # Setup quantization config
        if model_config['quantization'] == '4bit':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config['quantization_config']['load_in_4bit'],
                bnb_4bit_compute_dtype=getattr(torch, 
                    self.config['quantization_config']['bnb_4bit_compute_dtype']),
                bnb_4bit_use_double_quant=self.config['quantization_config']['bnb_4bit_use_double_quant'],
                bnb_4bit_quant_type=self.config['quantization_config']['bnb_4bit_quant_type']
            )
        elif model_config['quantization'] == '8bit':
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None
        
        # Load model
        model = QwenVLForConditionalGeneration.from_pretrained(
            model_config['name'],
            torch_dtype=getattr(torch, model_config['torch_dtype']),
            device_map=model_config['device_map'],
            quantization_config=quantization_config,
            trust_remote_code=model_config['trust_remote_code']
        )
        
        # Set to evaluation mode
        model.eval()
        
        return model
    
    def _load_processor(self) -> AutoProcessor:
        """Load Qwen-VL processor"""
        processor = AutoProcessor.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True
        )
        return processor
    
    def _load_lora_weights(self, lora_path: str):
        """Load LoRA weights from checkpoint"""
        from peft import PeftModel
        
        print(f"📦 Loading LoRA weights from: {lora_path}")
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_path,
            is_trainable=False
        )
        self.model.eval()
        print(f"✅ LoRA weights loaded")
    
    def _print_model_info(self):
        """Print model information"""
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n📊 Model Information:")
        print(f"   Total parameters: {total_params / 1e9:.2f}B")
        print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M")
        
        # Check memory usage if CUDA
        if torch.cuda.is_available():
            print(f"   GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            print(f"   GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    
    def prepare_inputs(self, frames, text: str):
        """
        Prepare inputs for the model
        
        Args:
            frames: List of PIL Images or numpy arrays
            text: Text prompt
            
        Returns:
            Processed inputs ready for model
        """
        from qwen_vl_utils import process_vision_info
        
        # Prepare messages format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,  # List of PIL images
                        "fps": 1.0,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]
        
        # Apply chat template
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process inputs
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = inputs.to(self.device)
        
        return inputs
    
    @torch.no_grad()
    def generate(self, inputs, **generation_kwargs):
        """
        Generate response from model
        
        Args:
            inputs: Processed inputs from prepare_inputs
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Merge with default generation config
        gen_config = self.config['inference'].copy()
        gen_config.update(generation_kwargs)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=gen_config['max_new_tokens'],
            temperature=gen_config['temperature'],
            do_sample=gen_config['do_sample'],
            top_p=gen_config['top_p'],
        )
        
        # Decode
        generated_text = self.processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return generated_text


def test_model_loading():
    """Test model loading"""
    print("Testing Qwen-VL model loading...")
    
    try:
        model = QwenVLModel()
        print("\n✅ Model loading test successful!")
        return model
    except Exception as e:
        print(f"\n❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test model loading
    model = test_model_loading()

