"""
Qwen3-VL Solver for RoadBuddy Challenge
Optimized for speed while maintaining accuracy
Compatible with both Qwen2-VL and Qwen3-VL
"""

import json
import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from PIL import Image
import torch

from qwen_model import QwenVLModel


class QwenVLSolver:
    """
    RoadBuddy solver using Qwen3-VL model
    Optimized for <10s per sample inference
    """
    
    def __init__(self, config_path: str = "config.yaml", use_lora: bool = False,
                 lora_path: Optional[str] = None):
        """
        Initialize Qwen-VL solver
        
        Args:
            config_path: Path to configuration file
            use_lora: Whether to use fine-tuned LoRA weights
            lora_path: Path to LoRA checkpoint
        """
        self.qwen_model = QwenVLModel(config_path, use_lora, lora_path)
        self.config = self.qwen_model.config
        self.video_cache = {}  # Cache for videos with multiple questions
        
        print(f"\n🎯 QwenVL Solver initialized")
        print(f"   Frame extraction: {self.config['frames']['method']}")
        print(f"   Max frames: {self.config['frames']['max_frames']}")
    
    def extract_frames(self, video_path: str, method: str = None, 
                      support_frames: List[float] = None) -> List[Image.Image]:
        """
        Extract frames from video with optimization
        
        Args:
            video_path: Path to video file
            method: 'uniform', 'support', or 'smart'
            support_frames: Key timestamps (in seconds)
            
        Returns:
            List of PIL Images
        """
        if method is None:
            method = self.config['frames']['method']
        
        max_frames = self.config['frames']['max_frames']
        target_size = tuple(self.config['frames']['target_size'])
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        frames = []
        timestamps = []
        
        if method == "support" and support_frames:
            # Extract frames around support timestamps
            timestamps = support_frames[:max_frames]
            
        elif method == "smart":
            # Smart sampling: beginning, middle, end + support frames if available
            base_timestamps = [
                0.5,  # Near beginning
                duration * 0.33,
                duration * 0.67,
                max(0, duration - 0.5)  # Near end
            ]
            
            if support_frames:
                # Add support frames
                base_timestamps.extend(support_frames)
            
            # Remove duplicates and sort
            timestamps = sorted(set(base_timestamps))[:max_frames]
            
        else:  # uniform
            fps = self.config['frames']['fps']
            num_frames = min(int(duration * fps), max_frames)
            timestamps = np.linspace(0, duration - 0.1, num_frames).tolist()
        
        # Extract frames at timestamps
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize for efficiency
                frame_resized = cv2.resize(frame_rgb, (target_size[1], target_size[0]))
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_resized)
                frames.append(pil_image)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")
        
        return frames
    
    def create_prompt(self, question: str, choices: List[str]) -> str:
        """
        Create optimized prompt for Vietnamese traffic VQA
        
        Args:
            question: Question text in Vietnamese
            choices: List of answer choices
            
        Returns:
            Formatted prompt
        """
        # Format choices
        choices_text = "\n".join(choices)
        
        # Optimized prompt for speed and accuracy
        prompt = f"""Bạn là trợ lý phân tích video giao thông theo luật giao thông Việt Nam.

Câu hỏi: {question}

Lựa chọn:
{choices_text}

Hãy phân tích video và trả lời chỉ bằng một chữ cái (A, B, C, hoặc D):"""
        
        return prompt
    
    def parse_answer(self, model_output: str, choices: List[str]) -> str:
        """
        Parse model output to extract answer letter
        
        Args:
            model_output: Raw output from model
            choices: List of answer choices
            
        Returns:
            Single letter: 'A', 'B', 'C', or 'D'
        """
        # Clean output
        output_clean = model_output.strip().upper()
        
        # Try to find answer letter at the beginning
        for letter in ['A', 'B', 'C', 'D']:
            # Check if letter is at start or first character
            if output_clean.startswith(letter):
                return letter
            # Check if it's the only letter in a short output
            if letter in output_clean[:5]:
                return letter
        
        # Try to match full choice text
        for i, choice in enumerate(choices):
            choice_letter = chr(65 + i)  # A=65, B=66, etc.
            choice_text = choice.split('.', 1)[-1].strip().lower()
            if choice_text in model_output.lower():
                return choice_letter
        
        # Default to first valid letter found
        for letter in ['A', 'B', 'C', 'D'][:len(choices)]:
            if letter in output_clean:
                return letter
        
        # Last resort: return 'A'
        print(f"⚠️ Could not parse answer from: '{model_output}', defaulting to A")
        return 'A'
    
    def answer_question(self, video_path: str, question: str, 
                       choices: List[str], support_frames: List[float] = None) -> str:
        """
        Answer a single question about a video
        
        Args:
            video_path: Path to video file
            question: Question text
            choices: List of answer choices
            support_frames: Key timestamps (training only)
            
        Returns:
            Answer letter: 'A', 'B', 'C', or 'D'
        """
        try:
            # Extract frames (use cache if available)
            cache_key = f"{video_path}_{support_frames}"
            if cache_key in self.video_cache:
                frames = self.video_cache[cache_key]
            else:
                frames = self.extract_frames(video_path, support_frames=support_frames)
                # Cache if video is reused
                self.video_cache[cache_key] = frames
                # Limit cache size
                if len(self.video_cache) > 50:
                    # Remove oldest entry
                    self.video_cache.pop(next(iter(self.video_cache)))
            
            # Create prompt
            prompt = self.create_prompt(question, choices)
            
            # Prepare inputs
            inputs = self.qwen_model.prepare_inputs(frames, prompt)
            
            # Generate answer
            output = self.qwen_model.generate(inputs)
            
            # Parse answer
            answer_letter = self.parse_answer(output, choices)
            
            return answer_letter
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            import traceback
            traceback.print_exc()
            return 'A'  # Default fallback
    
    def solve_dataset(self, json_path: str, output_csv: str, base_dir: str = "."):
        """
        Solve all questions in a dataset
        
        Args:
            json_path: Path to JSON file (train or test)
            output_csv: Path to save predictions
            base_dir: Base directory for video paths
        """
        print(f"\n{'='*70}")
        print(f"🎯 Solving dataset: {json_path}")
        print(f"{'='*70}\n")
        
        # Load data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)['data']
        
        results = []
        start_time = time.time()
        times = []
        
        # Group by video for efficiency
        video_groups = {}
        for sample in data:
            video_path = sample['video_path']
            if video_path not in video_groups:
                video_groups[video_path] = []
            video_groups[video_path].append(sample)
        
        print(f"📊 Processing {len(data)} questions from {len(video_groups)} unique videos\n")
        
        processed = 0
        for video_path, samples in video_groups.items():
            full_video_path = os.path.join(base_dir, video_path)
            
            if not os.path.exists(full_video_path):
                print(f"⚠️ Video not found: {video_path}")
                for sample in samples:
                    results.append({'id': sample['id'], 'answer': 'A'})
                    processed += 1
                continue
            
            # Process all questions for this video
            for sample in samples:
                sample_start = time.time()
                
                support_frames = sample.get('support_frames', None)
                
                # Answer question
                predicted_answer = self.answer_question(
                    video_path=full_video_path,
                    question=sample['question'],
                    choices=sample['choices'],
                    support_frames=support_frames
                )
                
                # Store result
                results.append({
                    'id': sample['id'],
                    'answer': predicted_answer
                })
                
                sample_time = time.time() - sample_start
                times.append(sample_time)
                processed += 1
                
                # Show progress
                print(f"  [{processed}/{len(data)}] {sample['id']}: {predicted_answer} ({sample_time:.2f}s)")
                
                # Warning for slow samples
                if sample_time > 30:
                    print(f"  ⚠️ WARNING: Exceeded 30s time limit!")
        
        total_time = time.time() - start_time
        avg_time = np.mean(times) if times else 0
        median_time = np.median(times) if times else 0
        
        print(f"\n{'='*70}")
        print(f"✅ Completed {len(data)} samples")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"⏱️  Average time: {avg_time:.2f}s")
        print(f"⏱️  Median time: {median_time:.2f}s")
        print(f"⏱️  Min time: {min(times):.2f}s")
        print(f"⏱️  Max time: {max(times):.2f}s")
        
        # Count samples over 30s
        over_limit = sum(1 for t in times if t > 30)
        if over_limit > 0:
            print(f"⚠️  Samples over 30s limit: {over_limit} ({over_limit/len(times)*100:.1f}%)")
        
        print(f"{'='*70}\n")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"💾 Predictions saved to: {output_csv}\n")
        
        return results


def main():
    """Test the solver"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3-VL RoadBuddy Solver")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, default='test',
                       choices=['train', 'test'],
                       help='Dataset to solve')
    parser.add_argument('--use-lora', action='store_true',
                       help='Use fine-tuned LoRA weights')
    parser.add_argument('--lora-path', type=str, default=None,
                       help='Path to LoRA checkpoint')
    parser.add_argument('--output', type=str, default='submission.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    # Initialize solver
    solver = QwenVLSolver(
        config_path=args.config,
        use_lora=args.use_lora,
        lora_path=args.lora_path
    )
    
    # Setup paths
    base_dir = Path(__file__).parent / "traffic_buddy_train+public_test"
    
    if args.dataset == 'train':
        json_path = base_dir / "train" / "train.json"
    else:
        json_path = base_dir / "public_test" / "public_test.json"
    
    if not json_path.exists():
        print(f"❌ Dataset not found: {json_path}")
        return
    
    # Solve dataset
    solver.solve_dataset(
        json_path=str(json_path),
        output_csv=args.output,
        base_dir=str(base_dir)
    )
    
    print(f"🎉 Done! Submit {args.output} to the challenge platform.")


if __name__ == "__main__":
    main()

