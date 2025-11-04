"""
Generate submission file for RoadBuddy challenge
Simple wrapper around qwen_solver.py
"""

from pathlib import Path
from qwen_solver import QwenVLSolver
import argparse


def generate_submission(use_lora: bool = False, lora_path: str = None,
                       output_file: str = "submission.csv"):
    """
    Generate submission file for public test
    
    Args:
        use_lora: Whether to use fine-tuned LoRA weights
        lora_path: Path to LoRA checkpoint
        output_file: Output CSV filename
    """
    print("=" * 70)
    print("🎯 RoadBuddy Submission Generator")
    print("=" * 70)
    
    # Initialize solver
    solver = QwenVLSolver(
        config_path="config.yaml",
        use_lora=use_lora,
        lora_path=lora_path
    )
    
    # Setup paths
    base_dir = Path(__file__).parent / "traffic_buddy_train+public_test"
    test_json = base_dir / "public_test" / "public_test.json"
    
    if not test_json.exists():
        print(f"❌ Test file not found: {test_json}")
        print(f"   Please ensure the dataset is downloaded and extracted.")
        return
    
    # Generate predictions
    print(f"\n🚀 Generating predictions for public test...")
    solver.solve_dataset(
        json_path=str(test_json),
        output_csv=output_file,
        base_dir=str(base_dir)
    )
    
    print(f"\n{'='*70}")
    print(f"✅ Submission file created: {output_file}")
    print(f"📤 Submit this file to the challenge platform!")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Generate RoadBuddy submission")
    parser.add_argument('--use-lora', action='store_true',
                       help='Use fine-tuned LoRA weights')
    parser.add_argument('--lora-path', type=str, default=None,
                       help='Path to LoRA checkpoint (e.g., checkpoints/qwen2vl-roadbuddy/final)')
    parser.add_argument('--output', type=str, default='submission.csv',
                       help='Output CSV filename')
    
    args = parser.parse_args()
    
    # Validate LoRA path if specified
    if args.use_lora and not args.lora_path:
        print("❌ Error: --lora-path required when using --use-lora")
        return
    
    if args.lora_path and not Path(args.lora_path).exists():
        print(f"❌ Error: LoRA path not found: {args.lora_path}")
        return
    
    # Generate submission
    generate_submission(
        use_lora=args.use_lora,
        lora_path=args.lora_path,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

