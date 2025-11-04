#!/bin/bash
# RoadBuddy Qwen3-VL Runner Script
# Convenience script for running different stages

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "🚗 RoadBuddy Qwen3-VL Runner"
echo "=========================================="
echo ""

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if config exists
if [ ! -f "config.yaml" ]; then
    print_error "config.yaml not found!"
    exit 1
fi

# Parse command
COMMAND=${1:-help}

case $COMMAND in
    setup)
        print_info "Installing dependencies..."
        pip install -r requirements_qwen.txt
        print_info "Setup complete!"
        ;;
    
    test-model)
        print_info "Testing model loading..."
        python qwen_model.py
        ;;
    
    inference-baseline)
        print_info "Running baseline inference on public test..."
        python qwen_solver.py \
            --dataset test \
            --output submission_baseline.csv
        ;;
    
    inference-train)
        print_info "Running inference on training set (for evaluation)..."
        python qwen_solver.py \
            --dataset train \
            --output train_predictions.csv
        ;;
    
    finetune)
        print_info "Starting fine-tuning..."
        print_warning "This will take 4-8 hours on RTX 3090!"
        python finetune_qwen.py --config config.yaml
        ;;
    
    inference-finetuned)
        if [ -z "$2" ]; then
            LORA_PATH="checkpoints/qwen3vl-roadbuddy/final"
        else
            LORA_PATH="$2"
        fi
        
        if [ ! -d "$LORA_PATH" ]; then
            print_error "LoRA checkpoint not found: $LORA_PATH"
            print_info "Usage: ./run_qwen.sh inference-finetuned [lora_path]"
            exit 1
        fi
        
        print_info "Running inference with fine-tuned model..."
        python qwen_solver.py \
            --dataset test \
            --use-lora \
            --lora-path "$LORA_PATH" \
            --output submission_finetuned.csv
        ;;
    
    submission)
        if [ -z "$2" ]; then
            print_info "Generating submission with baseline model..."
            python generate_submission.py --output submission.csv
        else
            LORA_PATH="$2"
            if [ ! -d "$LORA_PATH" ]; then
                print_error "LoRA checkpoint not found: $LORA_PATH"
                exit 1
            fi
            print_info "Generating submission with fine-tuned model..."
            python generate_submission.py \
                --use-lora \
                --lora-path "$LORA_PATH" \
                --output submission.csv
        fi
        ;;
    
    evaluate)
        if [ -z "$2" ]; then
            PRED_FILE="train_predictions.csv"
        else
            PRED_FILE="$2"
        fi
        
        if [ ! -f "$PRED_FILE" ]; then
            print_error "Predictions file not found: $PRED_FILE"
            print_info "Usage: ./run_qwen.sh evaluate [predictions.csv]"
            exit 1
        fi
        
        print_info "Evaluating predictions..."
        python evaluate.py \
            --predictions "$PRED_FILE" \
            --output-dir evaluation_results
        ;;
    
    full-pipeline)
        print_info "Running full pipeline..."
        print_warning "This will take many hours!"
        
        print_info "Step 1/4: Baseline inference on test..."
        python qwen_solver.py --dataset test --output submission_baseline.csv
        
        print_info "Step 2/4: Fine-tuning..."
        python finetune_qwen.py --config config.yaml
        
        print_info "Step 3/4: Fine-tuned inference on test..."
        python qwen_solver.py \
            --dataset test \
            --use-lora \
            --lora-path checkpoints/qwen3vl-roadbuddy/final \
            --output submission_finetuned.csv
        
        print_info "Step 4/4: Evaluation on train..."
        python qwen_solver.py \
            --dataset train \
            --use-lora \
            --lora-path checkpoints/qwen3vl-roadbuddy/final \
            --output train_predictions.csv
        
        python evaluate.py \
            --predictions train_predictions.csv \
            --output-dir evaluation_results
        
        print_info "Pipeline complete!"
        ;;
    
    help|*)
        echo "Usage: ./run_qwen.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  setup                 - Install dependencies"
        echo "  test-model            - Test model loading"
        echo "  inference-baseline    - Run baseline inference on test"
        echo "  inference-train       - Run inference on train (for eval)"
        echo "  finetune              - Fine-tune model on training data"
        echo "  inference-finetuned [lora_path]  - Run inference with fine-tuned model"
        echo "  submission [lora_path]           - Generate submission file"
        echo "  evaluate [pred_file]             - Evaluate predictions"
        echo "  full-pipeline         - Run complete pipeline (baseline → finetune → submission)"
        echo ""
        echo "Examples:"
        echo "  ./run_qwen.sh setup"
        echo "  ./run_qwen.sh test-model"
        echo "  ./run_qwen.sh inference-baseline"
        echo "  ./run_qwen.sh finetune"
        echo "  ./run_qwen.sh submission checkpoints/qwen3vl-roadbuddy/final"
        echo "  ./run_qwen.sh evaluate train_predictions.csv"
        echo ""
        echo "For more details, see QWEN_IMPLEMENTATION.md"
        ;;
esac

