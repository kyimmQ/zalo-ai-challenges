#!/bin/bash

# Test model on validation/train set with accuracy calculation

CHECKPOINT=${1:-"ckpts/phase1/best.pth"}
TEST_JSON=${2:-"data/train/train.json"}

python test.py \
  --ckpt $CHECKPOINT \
  --test_json $TEST_JSON \
  --video_dir data \
  --output_dir test_results \
  --batch_size 16 \
  --max_frames 16 \
  --calc_acc

echo ""
echo "Results saved to test_results/"
