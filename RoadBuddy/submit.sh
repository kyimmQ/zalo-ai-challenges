#!/bin/bash

# Generate submission for public test set

CHECKPOINT=${1:-"ckpts/phase1/best.pth"}

python submit.py \
  --ckpt $CHECKPOINT \
  --test_json data/public_test/public_test.json \
  --video_dir data \
  --output_dir submissions \
  --sample_csv data/public_test/public_test_sample_submission.csv \
  --batch_size 16 \
  --max_frames 16

echo ""
echo "Submission saved to submissions/submission.csv"
echo "Ready to submit to portal!"
