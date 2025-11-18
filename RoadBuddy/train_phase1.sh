#!/bin/bash

# Phase 1: Train with frozen encoders
# This is the recommended approach for initial training

python train.py \
  --train_json data/train/train.json \
  --video_dir data \
  --output_dir ckpts/phase1 \
  --batch_size 16 \
  --epochs 15 \
  --lr 5e-4 \
  --max_frames 16 \
  --freeze_vid \
  --freeze_txt \
  --grad_accum 2 \
  --mixed_precision fp16 \
  --early_stop 3 \
  --save_every 5
