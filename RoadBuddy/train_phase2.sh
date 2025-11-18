#!/bin/bash

# Phase 2: Fine-tune all layers (optional)
# Run this after phase1 completes if you want higher accuracy

python train.py \
  --train_json data/train/train.json \
  --video_dir data \
  --output_dir ckpts/phase2 \
  --resume ckpts/phase1/best.pth \
  --batch_size 8 \
  --epochs 10 \
  --lr 5e-5 \
  --vid_lr 1e-6 \
  --txt_lr 1e-5 \
  --max_frames 16 \
  --grad_accum 4 \
  --mixed_precision fp16 \
  --early_stop 5 \
  --save_every 5
