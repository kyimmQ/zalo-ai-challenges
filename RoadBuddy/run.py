#!/usr/bin/env python
import sys
import os

cmds = {
    'train': 'python train_simple.py --train_json data/train/train.json --video_dir data --output_dir ckpts/exp1 --batch_size 8 --epochs 5 --freeze_vid --freeze_txt',
    'test': 'python test_simple.py --ckpt ckpts/exp1/best.pth --test_json data/train/train.json --video_dir data --output_dir test_out --calc_acc',
    'submit': 'python submit_simple.py --ckpt ckpts/exp1/best.pth --test_json data/public_test/public_test.json --video_dir data --output_dir submissions',
    'docker': 'docker build -t roadbuddy . && docker save roadbuddy | gzip > roadbuddy.tar.gz',
}

if len(sys.argv) < 2 or sys.argv[1] not in cmds:
    print("Usage: python run.py <command>")
    print("\nAvailable commands:")
    for name, cmd in cmds.items():
        print(f"  {name:10s} - {cmd[:60]}...")
    sys.exit(1)

os.system(cmds[sys.argv[1]])
