import os
import sys
import argparse
import json
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RoadBuddyDataset, collate_fn
from model import build_model


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    results = {}
    
    for batch in tqdm(loader, desc="Inference"):
        frames = batch['frames'].to(device)
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        n_choices = batch['num_choices']
        qids = batch['question_ids']
        
        logits = model(frames, ids, mask)
        
        m = torch.arange(4).expand(len(n_choices), 4).to(device)
        m = m < n_choices.unsqueeze(1)
        logits = logits.masked_fill(~m, float('-inf'))
        
        preds = logits.argmax(dim=-1)
        
        for qid, p in zip(qids, preds):
            results[qid] = chr(ord('A') + p.item())
    
    return results


def save_submission(preds, out_csv, sample_csv=None):
    if sample_csv and os.path.exists(sample_csv):
        with open(sample_csv, encoding='utf-8') as f:
            qids = [row['id'] for row in csv.DictReader(f)]
    else:
        qids = sorted(preds.keys())
    
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'answer'])
        for qid in qids:
            w.writerow([qid, preds.get(qid, 'A')])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_json", required=True, help="Path to test JSON")
    p.add_argument("--video_dir", required=True, help="Root video directory")
    p.add_argument("--output_csv", required=True, help="Output submission CSV")
    p.add_argument("--checkpoint", default="best_model.pth", help="Model checkpoint")
    p.add_argument("--sample_csv", default=None, help="Sample submission for ordering")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_frames", type=int, default=16)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    
    args = p.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    
    model = build_model(n_frames=args.max_frames)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    if 'val_acc' in ckpt:
        print(f"Model validation accuracy: {ckpt['val_acc']:.4f}")
    
    print(f"Loading test data from: {args.test_json}")
    test_ds = RoadBuddyDataset(
        args.test_json, args.video_dir,
        max_frames=args.max_frames,
        frame_size=args.frame_size,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Number of test samples: {len(test_ds)}")
    
    predictions = run_inference(model, test_loader, device)
    
    print(f"Generated predictions for {len(predictions)} questions")
    
    dist = {}
    for ans in predictions.values():
        dist[ans] = dist.get(ans, 0) + 1
    
    print("\nPrediction distribution:")
    for c in ['A', 'B', 'C', 'D']:
        cnt = dist.get(c, 0)
        pct = cnt / len(predictions) * 100 if predictions else 0
        print(f"  {c}: {cnt} ({pct:.1f}%)")
    
    save_submission(predictions, args.output_csv, args.sample_csv)
    print(f"\nSubmission saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
