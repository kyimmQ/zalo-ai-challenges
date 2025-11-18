import os
import argparse
import json
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RoadBuddyDataset, collate_fn
from model import build_model


@torch.no_grad()
def inference(model, loader, device):
    model.eval()
    predictions = {}
    
    for batch in tqdm(loader, desc="Inference"):
        frames = batch['frames'].to(device)
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        n_choices = batch['num_choices']
        qids = batch['question_ids']
        
        logits = model(frames, ids, mask)
        
        choice_mask = torch.arange(4).expand(len(n_choices), 4).to(device)
        choice_mask = choice_mask < n_choices.unsqueeze(1)
        logits = logits.masked_fill(~choice_mask, float('-inf'))
        
        preds = logits.argmax(dim=-1)
        
        for qid, p in zip(qids, preds):
            predictions[qid] = chr(ord('A') + p.item())
    
    return predictions


def write_csv(preds, out_path, sample_csv=None):
    if sample_csv and os.path.exists(sample_csv):
        with open(sample_csv, encoding='utf-8') as f:
            qids = [row['id'] for row in csv.DictReader(f)]
    else:
        qids = sorted(preds.keys())
    
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'answer'])
        for qid in qids:
            w.writerow([qid, preds.get(qid, 'A')])
    
    print(f"Submission saved: {out_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    ckpt = torch.load(args.ckpt, map_location='cpu')
    
    model = build_model(n_frames=args.max_frames)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    if 'val_acc' in ckpt:
        print(f"Model val acc: {ckpt['val_acc']:.4f}")
    
    test_ds = RoadBuddyDataset(args.test_json, args.video_dir,
                               max_frames=args.max_frames, frame_size=args.frame_size, mode='test')
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    
    print(f"Test samples: {len(test_ds)}")
    
    preds = inference(model, test_loader, device)
    
    print(f"Generated {len(preds)} predictions")
    
    dist = {c: list(preds.values()).count(c) for c in ['A', 'B', 'C', 'D']}
    print("Distribution:")
    for c in ['A', 'B', 'C', 'D']:
        cnt = dist.get(c, 0)
        print(f"  {c}: {cnt} ({cnt/len(preds)*100:.1f}%)")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    csv_path = os.path.join(args.output_dir, 'submission.csv')
    write_csv(preds, csv_path, args.sample_csv)
    
    json_path = os.path.join(args.output_dir, 'predictions.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {json_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_json", required=True)
    p.add_argument("--video_dir", required=True)
    p.add_argument("--output_dir", default="./submissions")
    p.add_argument("--sample_csv", default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_frames", type=int, default=16)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    
    main(p.parse_args())
