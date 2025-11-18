import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RoadBuddyDataset, collate_fn
from model import build_model


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    preds, qids, all_logits = [], [], []
    
    for batch in tqdm(loader, desc="Testing"):
        frames = batch['frames'].to(device)
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        n_choices = batch['num_choices']
        q_ids = batch['question_ids']
        
        logits = model(frames, ids, mask)
        
        choice_mask = torch.arange(4).expand(len(n_choices), 4).to(device)
        choice_mask = choice_mask < n_choices.unsqueeze(1)
        logits = logits.masked_fill(~choice_mask, float('-inf'))
        
        pred = logits.argmax(dim=-1)
        
        preds.extend(pred.cpu().numpy())
        qids.extend(q_ids)
        all_logits.extend(logits.cpu().numpy())
    
    return preds, qids, all_logits


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
    
    preds, qids, logits = test(model, test_loader, device)
    
    results = []
    for qid, p, l in zip(qids, preds, logits):
        results.append({
            'question_id': qid,
            'prediction': chr(ord('A') + p),
            'prediction_idx': int(p),
            'logits': l.tolist()
        })
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    out_path = os.path.join(args.output_dir, 'predictions.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")
    
    if args.calc_acc:
        with open(args.test_json, encoding='utf-8') as f:
            labels = []
            for s in json.load(f)['data']:
                if 'answer' in s:
                    labels.append(ord(s['answer'][0].upper()) - ord('A'))
            
            if labels:
                acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
                print(f"Accuracy: {acc:.4f} ({sum(p == l for p, l in zip(preds, labels))}/{len(labels)})")
                
                with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
                    f.write(f"Accuracy: {acc:.4f}\n")
    
    dist = {i: preds.count(i) for i in range(4)}
    print("Prediction distribution:")
    for i, cnt in dist.items():
        print(f"  {chr(ord('A') + i)}: {cnt} ({cnt/len(preds)*100:.1f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_json", required=True)
    p.add_argument("--video_dir", required=True)
    p.add_argument("--output_dir", default="./test_out")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_frames", type=int, default=16)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--calc_acc", action="store_true")
    
    main(p.parse_args())
