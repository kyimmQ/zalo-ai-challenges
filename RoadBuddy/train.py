import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import numpy as np

from dataset import create_dataloaders, RoadBuddyDataset, collate_fn
from model import build_model


def train_one_epoch(model, loader, opt, sched, accel, epoch):
    model.train()
    loss_sum = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Ep {epoch}", disable=not accel.is_local_main_process)
    
    for i, batch in enumerate(pbar):
        frames = batch['frames']
        ids = batch['input_ids']
        mask = batch['attention_mask']
        answers = batch['answers']
        n_choices = batch['num_choices']
        
        logits = model(frames, ids, mask)
        
        choice_mask = torch.arange(4).expand(len(n_choices), 4).to(n_choices.device)
        choice_mask = choice_mask < n_choices.unsqueeze(1)
        logits = logits.masked_fill(~choice_mask, float('-inf'))
        
        loss = nn.CrossEntropyLoss()(logits, answers)
        
        accel.backward(loss)
        
        if (i + 1) % accel.gradient_accumulation_steps == 0:
            opt.step()
            sched.step()
            opt.zero_grad()
        
        preds = logits.argmax(dim=-1)
        correct += (preds == answers).sum().item()
        total += answers.size(0)
        loss_sum += loss.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{correct/total:.3f}'})
    
    return loss_sum / len(loader), correct / total


@torch.no_grad()
def eval_model(model, loader, accel):
    model.eval()
    loss_sum = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Eval", disable=not accel.is_local_main_process):
        frames = batch['frames']
        ids = batch['input_ids']
        mask = batch['attention_mask']
        answers = batch['answers']
        n_choices = batch['num_choices']
        
        logits = model(frames, ids, mask)
        
        choice_mask = torch.arange(4).expand(len(n_choices), 4).to(n_choices.device)
        choice_mask = choice_mask < n_choices.unsqueeze(1)
        logits = logits.masked_fill(~choice_mask, float('-inf'))
        
        loss = nn.CrossEntropyLoss()(logits, answers)
        
        preds = logits.argmax(dim=-1)
        correct += (preds == answers).sum().item()
        total += answers.size(0)
        loss_sum += loss.item()
    
    return loss_sum / len(loader), correct / total


def main(args):
    accel = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision
    )
    
    if accel.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    if args.val_json is None:
        full_ds = RoadBuddyDataset(args.train_json, args.video_dir, 
                                   max_frames=args.max_frames, frame_size=args.frame_size, mode='train')
        train_size = int(0.9 * len(full_ds))
        val_size = len(full_ds) - train_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size], 
                                        generator=torch.Generator().manual_seed(args.seed))
        
        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
        val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    else:
        train_loader, val_loader = create_dataloaders(
            args.train_json, args.val_json, args.video_dir, 
            args.batch_size, args.num_workers, args.max_frames, args.frame_size
        )
    
    model = build_model(
        vid_ckpt=args.vid_pretrained,
        freeze_vid=args.freeze_vid,
        freeze_txt=args.freeze_txt,
        fusion_mode=args.fusion_mode,
        n_frames=args.max_frames
    )
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
    
    params = []
    if not args.freeze_vid:
        params.append({'params': model.vid_enc.parameters(), 'lr': args.vid_lr})
    if not args.freeze_txt:
        params.append({'params': model.txt_enc.parameters(), 'lr': args.txt_lr})
    params.append({'params': list(model.fusion.parameters()) + list(model.head.parameters()), 'lr': args.lr})
    
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = args.warmup if args.warmup > 0 else int(0.1 * total_steps)
    
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    
    model, opt, train_loader, val_loader, sched = accel.prepare(
        model, opt, train_loader, val_loader, sched
    )
    
    if accel.is_main_process:
        print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset) if val_loader else 0}")
        print(f"BS: {args.batch_size}, Accum: {args.grad_accum}, Eff BS: {args.batch_size * args.grad_accum}")
        print(f"Epochs: {args.epochs}, Steps: {total_steps}, Warmup: {warmup_steps}")
    
    best_acc = 0.0
    patience_cnt = 0
    hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for ep in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, sched, accel, ep + 1)
        hist['train_loss'].append(train_loss)
        hist['train_acc'].append(train_acc)
        
        if accel.is_main_process:
            print(f"Ep {ep+1}/{args.epochs} | Train L: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        if val_loader:
            val_loss, val_acc = eval_model(model, val_loader, accel)
            hist['val_loss'].append(val_loss)
            hist['val_acc'].append(val_acc)
            
            if accel.is_main_process:
                print(f"Val L: {val_loss:.4f}, Acc: {val_acc:.4f}")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    patience_cnt = 0
                    
                    unwrapped = accel.unwrap_model(model)
                    torch.save({
                        'epoch': ep + 1,
                        'model_state_dict': unwrapped.state_dict(),
                        'opt_state_dict': opt.state_dict(),
                        'sched_state_dict': sched.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': val_loss
                    }, os.path.join(args.output_dir, 'best.pth'))
                    print(f"✓ Saved (acc: {val_acc:.4f})")
                else:
                    patience_cnt += 1
                    if args.early_stop > 0 and patience_cnt >= args.early_stop:
                        print(f"Early stop at epoch {ep+1}")
                        break
        
        if accel.is_main_process and (ep + 1) % args.save_every == 0:
            unwrapped = accel.unwrap_model(model)
            torch.save({
                'epoch': ep + 1,
                'model_state_dict': unwrapped.state_dict(),
                'opt_state_dict': opt.state_dict()
            }, os.path.join(args.output_dir, f'ep{ep+1}.pth'))
    
    if accel.is_main_process:
        with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
            json.dump(hist, f, indent=2)
        print(f"Done! Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    
    p.add_argument("--train_json", required=True)
    p.add_argument("--val_json", default=None)
    p.add_argument("--video_dir", required=True)
    p.add_argument("--output_dir", default="./ckpts")
    
    p.add_argument("--vid_pretrained", default=None)
    p.add_argument("--resume", default=None)
    
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--vid_lr", type=float, default=1e-6)
    p.add_argument("--txt_lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=-1)
    
    p.add_argument("--max_frames", type=int, default=16)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    
    p.add_argument("--freeze_vid", action="store_true")
    p.add_argument("--freeze_txt", action="store_true")
    p.add_argument("--fusion_mode", default="concat", choices=["concat", "attention"])
    
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--early_stop", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    
    args = p.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    main(args)
