import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import sys
sys.path.append('simple-tad')
from video_transforms import Compose, Resize, Normalize, CenterCrop, RandomCrop
from volume_transforms import ClipToTensor


class RoadBuddyDataset(Dataset):
    def __init__(self, json_path, video_dir, tokenizer_name='vinai/phobert-base',
                 max_frames=16, sampling_rate=1, frame_size=224, mode='train'):
        with open(json_path, encoding='utf-8') as f:
            self.samples = json.load(f)['data']
        
        self.vid_dir = video_dir
        self.n_frames = max_frames
        self.sample_rate = sampling_rate
        self.size = frame_size
        self.mode = mode
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        
        crop = RandomCrop((frame_size, frame_size)) if mode == 'train' else CenterCrop((frame_size, frame_size))
        self.transform = Compose([
            Resize(256, interpolation='bilinear'),
            crop,
            ClipToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def _load_vid(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video not found: {path}")
        
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            raise ValueError(f"Empty video: {path}")
        
        indices = self._sample_indices(total)
        frames = []
        
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        
        while len(frames) < self.n_frames:
            frames.append(frames[-1].copy() if frames else np.zeros((self.size, self.size, 3), dtype=np.uint8))
        
        return frames[:self.n_frames]
    
    def _sample_indices(self, total):
        if total <= self.n_frames * self.sample_rate:
            return set(np.linspace(0, total - 1, self.n_frames, dtype=int))
        
        idx = np.arange(0, total, self.sample_rate)
        if len(idx) > self.n_frames:
            step = len(idx) / self.n_frames
            idx = [idx[int(i * step)] for i in range(self.n_frames)]
        else:
            idx = np.linspace(0, total - 1, self.n_frames, dtype=int)
        return set(idx)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        vid_path = os.path.join(self.vid_dir, s['video_path'])
        
        try:
            frames = self._load_vid(vid_path)
            frames = self.transform(frames)
        except Exception as e:
            frames = torch.zeros(3, self.n_frames, self.size, self.size)
        
        q = s['question']
        choices = s['choices']
        n_choices = len(choices)
        
        ans_idx = None
        if 'answer' in s:
            ans_idx = ord(s['answer'][0].upper()) - ord('A')
        
        texts = [f"{q} [SEP] {c}" for c in choices]
        while len(texts) < 4:
            texts.append(texts[-1])
        
        tok_out = self.tok(texts, padding='max_length', truncation=True, 
                           max_length=128, return_tensors='pt')
        
        out = {
            'frames': frames,
            'input_ids': tok_out['input_ids'],
            'attention_mask': tok_out['attention_mask'],
            'num_choices': n_choices,
            'question_id': s['id']
        }
        
        if ans_idx is not None:
            out['answer'] = ans_idx
        
        return out


def collate_fn(batch):
    out = {
        'frames': torch.stack([x['frames'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'num_choices': torch.tensor([x['num_choices'] for x in batch]),
        'question_ids': [x['question_id'] for x in batch]
    }
    if 'answer' in batch[0]:
        out['answers'] = torch.tensor([x['answer'] for x in batch])
    return out


def create_dataloaders(train_json, val_json, video_dir, batch_size=8,
                       num_workers=4, max_frames=16, frame_size=224,
                       tokenizer_name='vinai/phobert-base'):
    train_ds = RoadBuddyDataset(train_json, video_dir, tokenizer_name,
                                max_frames, 1, frame_size, 'train')
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    
    val_loader = None
    if val_json and os.path.exists(val_json):
        val_ds = RoadBuddyDataset(val_json, video_dir, tokenizer_name,
                                  max_frames, 1, frame_size, 'val')
        val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    
    return train_loader, val_loader


if __name__ == "__main__":
    ds = RoadBuddyDataset("data/train/train.json", "data", max_frames=8, mode='train')
    print(f"Dataset size: {len(ds)}")
    
    s = ds[0]
    print(f"Keys: {s.keys()}")
    print(f"Frames: {s['frames'].shape}, IDs: {s['input_ids'].shape}")
    print(f"Choices: {s['num_choices']}, Answer: {s.get('answer', 'N/A')}")
    
    loader, _ = create_dataloaders("data/train/train.json", None, "data", batch_size=2, num_workers=0)
    b = next(iter(loader))
    print(f"\nBatch frames: {b['frames'].shape}, IDs: {b['input_ids'].shape}")
    print(f"Answers: {b.get('answers', 'N/A')}")
