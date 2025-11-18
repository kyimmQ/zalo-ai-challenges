import os
import sys
sys.path.append('simple-tad')

import torch
import torch.nn as nn
from transformers import AutoModel

from modeling_finetune import vit_base_patch16_224, vit_large_patch16_224


class VideoEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained_path=None,
                 num_frames=16, tubelet_size=2, drop_path_rate=0.1, 
                 use_checkpoint=False, freeze=False):
        super().__init__()
        
        if 'base' in model_name:
            self.backbone = vit_base_patch16_224(
                pretrained=False, num_classes=2, all_frames=num_frames,
                tubelet_size=tubelet_size, drop_path_rate=drop_path_rate,
                use_checkpoint=use_checkpoint, use_mean_pooling=True, use_flash_attn=False
            )
        elif 'large' in model_name:
            self.backbone = vit_large_patch16_224(
                pretrained=False, num_classes=2, all_frames=num_frames,
                tubelet_size=tubelet_size, drop_path_rate=drop_path_rate,
                use_checkpoint=use_checkpoint, use_mean_pooling=True, use_flash_attn=False
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        if pretrained_path and os.path.exists(pretrained_path):
            ckpt = torch.load(pretrained_path, map_location='cpu')
            state = ckpt.get('model', ckpt.get('state_dict', ckpt))
            state = {k.replace('module.', ''): v for k, v in state.items()}
            self.backbone.load_state_dict(state, strict=False)
        
        self.feat_dim = self.backbone.embed_dim
        
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
    
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if hasattr(self.backbone, 'fc_norm'):
            return self.backbone.fc_norm(feats.mean(1))
        return feats.mean(1)


class TextEncoder(nn.Module):
    def __init__(self, model_name='vinai/phobert-base', freeze=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.feat_dim = self.bert.config.hidden_size
        
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        bs, n_choices, seq_len = input_ids.shape
        ids = input_ids.view(-1, seq_len)
        mask = attention_mask.view(-1, seq_len)
        
        out = self.bert(input_ids=ids, attention_mask=mask)
        cls_emb = out.last_hidden_state[:, 0]
        return cls_emb.view(bs, n_choices, -1)


class FusionModule(nn.Module):
    def __init__(self, vid_dim, txt_dim, hidden=512, dropout=0.1, mode='concat'):
        super().__init__()
        self.mode = mode
        
        if mode == 'concat':
            self.net = nn.Sequential(
                nn.Linear(vid_dim + txt_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.LayerNorm(hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.out_dim = hidden // 2
        elif mode == 'attention':
            self.vid_proj = nn.Linear(vid_dim, hidden)
            self.txt_proj = nn.Linear(txt_dim, hidden)
            self.attn = nn.MultiheadAttention(hidden, num_heads=8, dropout=dropout)
            self.net = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.LayerNorm(hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.out_dim = hidden // 2
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")
    
    def forward(self, vid_feat, txt_feat):
        bs, n_choices, _ = txt_feat.shape
        vid_feat = vid_feat.unsqueeze(1).expand(-1, n_choices, -1)
        
        if self.mode == 'concat':
            return self.net(torch.cat([vid_feat, txt_feat], dim=-1))
        else:
            v = self.vid_proj(vid_feat).transpose(0, 1)
            t = self.txt_proj(txt_feat).transpose(0, 1)
            attn_out, _ = self.attn(t, v, v)
            return self.net(attn_out.transpose(0, 1))


class RoadBuddyModel(nn.Module):
    def __init__(self, vid_model='vit_base_patch16_224', vid_pretrained=None,
                 txt_model='vinai/phobert-base', n_frames=16, tubelet_sz=2,
                 fusion_mode='concat', fusion_hidden=512, drop=0.1,
                 freeze_vid=False, freeze_txt=False):
        super().__init__()
        
        self.vid_enc = VideoEncoder(
            model_name=vid_model, pretrained_path=vid_pretrained,
            num_frames=n_frames, tubelet_size=tubelet_sz, freeze=freeze_vid
        )
        self.txt_enc = TextEncoder(model_name=txt_model, freeze=freeze_txt)
        self.fusion = FusionModule(
            vid_dim=self.vid_enc.feat_dim, txt_dim=self.txt_enc.feat_dim,
            hidden=fusion_hidden, dropout=drop, mode=fusion_mode
        )
        self.head = nn.Linear(self.fusion.out_dim, 1)
    
    def forward(self, frames, input_ids, attn_mask):
        vid_feat = self.vid_enc(frames)
        txt_feat = self.txt_enc(input_ids, attn_mask)
        fused = self.fusion(vid_feat, txt_feat)
        return self.head(fused).squeeze(-1)


def build_model(vid_ckpt=None, freeze_vid=True, freeze_txt=True, 
                fusion_mode='concat', n_frames=16):
    return RoadBuddyModel(
        vid_model='vit_base_patch16_224', vid_pretrained=vid_ckpt,
        txt_model='vinai/phobert-base', n_frames=n_frames, tubelet_sz=2,
        fusion_mode=fusion_mode, fusion_hidden=512, drop=0.1,
        freeze_vid=freeze_vid, freeze_txt=freeze_txt
    )


if __name__ == "__main__":
    m = build_model(freeze_vid=False, freeze_txt=False, n_frames=8)
    frames = torch.randn(2, 3, 8, 224, 224)
    ids = torch.randint(0, 1000, (2, 4, 128))
    mask = torch.ones(2, 4, 128)
    out = m(frames, ids, mask)
    print(f"Output shape: {out.shape}, expected: (2, 4)")
    print(f"Params: {sum(p.numel() for p in m.parameters()):,}")
