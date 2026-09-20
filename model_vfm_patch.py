from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2PatchFAS(nn.Module):
    """DINOv2-Reg with global CLS and local patch classifiers."""

    def __init__(
        self,
        model_name="dinov2_vitb14_reg",
        pretrained=True,
        dinov2_repo=None,
        dropout=0.2,
        freeze_first_blocks=10,
        patch_scale=16.0,
    ):
        super().__init__()

        if dinov2_repo:
            repo = Path(dinov2_repo).expanduser().resolve()
        else:
            cached = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
            repo = cached if cached.exists() else None

        if repo is not None:
            print(f"[DINOv2] Using local Torch Hub repo: {repo}")
            self.backbone = torch.hub.load(
                str(repo), model_name, source="local", pretrained=pretrained
            )
        else:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=pretrained
            )

        self.embed_dim = int(self.backbone.embed_dim)
        self.num_register_tokens = int(getattr(self.backbone, "num_register_tokens", 0))
        self.patch_scale = float(patch_scale)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, 2),
        )
        self.patch_norm = nn.LayerNorm(self.embed_dim)
        self.patch_weight = nn.Parameter(torch.empty(2, self.embed_dim))
        self.patch_bias = nn.Parameter(torch.zeros(2))
        nn.init.trunc_normal_(self.patch_weight, std=0.02)

        self._last_qkv = None
        self._qkv_hook = self.backbone.blocks[-1].attn.qkv.register_forward_hook(
            self._capture_qkv
        )

        self.configure_backbone(freeze_first_blocks)

    def _capture_qkv(self, module, inputs, output):
        self._last_qkv = output

    def configure_backbone(self, freeze_first_blocks):
        n = int(freeze_first_blocks)
        if n <= 0:
            for p in self.backbone.parameters():
                p.requires_grad = True
            return

        for p in self.backbone.parameters():
            p.requires_grad = False

        blocks = list(self.backbone.blocks)
        for i, block in enumerate(blocks):
            if i >= n:
                for p in block.parameters():
                    p.requires_grad = True

    def _class_to_patch_attention(self, num_patches):
        qkv = self._last_qkv
        if qkv is None:
            return None

        b, n, three_d = qkv.shape
        heads = int(self.backbone.blocks[-1].attn.num_heads)
        head_dim = three_d // 3 // heads
        qkv = qkv.reshape(b, n, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        scale = float(getattr(self.backbone.blocks[-1].attn, "scale", head_dim ** -0.5))
        cls_q = q[:, :, 0:1, :]
        scores = (cls_q * scale) @ k.transpose(-2, -1)
        attn = scores.softmax(dim=-1).mean(dim=1).squeeze(1)

        patch_start = 1 + self.num_register_tokens
        patch_attn = attn[:, patch_start:patch_start + num_patches]
        patch_attn = patch_attn / patch_attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return patch_attn

    def patch_logits(self, patch_tokens):
        x = self.patch_norm(patch_tokens)
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.patch_weight, dim=-1)
        return self.patch_scale * F.linear(x, w, self.patch_bias)

    def forward(self, x, return_patch=False):
        self._last_qkv = None
        feats = self.backbone.forward_features(x)
        cls = feats["x_norm_clstoken"]
        patches = feats["x_norm_patchtokens"]
        global_logits = self.classifier(cls)

        if not return_patch:
            return global_logits

        local_logits = self.patch_logits(patches)
        attention = self._class_to_patch_attention(patches.shape[1])
        if attention is None:
            attention = torch.full(
                (patches.shape[0], patches.shape[1]),
                1.0 / patches.shape[1],
                device=patches.device,
                dtype=patches.dtype,
            )
        return {
            "global_logits": global_logits,
            "patch_logits": local_logits,
            "patch_attention": attention,
        }
