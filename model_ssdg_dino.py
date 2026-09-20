from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from week3_common import CosineClassifier


class DINOv2SSDG(nn.Module):
    """DINOv2-Reg adapted with the core SSDG representation design."""

    def __init__(
        self,
        model_name="dinov2_vitb14_reg",
        pretrained=True,
        dinov2_repo=None,
        freeze_first_blocks=10,
        embed_dim=512,
        dropout=0.5,
        cosine_scale=16.0,
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
                str(repo),
                model_name,
                source="local",
                pretrained=pretrained,
            )
        else:
            print("[DINOv2] Local cache not found; loading from GitHub.")
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=pretrained,
            )

        backbone_dim = int(self.backbone.embed_dim)
        self.embedder = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = CosineClassifier(
            embed_dim,
            num_classes=2,
            scale=cosine_scale,
        )
        self.feature_dim = int(embed_dim)
        self.configure_backbone(freeze_first_blocks)

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

    def forward(self, x, return_features=False):
        feats = self.backbone.forward_features(x)
        cls = feats["x_norm_clstoken"]
        embedding = self.embedder(cls)
        embedding = F.normalize(embedding, dim=1)
        logits = self.classifier(embedding)

        if return_features:
            return logits, embedding
        return logits
