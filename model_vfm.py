from pathlib import Path

import torch
import torch.nn as nn


class DINOv2FAS(nn.Module):
    """DINOv2-with-registers backbone with an FAS classification head."""

    def __init__(
        self,
        model_name="dinov2_vitb14_reg",
        num_classes=2,
        pretrained=True,
        dropout=0.2,
        feature_mode="cls_mean",
        dinov2_repo=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.feature_mode = feature_mode

        if dinov2_repo:
            repo = str(Path(dinov2_repo).expanduser().resolve())
            self.backbone = torch.hub.load(
                repo,
                model_name,
                source="local",
                pretrained=pretrained,
            )
        else:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=pretrained,
            )

        dim = int(self.backbone.embed_dim)
        if feature_mode == "cls":
            feature_dim = dim
        elif feature_mode == "cls_mean":
            feature_dim = dim * 2
        elif feature_mode == "cls_mean_std":
            feature_dim = dim * 3
        else:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def extract_features(self, x):
        feats = self.backbone.forward_features(x)
        cls = feats["x_norm_clstoken"]

        if self.feature_mode == "cls":
            return cls

        patches = feats["x_norm_patchtokens"]
        patch_mean = patches.mean(dim=1)

        if self.feature_mode == "cls_mean":
            return torch.cat([cls, patch_mean], dim=1)

        patch_std = patches.float().std(dim=1, unbiased=False).to(patches.dtype)
        return torch.cat([cls, patch_mean, patch_std], dim=1)

    def forward(self, x):
        return self.classifier(self.extract_features(x))

    def freeze_first_blocks(self, n):
        """Freeze the whole DINOv2 backbone, then unfreeze blocks from index n onward.

        For ViT-B/14 (12 blocks), n=11 means only the final encoder block is
        trainable. Patch embedding, positional/class/register tokens and the
        remaining backbone parameters stay frozen, preserving pretrained
        representations for cross-domain generalization.
        """
        n = max(0, int(n))
        blocks = list(self.backbone.blocks)

        if n <= 0:
            for p in self.backbone.parameters():
                p.requires_grad = True
            return

        for p in self.backbone.parameters():
            p.requires_grad = False

        for i, block in enumerate(blocks):
            if i >= n:
                for p in block.parameters():
                    p.requires_grad = True


def build_dinov2_fas(
    model_name="dinov2_vitb14_reg",
    num_classes=2,
    pretrained=True,
    dropout=0.2,
    feature_mode="cls_mean",
    dinov2_repo=None,
    freeze_first_blocks=0,
):
    model = DINOv2FAS(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        feature_mode=feature_mode,
        dinov2_repo=dinov2_repo,
    )
    if freeze_first_blocks:
        model.freeze_first_blocks(freeze_first_blocks)
    return model
