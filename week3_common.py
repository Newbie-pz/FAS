import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from torchvision.models import ResNet18_Weights, resnet18

from vfm_data import aggregate_video_scores
from metrics import compute_metrics


class DomainClassBatchSampler(Sampler):
    """Balanced batches containing every (domain, class) stratum."""

    def __init__(self, samples, batch_size=24, seed=42):
        self.samples = samples
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        groups = defaultdict(list)
        for i, sample in enumerate(samples):
            domain, label = sample[3], int(sample[1])
            groups[(domain, label)].append(i)

        self.keys = sorted(groups)
        self.groups = groups
        if len(self.keys) == 0:
            raise RuntimeError("No samples for balanced batch sampler.")
        if self.batch_size % len(self.keys) != 0:
            raise ValueError(
                f"batch_size={self.batch_size} must be divisible by "
                f"{len(self.keys)} domain-class strata."
            )
        self.per_group = self.batch_size // len(self.keys)
        self.num_batches = max(1, math.ceil(len(samples) / self.batch_size))
        self.epoch = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        pools = {}
        ptr = {}
        for key in self.keys:
            arr = list(self.groups[key])
            rng.shuffle(arr)
            pools[key] = arr
            ptr[key] = 0

        for _ in range(self.num_batches):
            batch = []
            for key in self.keys:
                arr = pools[key]
                for _ in range(self.per_group):
                    if ptr[key] >= len(arr):
                        rng.shuffle(arr)
                        ptr[key] = 0
                    batch.append(arr[ptr[key]])
                    ptr[key] += 1
            rng.shuffle(batch)
            yield batch


class ResNet18FeatureFAS(nn.Module):
    def __init__(self, pretrained=True, dropout=0.2):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet18(weights=weights)
        self.feature_dim = m.fc.in_features
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 2),
        )

    def forward(self, x, return_features=False):
        feat = self.backbone(x).flatten(1)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = float(coeff)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coeff * grad_output, None


def grad_reverse(x, coeff=1.0):
    return GradientReverse.apply(x, coeff)


class CosineClassifier(nn.Module):
    def __init__(self, in_dim, num_classes=2, scale=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.normal_(self.weight, std=0.01)
        self.scale = float(scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(x, w)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, num_domains, hidden=512, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, x, grl_coeff=1.0):
        return self.net(grad_reverse(x, grl_coeff))


def batch_all_triplet_loss(embeddings, labels, margin=0.1):
    """Batch-all triplet loss matching the core SSDG asymmetric grouping idea."""
    embeddings = F.normalize(embeddings, dim=1)
    dist = torch.cdist(embeddings, embeddings, p=2)
    n = labels.shape[0]

    eye = torch.eye(n, device=labels.device, dtype=torch.bool)
    same = labels[:, None].eq(labels[None, :])
    pos_mask = same & (~eye)
    neg_mask = ~same

    ap = dist[:, :, None]
    an = dist[:, None, :]
    loss = ap - an + float(margin)

    mask = pos_mask[:, :, None] & neg_mask[:, None, :]
    valid = loss[mask]
    if valid.numel() == 0:
        return embeddings.sum() * 0.0
    valid = F.relu(valid)
    hard = valid[valid > 1e-12]
    if hard.numel() == 0:
        return valid.mean() * 0.0
    return hard.mean()


def compute_video_metrics_by_domain(y_true, y_score, video_ids, domains):
    result = {}
    unique_domains = sorted(set(domains))

    _, vy, vs = aggregate_video_scores(y_true, y_score, video_ids)
    overall = compute_metrics(vy, vs)

    per_domain = {}
    for domain in unique_domains:
        idx = [i for i, d in enumerate(domains) if d == domain]
        dy = [y_true[i] for i in idx]
        ds = [y_score[i] for i in idx]
        dv = [video_ids[i] for i in idx]
        _, dvy, dvs = aggregate_video_scores(dy, ds, dv)
        per_domain[domain] = compute_metrics(dvy, dvs)

    aucs = [
        float(m["auc"])
        for m in per_domain.values()
        if np.isfinite(float(m["auc"]))
    ]
    result["overall"] = overall
    result["per_domain"] = per_domain
    result["worst_domain_auc"] = min(aucs) if aucs else float("nan")
    result["mean_domain_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    return result
