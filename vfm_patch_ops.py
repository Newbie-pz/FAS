import random

import torch
import torch.nn.functional as F


def apply_patch_data_augmentation(
    images,
    labels,
    patch_size=14,
    apply_p=0.5,
    replace_ratio=0.25,
):
    """PDA: replace random patches in spoof images with patches from live images.

    Returns mixed images and per-patch labels. Patch labels track whether each
    local region originates from Live (1) or Spoof (0).
    """
    b, c, h, w = images.shape
    gh, gw = h // patch_size, w // patch_size
    num_patches = gh * gw

    patch_labels = labels[:, None].repeat(1, num_patches).clone()
    live_idx = torch.where(labels == 1)[0]
    spoof_idx = torch.where(labels == 0)[0]

    if len(live_idx) == 0 or len(spoof_idx) == 0:
        return images, patch_labels

    mixed = images.clone()
    for s in spoof_idx.tolist():
        if random.random() >= apply_p:
            continue
        donor = int(live_idx[torch.randint(len(live_idx), (1,), device=images.device)])
        mask = torch.rand((gh, gw), device=images.device) < replace_ratio
        if not mask.any():
            mask[
                torch.randint(gh, (1,), device=images.device),
                torch.randint(gw, (1,), device=images.device),
            ] = True

        for iy, ix in mask.nonzero(as_tuple=False).tolist():
            y0, y1 = iy * patch_size, (iy + 1) * patch_size
            x0, x1 = ix * patch_size, (ix + 1) * patch_size
            mixed[s, :, y0:y1, x0:x1] = images[donor, :, y0:y1, x0:x1]
            patch_labels[s, iy * gw + ix] = 1

    return mixed, patch_labels


def focal_loss(logits, targets, gamma=2.0, reduction="mean"):
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** gamma) * ce
    if reduction == "none":
        return loss
    return loss.mean()


def attention_weighted_patch_loss(
    patch_logits,
    patch_labels,
    patch_attention,
    gamma=2.0,
):
    b, n, _ = patch_logits.shape
    raw = focal_loss(
        patch_logits.reshape(b * n, 2),
        patch_labels.reshape(b * n),
        gamma=gamma,
        reduction="none",
    ).reshape(b, n)

    weights = patch_attention.detach().float()
    weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-6)
    return (raw.float() * weights).mean()
