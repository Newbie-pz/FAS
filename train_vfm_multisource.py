import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from metrics import compute_metrics
from model_vfm import build_dinov2_fas
from vfm_data import (
    MICO_DATASETS,
    VFMImageDataset,
    aggregate_video_scores,
    build_domain_class_balanced_sampler,
    evenly_subsample_frames_per_video,
    scan_vfm_domain,
    split_train_val_by_group,
)



class FocalLoss(nn.Module):
    """Binary/multiclass focal loss over logits with optional label smoothing."""

    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def parse_args():
    p = argparse.ArgumentParser(
        description="High-performance multi-source FAS training with DINOv2 registers."
    )
    p.add_argument("--data_root", default="/root/Desktop/code/FAS/ProcessedData")
    p.add_argument("--target_dataset", required=True, choices=MICO_DATASETS)
    p.add_argument("--source_datasets", nargs="+", default=None)
    p.add_argument("--output_dir", default="outputs_week2_vfm")
    p.add_argument("--model_name", default="dinov2_vitb14_reg")
    p.add_argument("--dinov2_repo", default=None)
    p.add_argument("--feature_mode", choices=["cls", "cls_mean", "cls_mean_std"], default="cls_mean")
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--freeze_first_blocks", type=int, default=0)

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--head_lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--loss", choices=["ce", "focal"], default="focal")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--max_train_frames_per_video", type=int, default=20)
    p.add_argument("--max_val_frames_per_video", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model, args):
    groups = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "head_decay": [],
        "head_no_decay": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = name.startswith("classifier.")
        no_decay = param.ndim == 1 or name.endswith(".bias")
        prefix = "head" if is_head else "backbone"
        suffix = "no_decay" if no_decay else "decay"
        groups[f"{prefix}_{suffix}"].append(param)

    param_groups = []
    if groups["backbone_decay"]:
        param_groups.append({
            "params": groups["backbone_decay"],
            "lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
        })
    if groups["backbone_no_decay"]:
        param_groups.append({
            "params": groups["backbone_no_decay"],
            "lr": args.backbone_lr,
            "weight_decay": 0.0,
        })
    if groups["head_decay"]:
        param_groups.append({
            "params": groups["head_decay"],
            "lr": args.head_lr,
            "weight_decay": args.weight_decay,
        })
    if groups["head_no_decay"]:
        param_groups.append({
            "params": groups["head_no_decay"],
            "lr": args.head_lr,
            "weight_decay": 0.0,
        })

    return AdamW(param_groups)


def build_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def evaluate_epoch(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    y_true, y_score, video_ids = [], [], []

    with torch.no_grad():
        for x, y, _, _, _, vids in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(x)
                loss = criterion(logits, y)

            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            total_loss += loss.item() * x.size(0)
            y_true.extend(y.cpu().tolist())
            y_score.extend(probs.cpu().tolist())
            video_ids.extend(list(vids))

    frame_metrics = compute_metrics(y_true, y_score)
    _, video_true, video_score = aggregate_video_scores(y_true, y_score, video_ids)
    video_metrics = compute_metrics(video_true, video_score)

    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "frame": frame_metrics,
        "video": video_metrics,
        "num_frames": len(y_true),
        "num_videos": len(video_true),
    }


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    use_amp,
    grad_accum,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    y_true, y_score = [], []

    for step, (x, y, _, _, _, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(x)
            raw_loss = criterion(logits, y)
            loss = raw_loss / grad_accum

        scaler.scale(loss).backward()

        if step % grad_accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        probs = torch.softmax(logits.detach().float(), dim=1)[:, 1]
        total_loss += raw_loss.item() * x.size(0)
        y_true.extend(y.detach().cpu().tolist())
        y_score.extend(probs.cpu().tolist())

    metrics = compute_metrics(y_true, y_score)
    return total_loss / max(len(loader.dataset), 1), metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    sources = args.source_datasets or [
        d for d in MICO_DATASETS if d != args.target_dataset
    ]
    if args.target_dataset in sources:
        raise ValueError("Target dataset must not appear in source_datasets.")
    if len(sources) < 2:
        raise ValueError("Multi-source training requires at least two source domains.")

    train_samples, val_samples = [], []
    domain_summary = {}

    for domain_index, domain in enumerate(sources):
        all_samples = scan_vfm_domain(args.data_root, domain)
        train_s, val_s = split_train_val_by_group(
            all_samples,
            val_ratio=args.val_ratio,
            seed=args.seed + domain_index * 17,
        )

        raw_train_frames = len(train_s)
        raw_val_frames = len(val_s)

        train_s = evenly_subsample_frames_per_video(
            train_s,
            args.max_train_frames_per_video,
        )
        val_s = evenly_subsample_frames_per_video(
            val_s,
            args.max_val_frames_per_video,
        )

        train_samples.extend(train_s)
        val_samples.extend(val_s)

        domain_summary[domain] = {
            "raw_total_frames": len(all_samples),
            "raw_train_frames": raw_train_frames,
            "raw_val_frames": raw_val_frames,
            "selected_train_frames": len(train_s),
            "selected_val_frames": len(val_s),
            "train_groups": len({x[2] for x in train_s}),
            "val_groups": len({x[2] for x in val_s}),
            "train_videos": len({x[4] for x in train_s}),
            "val_videos": len({x[4] for x in val_s}),
        }

    if any(x[3] == args.target_dataset for x in train_samples + val_samples):
        raise RuntimeError("Target-domain leakage detected.")

    sampler = build_domain_class_balanced_sampler(train_samples)
    train_ds = VFMImageDataset(
        train_samples,
        train=True,
        image_size=args.image_size,
    )
    val_ds = VFMImageDataset(
        val_samples,
        train=False,
        image_size=args.image_size,
    )

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(args.batch_size, 32),
        shuffle=False,
        **loader_kwargs,
    )

    print("=" * 72)
    print("DINOv2-Reg multi-source FAS training")
    print(f"Target domain : {args.target_dataset}")
    print(f"Source domains: {', '.join(sources)}")
    print(f"Train frames  : {len(train_ds)}")
    print(f"Val frames    : {len(val_ds)}")
    print(f"Model         : {args.model_name}")
    print(f"Feature mode  : {args.feature_mode}")
    print(f"AMP           : {use_amp}")
    print("=" * 72)

    model = build_dinov2_fas(
        model_name=args.model_name,
        pretrained=True,
        dropout=args.dropout,
        feature_mode=args.feature_mode,
        dinov2_repo=args.dinov2_repo,
        freeze_first_blocks=args.freeze_first_blocks,
    ).to(device)

    if args.loss == "focal":
        criterion = FocalLoss(
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = make_optimizer(model, args)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    scheduler = build_scheduler(
        optimizer,
        total_steps=max(1, updates_per_epoch * args.epochs),
        warmup_ratio=args.warmup_ratio,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out = Path(args.output_dir) / args.target_dataset
    out.mkdir(parents=True, exist_ok=True)

    split_record = {
        "protocol": "MICO-style multi-source DG; target domain excluded from training and validation",
        "target_dataset": args.target_dataset,
        "source_datasets": sources,
        "domain_summary": domain_summary,
        "train_frames": len(train_samples),
        "val_frames": len(val_samples),
        "train_videos": len({x[4] for x in train_samples}),
        "val_videos": len({x[4] for x in val_samples}),
        "domain_class_counts_after_subsampling": {
            f"{domain}|{label}": count
            for (domain, label), count in Counter(
                (x[3], x[1]) for x in train_samples
            ).items()
        },
        "sampler": "domain-class balanced weighted sampling",
        "loss": args.loss,
        "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
        "max_train_frames_per_video": args.max_train_frames_per_video,
        "max_val_frames_per_video": args.max_val_frames_per_video,
    }
    (out / "split_summary.json").write_text(
        json.dumps(split_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    history = []
    best_auc = -1.0
    best_eer = float("inf")
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            use_amp,
            args.grad_accum,
        )
        val_result = evaluate_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
        )

        val_auc = float(val_result["video"]["auc"])
        val_eer = float(val_result["video"]["eer"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_result["loss"],
            "val_frame_auc": val_result["frame"]["auc"],
            "val_frame_eer": val_result["frame"]["eer"],
            "val_video_auc": val_auc,
            "val_video_eer": val_eer,
            "val_videos": val_result["num_videos"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        improved = (
            val_auc > best_auc + 1e-6
            or (abs(val_auc - best_auc) <= 1e-6 and val_eer < best_eer)
        )
        if improved:
            best_auc = val_auc
            best_eer = val_eer
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "source_datasets": sources,
                    "target_dataset": args.target_dataset,
                    "best_val_video_auc": best_auc,
                    "best_val_video_eer": best_eer,
                },
                out / "best.pth",
            )
            print(
                f"[BEST] epoch={epoch} "
                f"val_video_auc={best_auc:.6f} "
                f"val_video_eer={best_eer:.6f}"
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    (out / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    final_summary = {
        "target_dataset": args.target_dataset,
        "source_datasets": sources,
        "best_val_video_auc": best_auc,
        "best_val_video_eer": best_eer,
        "epochs_ran": len(history),
        "checkpoint": str(out / "best.pth"),
    }
    (out / "training_summary.json").write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Training complete:", final_summary)


if __name__ == "__main__":
    main()
