import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from metrics import compute_metrics
from vfm_data import (
    MICO_DATASETS,
    VFMImageDataset,
    aggregate_video_scores,
    evenly_subsample_frames_per_video,
    scan_vfm_domain,
    split_train_val_by_group,
)
from week3_common import (
    DomainClassBatchSampler,
    ResNet18FeatureFAS,
    compute_video_metrics_by_domain,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/root/Desktop/code/FAS/ProcessedData")
    p.add_argument("--target_dataset", required=True, choices=MICO_DATASETS)
    p.add_argument("--output_dir", default="outputs_week3_three_routes/route1_resnet")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--max_frames_per_video", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, use_amp):
    model.eval()
    y_true, y_score, videos, domains = [], [], [], []
    with torch.no_grad():
        for x, y, _, _, domain, video in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(x)
            score = torch.softmax(logits.float(), dim=1)[:, 1]
            y_true.extend(y.tolist())
            y_score.extend(score.cpu().tolist())
            videos.extend(list(video))
            domains.extend(list(domain))
    return compute_video_metrics_by_domain(y_true, y_score, videos, domains)


def target_evaluate(model, loader, device, use_amp):
    model.eval()
    y_true, y_score, videos = [], [], []
    with torch.no_grad():
        for x, y, _, _, _, video in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(x)
            score = torch.softmax(logits.float(), dim=1)[:, 1]
            y_true.extend(y.tolist())
            y_score.extend(score.cpu().tolist())
            videos.extend(list(video))

    frame = compute_metrics(y_true, y_score)
    _, vy, vs = aggregate_video_scores(y_true, y_score, videos)
    video = compute_metrics(vy, vs)
    return {
        "frame_metrics": frame,
        "video_metrics": video,
        "num_frames": len(y_true),
        "num_videos": len(vy),
    }


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    sources = [d for d in MICO_DATASETS if d != args.target_dataset]
    train_samples, val_samples = [], []

    for i, domain in enumerate(sources):
        samples = scan_vfm_domain(args.data_root, domain)
        tr, va = split_train_val_by_group(
            samples,
            val_ratio=args.val_ratio,
            seed=args.seed + i * 17,
        )
        tr = evenly_subsample_frames_per_video(tr, args.max_frames_per_video)
        va = evenly_subsample_frames_per_video(va, args.max_frames_per_video)
        train_samples.extend(tr)
        val_samples.extend(va)

    train_ds = VFMImageDataset(train_samples, train=True, image_size=args.image_size)
    val_ds = VFMImageDataset(val_samples, train=False, image_size=args.image_size)
    batch_sampler = DomainClassBatchSampler(
        train_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    common = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, **common)
    val_loader = DataLoader(
        val_ds,
        batch_size=max(args.batch_size, 64),
        shuffle=False,
        **common,
    )

    model = ResNet18FeatureFAS(pretrained=True, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out = Path(args.output_dir) / args.target_dataset
    out.mkdir(parents=True, exist_ok=True)

    best_worst_auc = -1.0
    best_mean_auc = -1.0
    bad_epochs = 0
    history = []

    print("=" * 72)
    print("Route 1: Multi-source DG + ResNet18")
    print(f"Sources: {sources}")
    print(f"Target : {args.target_dataset}")
    print(f"Train  : {len(train_ds)} frames")
    print(f"Val    : {len(val_ds)} frames")
    print("=" * 72)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        y_true, y_score = [], []

        for x, y, _, _, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            score = torch.softmax(logits.detach().float(), dim=1)[:, 1]
            y_true.extend(y.detach().cpu().tolist())
            y_score.extend(score.cpu().tolist())
            total_loss += loss.item() * x.size(0)
            n += x.size(0)

        scheduler.step()
        train_auc = compute_metrics(y_true, y_score)["auc"]
        val = evaluate(model, val_loader, device, use_amp)

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(n, 1),
            "train_auc": train_auc,
            "val_video_auc": val["overall"]["auc"],
            "val_worst_domain_auc": val["worst_domain_auc"],
            "val_mean_domain_auc": val["mean_domain_auc"],
            "val_by_domain": {
                d: {
                    "auc": m["auc"],
                    "eer": m["eer"],
                }
                for d, m in val["per_domain"].items()
            },
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        worst = float(val["worst_domain_auc"])
        mean = float(val["mean_domain_auc"])
        improved = (
            worst > best_worst_auc + 1e-6
            or (
                abs(worst - best_worst_auc) <= 1e-6
                and mean > best_mean_auc + 1e-6
            )
        )
        if improved:
            best_worst_auc = worst
            best_mean_auc = mean
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "sources": sources,
                    "target": args.target_dataset,
                },
                out / "best.pth",
            )
            print(f"[BEST] worst={worst:.6f} mean={mean:.6f}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    (out / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    ckpt = torch.load(out / "best.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    target_samples = scan_vfm_domain(args.data_root, args.target_dataset)
    target_ds = VFMImageDataset(
        target_samples,
        train=False,
        image_size=args.image_size,
    )
    target_loader = DataLoader(
        target_ds,
        batch_size=64,
        shuffle=False,
        **common,
    )
    result = target_evaluate(model, target_loader, device, use_amp)
    summary = {
        "route": "multi_source_resnet18",
        "source_datasets": sources,
        "target_dataset": args.target_dataset,
        **result,
    }
    (out / "target_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("")
    print(f"Frame AUC : {result['frame_metrics']['auc']:.6f}")
    print(f"VIDEO AUC : {result['video_metrics']['auc']:.6f}")
    print(f"VIDEO EER : {result['video_metrics']['eer']:.6f}")


if __name__ == "__main__":
    main()
