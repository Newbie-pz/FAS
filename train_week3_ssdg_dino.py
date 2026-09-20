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
from model_ssdg_dino import DINOv2SSDG
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
    DomainDiscriminator,
    batch_all_triplet_loss,
    compute_video_metrics_by_domain,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/root/Desktop/code/FAS/ProcessedData")
    p.add_argument("--target_dataset", required=True, choices=MICO_DATASETS)
    p.add_argument("--output_dir", default="outputs_week3_three_routes/route3_ssdg_dino")
    p.add_argument("--model_name", default="dinov2_vitb14_reg")
    p.add_argument("--dinov2_repo", default=None)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--freeze_first_blocks", type=int, default=10)
    p.add_argument("--backbone_lr", type=float, default=2e-6)
    p.add_argument("--head_lr", type=float, default=5e-5)
    p.add_argument("--disc_lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)

    p.add_argument("--lambda_triplet", type=float, default=1.0)
    p.add_argument("--lambda_adreal", type=float, default=0.5)
    p.add_argument("--triplet_margin", type=float, default=0.1)
    p.add_argument("--grl_alpha", type=float, default=10.0)

    p.add_argument("--patience", type=int, default=8)
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


def grl_coeff(progress, alpha=10.0):
    progress = min(max(float(progress), 0.0), 1.0)
    return float(2.0 / (1.0 + math.exp(-alpha * progress)) - 1.0)


def build_optimizer(model, discriminator, args):
    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        no_decay = p.ndim == 1 or name.endswith(".bias")
        if is_backbone:
            (backbone_no_decay if no_decay else backbone_decay).append(p)
        else:
            (head_no_decay if no_decay else head_decay).append(p)

    groups = []
    if backbone_decay:
        groups.append({
            "params": backbone_decay,
            "lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
        })
    if backbone_no_decay:
        groups.append({
            "params": backbone_no_decay,
            "lr": args.backbone_lr,
            "weight_decay": 0.0,
        })
    if head_decay:
        groups.append({
            "params": head_decay,
            "lr": args.head_lr,
            "weight_decay": args.weight_decay,
        })
    if head_no_decay:
        groups.append({
            "params": head_no_decay,
            "lr": args.head_lr,
            "weight_decay": 0.0,
        })
    groups.append({
        "params": discriminator.parameters(),
        "lr": args.disc_lr,
        "weight_decay": args.weight_decay,
    })
    return AdamW(groups)


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

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    sources = [d for d in MICO_DATASETS if d != args.target_dataset]
    domain_to_id = {d: i for i, d in enumerate(sources)}

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
        batch_size=max(64, args.batch_size),
        shuffle=False,
        **common,
    )

    model = DINOv2SSDG(
        model_name=args.model_name,
        pretrained=True,
        dinov2_repo=args.dinov2_repo,
        freeze_first_blocks=args.freeze_first_blocks,
    ).to(device)
    discriminator = DomainDiscriminator(
        in_dim=model.feature_dim,
        num_domains=len(sources),
        hidden=512,
        dropout=0.5,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, discriminator, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out = Path(args.output_dir) / args.target_dataset
    out.mkdir(parents=True, exist_ok=True)

    history = []
    best_worst_auc = -1.0
    best_mean_auc = -1.0
    bad_epochs = 0
    total_steps = max(1, args.epochs * len(train_loader))
    global_step = 0

    print("=" * 76)
    print("Route 3: DINOv2-Reg + SSDG-style single-side DG")
    print(f"Sources          : {sources}")
    print(f"Target           : {args.target_dataset}")
    print(f"Freeze blocks    : {args.freeze_first_blocks}")
    print(f"Triplet weight   : {args.lambda_triplet}")
    print(f"Real adv weight  : {args.lambda_adreal}")
    print(f"Triplet margin   : {args.triplet_margin}")
    print("=" * 76)

    for epoch in range(1, args.epochs + 1):
        model.train()
        discriminator.train()

        totals = {
            "loss": 0.0,
            "cls": 0.0,
            "triplet": 0.0,
            "adreal": 0.0,
            "n": 0,
        }
        y_true, y_score = [], []

        for x, y, _, _, domains, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            domain_ids = torch.tensor(
                [domain_to_id[d] for d in domains],
                dtype=torch.long,
                device=device,
            )

            progress = global_step / max(total_steps - 1, 1)
            coeff = grl_coeff(progress, args.grl_alpha)
            global_step += 1

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits, features = model(x, return_features=True)
                cls_loss = criterion(logits, y)

                # SSDG asymmetric grouping:
                # all Live -> group 0; Spoof -> one group per source domain.
                triplet_labels = torch.where(
                    y == 1,
                    torch.zeros_like(domain_ids),
                    domain_ids + 1,
                )
                triplet_loss = batch_all_triplet_loss(
                    features.float(),
                    triplet_labels,
                    margin=args.triplet_margin,
                )

                live_mask = y == 1
                if live_mask.any():
                    domain_logits = discriminator(
                        features[live_mask],
                        grl_coeff=coeff,
                    )
                    adreal_loss = criterion(
                        domain_logits,
                        domain_ids[live_mask],
                    )
                else:
                    adreal_loss = features.sum() * 0.0

                loss = (
                    cls_loss
                    + args.lambda_triplet * triplet_loss
                    + args.lambda_adreal * adreal_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(discriminator.parameters()),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()

            score = torch.softmax(logits.detach().float(), dim=1)[:, 1]
            y_true.extend(y.detach().cpu().tolist())
            y_score.extend(score.cpu().tolist())

            bs = x.size(0)
            totals["loss"] += loss.item() * bs
            totals["cls"] += cls_loss.item() * bs
            totals["triplet"] += triplet_loss.item() * bs
            totals["adreal"] += adreal_loss.item() * bs
            totals["n"] += bs

        scheduler.step()
        train_auc = compute_metrics(y_true, y_score)["auc"]
        val = evaluate(model, val_loader, device, use_amp)

        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(totals["n"], 1),
            "cls_loss": totals["cls"] / max(totals["n"], 1),
            "triplet_loss": totals["triplet"] / max(totals["n"], 1),
            "adreal_loss": totals["adreal"] / max(totals["n"], 1),
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
                    "best_worst_domain_auc": best_worst_auc,
                    "best_mean_domain_auc": best_mean_auc,
                },
                out / "best.pth",
            )
            print(
                f"[BEST] worst_domain_auc={worst:.6f} "
                f"mean_domain_auc={mean:.6f}"
            )
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
        "route": "dinov2_reg_ssdg_style",
        "ssdg_core": {
            "single_side_live_domain_adversarial": True,
            "asymmetric_triplet": True,
            "feature_weight_normalization": True,
            "lambda_triplet": args.lambda_triplet,
            "lambda_adreal": args.lambda_adreal,
            "triplet_margin": args.triplet_margin,
        },
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
