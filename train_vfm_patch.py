import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from metrics import compute_metrics
from model_vfm_patch import DINOv2PatchFAS
from vfm_data import (
    MICO_DATASETS,
    aggregate_video_scores,
    build_domain_class_balanced_sampler,
    evenly_subsample_frames_per_video,
    scan_vfm_domain,
    split_train_val_by_group,
)
from vfm_patch_data import VFMPatchDataset
from vfm_patch_ops import (
    apply_patch_data_augmentation,
    attention_weighted_patch_loss,
    focal_loss,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="DINOv2-Reg + FAS-Aug + PDA + APL for domain-generalizable FAS."
    )
    p.add_argument("--data_root", default="/root/Desktop/code/FAS/ProcessedData")
    p.add_argument("--target_dataset", required=True, choices=MICO_DATASETS)
    p.add_argument("--source_datasets", nargs="+", default=None)
    p.add_argument("--output_dir", default="outputs_week2_vfm_strong")
    p.add_argument("--model_name", default="dinov2_vitb14_reg")
    p.add_argument("--dinov2_repo", default=None)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--backbone_lr", type=float, default=2e-6)
    p.add_argument("--head_lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--freeze_first_blocks", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.20)

    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--apl_weight", type=float, default=1.0)
    p.add_argument("--fas_aug_p", type=float, default=0.75)
    p.add_argument("--pda_p", type=float, default=0.5)
    p.add_argument("--pda_replace_ratio", type=float, default=0.25)
    p.add_argument("--patch_size", type=int, default=14)

    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--max_train_frames_per_video", type=int, default=20)
    p.add_argument("--max_val_frames_per_video", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model, args):
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
    for params, lr, wd in (
        (backbone_decay, args.backbone_lr, args.weight_decay),
        (backbone_no_decay, args.backbone_lr, 0.0),
        (head_decay, args.head_lr, args.weight_decay),
        (head_no_decay, args.head_lr, 0.0),
    ):
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": wd})

    return AdamW(groups)


def make_scheduler(optimizer, total_updates, warmup_ratio):
    warmup = max(1, int(total_updates * warmup_ratio))

    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_updates - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=fn)


def evaluate(model, loader, device, use_amp):
    model.eval()
    y_true, y_score, video_ids, domains = [], [], [], []

    with torch.no_grad():
        for x, y, _, _, domain, vids, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(x, return_patch=False)
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            y_true.extend(y.tolist())
            y_score.extend(probs.cpu().tolist())
            video_ids.extend(list(vids))
            domains.extend(list(domain))

    _, vy, vs = aggregate_video_scores(y_true, y_score, video_ids)
    overall = compute_metrics(vy, vs)

    by_domain = {}
    for domain in sorted(set(domains)):
        idx = [i for i, d in enumerate(domains) if d == domain]
        dy = [y_true[i] for i in idx]
        ds = [y_score[i] for i in idx]
        dv = [video_ids[i] for i in idx]
        _, dvy, dvs = aggregate_video_scores(dy, ds, dv)
        by_domain[domain] = compute_metrics(dvy, dvs)

    aucs = [m["auc"] for m in by_domain.values() if np.isfinite(m["auc"])]
    worst_auc = float(min(aucs)) if aucs else float("nan")
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")

    return {
        "overall_video": overall,
        "by_domain": by_domain,
        "worst_domain_video_auc": worst_auc,
        "mean_domain_video_auc": mean_auc,
        "num_videos": len(vy),
    }


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    use_amp,
    args,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_global = 0.0
    total_patch = 0.0
    count = 0
    ys, scores = [], []
    aug_counter = Counter()

    for step, (x, y, _, _, _, _, aug_names) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x, patch_labels = apply_patch_data_augmentation(
            x,
            y,
            patch_size=args.patch_size,
            apply_p=args.pda_p,
            replace_ratio=args.pda_replace_ratio,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(x, return_patch=True)
            global_loss = focal_loss(
                outputs["global_logits"],
                y,
                gamma=args.focal_gamma,
            )
            patch_loss = attention_weighted_patch_loss(
                outputs["patch_logits"],
                patch_labels,
                outputs["patch_attention"],
                gamma=args.focal_gamma,
            )
            raw_loss = global_loss + args.apl_weight * patch_loss
            loss = raw_loss / args.grad_accum

        scaler.scale(loss).backward()

        if step % args.grad_accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        probs = torch.softmax(outputs["global_logits"].detach().float(), dim=1)[:, 1]
        ys.extend(y.detach().cpu().tolist())
        scores.extend(probs.cpu().tolist())
        aug_counter.update(list(aug_names))

        bs = x.size(0)
        total_loss += raw_loss.item() * bs
        total_global += global_loss.item() * bs
        total_patch += patch_loss.item() * bs
        count += bs

    metrics = compute_metrics(ys, scores)
    return {
        "loss": total_loss / max(count, 1),
        "global_loss": total_global / max(count, 1),
        "patch_loss": total_patch / max(count, 1),
        "frame_auc": metrics["auc"],
        "frame_accuracy": metrics["accuracy"],
        "augmentations": dict(aug_counter),
    }


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.image_size % args.patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size.")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    sources = args.source_datasets or [
        d for d in MICO_DATASETS if d != args.target_dataset
    ]
    if args.target_dataset in sources:
        raise RuntimeError("Target leakage: target appears in source_datasets.")

    train_samples, val_samples = [], []
    domain_summary = {}
    for i, domain in enumerate(sources):
        samples = scan_vfm_domain(args.data_root, domain)
        tr, va = split_train_val_by_group(
            samples,
            val_ratio=args.val_ratio,
            seed=args.seed + i * 17,
        )
        tr = evenly_subsample_frames_per_video(
            tr, args.max_train_frames_per_video
        )
        va = evenly_subsample_frames_per_video(
            va, args.max_val_frames_per_video
        )
        train_samples.extend(tr)
        val_samples.extend(va)
        domain_summary[domain] = {
            "train_frames": len(tr),
            "val_frames": len(va),
            "train_groups": len({x[2] for x in tr}),
            "val_groups": len({x[2] for x in va}),
            "train_videos": len({x[4] for x in tr}),
            "val_videos": len({x[4] for x in va}),
        }

    sampler = build_domain_class_balanced_sampler(train_samples)
    train_ds = VFMPatchDataset(
        train_samples,
        train=True,
        image_size=args.image_size,
        fas_aug_p=args.fas_aug_p,
    )
    val_ds = VFMPatchDataset(
        val_samples,
        train=False,
        image_size=args.image_size,
        fas_aug_p=0.0,
    )

    common = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(32, args.batch_size),
        shuffle=False,
        **common,
    )

    print("=" * 76)
    print("Strong VFM: DINOv2-Reg + FAS-Aug + PDA + APL")
    print(f"Target          : {args.target_dataset}")
    print(f"Sources         : {', '.join(sources)}")
    print(f"Train frames    : {len(train_ds)}")
    print(f"Val frames      : {len(val_ds)}")
    print(f"Freeze blocks   : {args.freeze_first_blocks}")
    print(f"FAS-Aug p       : {args.fas_aug_p}")
    print(f"PDA p/ratio     : {args.pda_p}/{args.pda_replace_ratio}")
    print(f"APL weight      : {args.apl_weight}")
    print("=" * 76)

    model = DINOv2PatchFAS(
        model_name=args.model_name,
        pretrained=True,
        dinov2_repo=args.dinov2_repo,
        dropout=args.dropout,
        freeze_first_blocks=args.freeze_first_blocks,
    ).to(device)

    optimizer = make_optimizer(model, args)
    updates = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    scheduler = make_scheduler(optimizer, updates, args.warmup_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out = Path(args.output_dir) / args.target_dataset
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_summary.json").write_text(
        json.dumps(
            {
                "protocol": "multi-source DG; target excluded from train/val",
                "target_dataset": args.target_dataset,
                "source_datasets": sources,
                "domain_summary": domain_summary,
                "fas_aug_p": args.fas_aug_p,
                "pda_p": args.pda_p,
                "pda_replace_ratio": args.pda_replace_ratio,
                "apl_weight": args.apl_weight,
                "freeze_first_blocks": args.freeze_first_blocks,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    history = []
    best_worst_auc = -1.0
    best_mean_auc = -1.0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_result = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            use_amp,
            args,
        )
        val_result = evaluate(model, val_loader, device, use_amp)

        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_global_loss": train_result["global_loss"],
            "train_patch_loss": train_result["patch_loss"],
            "train_frame_auc": train_result["frame_auc"],
            "val_video_auc": val_result["overall_video"]["auc"],
            "val_video_eer": val_result["overall_video"]["eer"],
            "val_worst_domain_auc": val_result["worst_domain_video_auc"],
            "val_mean_domain_auc": val_result["mean_domain_video_auc"],
            "val_by_domain": {
                d: {
                    "auc": m["auc"],
                    "eer": m["eer"],
                }
                for d, m in val_result["by_domain"].items()
            },
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        worst_auc = val_result["worst_domain_video_auc"]
        mean_auc = val_result["mean_domain_video_auc"]
        improved = (
            worst_auc > best_worst_auc + 1e-6
            or (
                abs(worst_auc - best_worst_auc) <= 1e-6
                and mean_auc > best_mean_auc + 1e-6
            )
        )

        if improved:
            best_worst_auc = worst_auc
            best_mean_auc = mean_auc
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "source_datasets": sources,
                    "target_dataset": args.target_dataset,
                    "best_worst_domain_video_auc": best_worst_auc,
                    "best_mean_domain_video_auc": best_mean_auc,
                },
                out / "best.pth",
            )
            print(
                f"[BEST] epoch={epoch} "
                f"worst_domain_auc={best_worst_auc:.6f} "
                f"mean_domain_auc={best_mean_auc:.6f}"
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
    print(
        "Training complete:",
        {
            "best_worst_domain_video_auc": best_worst_auc,
            "best_mean_domain_video_auc": best_mean_auc,
            "checkpoint": str(out / "best.pth"),
        },
    )


if __name__ == "__main__":
    main()
