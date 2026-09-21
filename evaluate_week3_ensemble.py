import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from metrics import save_plots_and_metrics
from model_ssdg_dino import DINOv2SSDG
from model_vfm import build_dinov2_fas
from vfm_data import (
    MICO_DATASETS,
    VFMImageDataset,
    aggregate_video_scores,
    scan_vfm_domain,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Label-free DINOv2 + SSDG ensemble evaluation on an unseen target."
    )
    p.add_argument("--dino_checkpoint", required=True)
    p.add_argument("--ssdg_checkpoint", required=True)
    p.add_argument("--dataset", required=True, choices=MICO_DATASETS)
    p.add_argument("--data_root", default="/root/Desktop/code/FAS/ProcessedData")
    p.add_argument("--output_dir", default="outputs_week3_ensemble")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--dinov2_repo", default=None)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_tta", action="store_true")
    return p.parse_args()


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def infer(model, loader, device, use_amp, use_tta):
    model.eval()
    labels, scores, videos, paths = [], [], [], []

    with torch.no_grad():
        for x, y, frame_paths, _, _, vids in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(x)
                probs = torch.softmax(logits.float(), dim=1)[:, 1]
                if use_tta:
                    logits_flip = model(torch.flip(x, dims=[3]))
                    probs_flip = torch.softmax(logits_flip.float(), dim=1)[:, 1]
                    probs = 0.5 * (probs + probs_flip)

            labels.extend(y.tolist())
            scores.extend(probs.cpu().tolist())
            videos.extend(list(vids))
            paths.extend(list(frame_paths))

    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "scores": np.asarray(scores, dtype=np.float64),
        "videos": np.asarray(videos, dtype=object),
        "paths": np.asarray(paths, dtype=object),
    }


def rank01(x):
    """Convert scores to [0, 1] ranks without using labels."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= 1:
        return np.zeros_like(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks / (len(x) - 1.0)


def evaluate_scores(labels, scores, videos, out, prefix):
    out = Path(out)
    frame_metrics = save_plots_and_metrics(
        labels.tolist(),
        scores.tolist(),
        out / "frame_level",
        prefix=f"{prefix}_frame",
    )
    ordered_videos, vy, vs = aggregate_video_scores(
        labels.tolist(),
        scores.tolist(),
        videos.tolist(),
    )
    video_metrics = save_plots_and_metrics(
        vy,
        vs,
        out / "video_level",
        prefix=f"{prefix}_video",
    )
    return frame_metrics, video_metrics, ordered_videos, vy, vs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    use_tta = not args.no_tta

    dino_ckpt = torch.load(args.dino_checkpoint, map_location="cpu")
    dino_cfg = dino_ckpt.get("args", {})
    dino_sources = dino_ckpt.get(
        "source_datasets", dino_cfg.get("source_datasets", [])
    )
    dino_target = dino_ckpt.get(
        "target_dataset", dino_cfg.get("target_dataset")
    )

    ssdg_ckpt = torch.load(args.ssdg_checkpoint, map_location="cpu")
    ssdg_cfg = ssdg_ckpt.get("args", {})
    ssdg_sources = ssdg_ckpt.get("sources", [])
    ssdg_target = ssdg_ckpt.get("target", ssdg_cfg.get("target_dataset"))

    if args.dataset in dino_sources or args.dataset in ssdg_sources:
        raise RuntimeError("Target leakage detected: target appears in source domains.")
    if dino_target and dino_target != args.dataset:
        raise RuntimeError(
            f"DINO checkpoint was trained for target={dino_target}, not {args.dataset}."
        )
    if ssdg_target and ssdg_target != args.dataset:
        raise RuntimeError(
            f"SSDG checkpoint was trained for target={ssdg_target}, not {args.dataset}."
        )
    if set(dino_sources) != set(ssdg_sources):
        raise RuntimeError(
            f"Source mismatch: DINO={dino_sources}, SSDG={ssdg_sources}"
        )

    image_size = int(dino_cfg.get("image_size", 224))
    if int(ssdg_cfg.get("image_size", 224)) != image_size:
        raise RuntimeError("DINO and SSDG image sizes differ.")

    samples = scan_vfm_domain(args.data_root, args.dataset)
    ds = VFMImageDataset(samples, train=False, image_size=image_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    print("=" * 76)
    print("Week 3 DINOv2 + SSDG label-free ensemble")
    print(f"Target          : {args.dataset}")
    print(f"Sources         : {', '.join(dino_sources)}")
    print(f"Frames          : {len(ds)}")
    print(f"TTA             : {'flip' if use_tta else 'off'}")
    print("=" * 76)

    print("[1/2] DINOv2-Reg inference")
    dino_model = build_dinov2_fas(
        model_name=dino_cfg.get("model_name", "dinov2_vitb14_reg"),
        pretrained=False,
        dropout=dino_cfg.get("dropout", 0.20),
        feature_mode=dino_cfg.get("feature_mode", "cls"),
        dinov2_repo=args.dinov2_repo or dino_cfg.get("dinov2_repo"),
        freeze_first_blocks=dino_cfg.get("freeze_first_blocks", 0),
    )
    dino_model.load_state_dict(dino_ckpt["model"], strict=True)
    dino_model = dino_model.to(device)
    dino_pred = infer(dino_model, loader, device, use_amp, use_tta)
    release_model(dino_model)

    print("[2/2] SSDG-style inference")
    ssdg_model = DINOv2SSDG(
        model_name=ssdg_cfg.get("model_name", "dinov2_vitb14_reg"),
        pretrained=False,
        dinov2_repo=args.dinov2_repo or ssdg_cfg.get("dinov2_repo"),
        freeze_first_blocks=ssdg_cfg.get("freeze_first_blocks", 10),
    )
    ssdg_model.load_state_dict(ssdg_ckpt["model"], strict=True)
    ssdg_model = ssdg_model.to(device)
    ssdg_pred = infer(ssdg_model, loader, device, use_amp, use_tta)
    release_model(ssdg_model)

    if not np.array_equal(dino_pred["labels"], ssdg_pred["labels"]):
        raise RuntimeError("DINO and SSDG label order mismatch.")
    if not np.array_equal(dino_pred["paths"], ssdg_pred["paths"]):
        raise RuntimeError("DINO and SSDG frame order mismatch.")
    if not np.array_equal(dino_pred["videos"], ssdg_pred["videos"]):
        raise RuntimeError("DINO and SSDG video order mismatch.")

    labels = dino_pred["labels"]
    videos = dino_pred["videos"]
    dino_scores = dino_pred["scores"]
    ssdg_scores = ssdg_pred["scores"]

    prob_mean = 0.5 * dino_scores + 0.5 * ssdg_scores
    rank_mean = 0.5 * rank01(dino_scores) + 0.5 * rank01(ssdg_scores)

    out = Path(args.output_dir) / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    methods = {
        "dino_tta": dino_scores,
        "ssdg_tta": ssdg_scores,
        "prob_mean": prob_mean,
        "rank_mean": rank_mean,
    }
    summary = {
        "protocol": (
            "Strict unseen-target evaluation. Fixed 0.5/0.5 ensemble; "
            "no target labels used for weighting or tuning."
        ),
        "target_dataset": args.dataset,
        "source_datasets": dino_sources,
        "tta": "original+horizontal_flip" if use_tta else "none",
        "dino_checkpoint": args.dino_checkpoint,
        "ssdg_checkpoint": args.ssdg_checkpoint,
        "results": {},
    }

    video_rows = {}
    for name, scores in methods.items():
        fm, vm, ordered_videos, vy, vs = evaluate_scores(
            labels,
            scores,
            videos,
            out / name,
            name,
        )
        summary["results"][name] = {
            "frame_metrics": fm,
            "video_metrics": vm,
        }
        video_rows[name] = (ordered_videos, vy, vs)
        print(
            f"{name:10s} | Frame AUC={fm['auc']:.6f} "
            f"| VIDEO AUC={vm['auc']:.6f} | VIDEO EER={vm['eer']:.6f}"
        )

    # Save aligned frame-level scores for later analysis without rerunning DINO.
    with open(out / "frame_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "path", "video_id", "label",
            "dino_score", "ssdg_score", "prob_mean", "rank_mean",
        ])
        for row in zip(
            dino_pred["paths"],
            videos,
            labels,
            dino_scores,
            ssdg_scores,
            prob_mean,
            rank_mean,
        ):
            writer.writerow(row)

    # Save the primary rank-fusion video scores.
    rv, ry, rs = video_rows["rank_mean"]
    with open(out / "rank_mean_video_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "label", "live_score"])
        writer.writerows(zip(rv, ry, rs))

    (out / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("")
    print("=" * 76)
    print("Primary ensemble: rank_mean (fixed 0.5/0.5, label-free)")
    print(
        f"VIDEO AUC = "
        f"{summary['results']['rank_mean']['video_metrics']['auc']:.6f}"
    )
    print(
        f"VIDEO EER = "
        f"{summary['results']['rank_mean']['video_metrics']['eer']:.6f}"
    )
    print(f"Output    = {out}")
    print("=" * 76)


if __name__ == "__main__":
    main()
