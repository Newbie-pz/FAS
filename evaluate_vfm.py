import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from metrics import save_plots_and_metrics
from model_vfm import build_dinov2_fas
from vfm_data import (
    MICO_DATASETS,
    VFMImageDataset,
    aggregate_video_scores,
    scan_vfm_domain,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a DINOv2 FAS checkpoint on an unseen target domain."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, choices=MICO_DATASETS)
    p.add_argument("--data_root", default="/root/Desktop/code/FAS/ProcessedData")
    p.add_argument("--output_dir", default="outputs_week2_vfm_eval")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--dinov2_repo", default=None)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--tta", action="store_true", help="Average original and horizontal-flip predictions.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("args", {})
    trained_target = ckpt.get("target_dataset", cfg.get("target_dataset"))
    sources = ckpt.get("source_datasets", cfg.get("source_datasets", []))

    if args.dataset in sources:
        raise RuntimeError(
            f"{args.dataset} was used as a source domain; this is not unseen-domain evaluation."
        )
    if trained_target and args.dataset != trained_target:
        print(
            f"[警告] checkpoint target={trained_target}, "
            f"but evaluating dataset={args.dataset}."
        )

    model = build_dinov2_fas(
        model_name=cfg.get("model_name", "dinov2_vitb14_reg"),
        pretrained=False,
        dropout=cfg.get("dropout", 0.20),
        feature_mode=cfg.get("feature_mode", "cls_mean"),
        dinov2_repo=args.dinov2_repo or cfg.get("dinov2_repo"),
        freeze_first_blocks=cfg.get("freeze_first_blocks", 0),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)
    model.eval()

    image_size = int(cfg.get("image_size", 224))
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

    y_true, y_score, video_ids, frame_paths = [], [], [], []
    with torch.no_grad():
        for x, y, paths, _, _, vids in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(x)
                probs = torch.softmax(logits.float(), dim=1)[:, 1]
                if args.tta:
                    logits_flip = model(torch.flip(x, dims=[3]))
                    probs_flip = torch.softmax(logits_flip.float(), dim=1)[:, 1]
                    probs = 0.5 * (probs + probs_flip)

            y_true.extend(y.tolist())
            y_score.extend(probs.cpu().tolist())
            video_ids.extend(list(vids))
            frame_paths.extend(list(paths))

    out = Path(args.output_dir) / args.dataset
    frame_dir = out / "frame_level"
    video_dir = out / "video_level"

    frame_metrics = save_plots_and_metrics(
        y_true,
        y_score,
        frame_dir,
        prefix="frame",
    )

    ordered_videos, video_true, video_score = aggregate_video_scores(
        y_true,
        y_score,
        video_ids,
    )
    video_metrics = save_plots_and_metrics(
        video_true,
        video_score,
        video_dir,
        prefix="video",
    )

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "video_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "label", "live_score"])
        writer.writerows(zip(ordered_videos, video_true, video_score))

    summary = {
        "protocol": "MICO-style multi-source training; unseen target-only testing; no target fine-tuning",
        "model": cfg.get("model_name", "dinov2_vitb14_reg"),
        "feature_mode": cfg.get("feature_mode", "cls_mean"),
        "source_datasets": sources,
        "target_dataset": args.dataset,
        "num_frames": len(y_true),
        "num_videos": len(video_true),
        "frame_metrics": frame_metrics,
        "video_metrics": video_metrics,
        "checkpoint": args.checkpoint,
        "tta": "original+horizontal_flip" if args.tta else "none",
    }
    (out / "vfm_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("")
    print("=" * 72)
    print("VFM unseen-domain evaluation")
    print(f"Sources         : {', '.join(sources)}")
    print(f"Target          : {args.dataset}")
    print(f"Frames          : {len(y_true)}")
    print(f"Videos          : {len(video_true)}")
    print(f"Frame AUC       : {frame_metrics['auc']:.6f}")
    print(f"Frame EER       : {frame_metrics['eer']:.6f}")
    print(f"VIDEO AUC       : {video_metrics['auc']:.6f}")
    print(f"VIDEO EER       : {video_metrics['eer']:.6f}")
    print(f"TTA             : {'on' if args.tta else 'off'}")
    print(f"Output          : {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
