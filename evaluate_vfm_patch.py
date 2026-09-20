import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from metrics import save_plots_and_metrics
from model_vfm_patch import DINOv2PatchFAS
from vfm_data import MICO_DATASETS, aggregate_video_scores, scan_vfm_domain
from vfm_patch_data import VFMPatchDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, choices=MICO_DATASETS)
    p.add_argument("--data_root", default="/root/Desktop/code/FAS/ProcessedData")
    p.add_argument("--output_dir", default="outputs_week2_vfm_strong_eval")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--dinov2_repo", default=None)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["args"]
    sources = ckpt.get("source_datasets", [])

    if args.dataset in sources:
        raise RuntimeError("Target dataset appears in source domains.")

    model = DINOv2PatchFAS(
        model_name=cfg.get("model_name", "dinov2_vitb14_reg"),
        pretrained=False,
        dinov2_repo=args.dinov2_repo or cfg.get("dinov2_repo"),
        dropout=cfg.get("dropout", 0.2),
        freeze_first_blocks=cfg.get("freeze_first_blocks", 10),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)
    model.eval()

    samples = scan_vfm_domain(args.data_root, args.dataset)
    ds = VFMPatchDataset(
        samples,
        train=False,
        image_size=cfg.get("image_size", 224),
        fas_aug_p=0.0,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    y_true, y_score, video_ids = [], [], []
    with torch.no_grad():
        for x, y, _, _, _, vids, _ in loader:
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

    out = Path(args.output_dir) / args.dataset
    frame_metrics = save_plots_and_metrics(
        y_true,
        y_score,
        out / "frame_level",
        prefix="frame",
    )
    vids, vy, vs = aggregate_video_scores(y_true, y_score, video_ids)
    video_metrics = save_plots_and_metrics(
        vy,
        vs,
        out / "video_level",
        prefix="video",
    )

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "video_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "label", "live_score"])
        writer.writerows(zip(vids, vy, vs))

    summary = {
        "protocol": "DINOv2-Reg + FAS-Aug + PDA + APL; multi-source DG; unseen target",
        "source_datasets": sources,
        "target_dataset": args.dataset,
        "num_frames": len(y_true),
        "num_videos": len(vy),
        "frame_metrics": frame_metrics,
        "video_metrics": video_metrics,
        "checkpoint": args.checkpoint,
    }
    (out / "vfm_patch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("")
    print("=" * 72)
    print("Strong VFM unseen-domain evaluation")
    print(f"Sources         : {', '.join(sources)}")
    print(f"Target          : {args.dataset}")
    print(f"Frame AUC       : {frame_metrics['auc']:.6f}")
    print(f"Frame EER       : {frame_metrics['eer']:.6f}")
    print(f"VIDEO AUC       : {video_metrics['auc']:.6f}")
    print(f"VIDEO EER       : {video_metrics['eer']:.6f}")
    print(f"Output          : {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
