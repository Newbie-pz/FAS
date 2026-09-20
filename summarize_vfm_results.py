import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs_week2_vfm_eval")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []

    for path in sorted(root.glob("*/vfm_summary.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        frame = m["frame_metrics"]
        video = m["video_metrics"]
        rows.append({
            "target_dataset": m["target_dataset"],
            "source_datasets": "+".join(m.get("source_datasets", [])),
            "num_frames": m["num_frames"],
            "num_videos": m["num_videos"],
            "frame_accuracy": frame["accuracy"],
            "frame_auc": frame["auc"],
            "frame_eer": frame["eer"],
            "video_accuracy": video["accuracy"],
            "video_auc": video["auc"],
            "video_eer": video["eer"],
        })

    if not rows:
        raise RuntimeError(f"No vfm_summary.json found under {root}")

    csv_path = root / "vfm_cross_domain_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = root / "vfm_cross_domain_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# DINOv2-Reg 多源跨域结果\n\n")
        f.write("| Target | Sources | Frames | Videos | Frame AUC | Frame EER | Video AUC | Video EER |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['target_dataset']} | {r['source_datasets']} | "
                f"{r['num_frames']} | {r['num_videos']} | "
                f"{float(r['frame_auc']):.4%} | {float(r['frame_eer']):.4%} | "
                f"{float(r['video_auc']):.4%} | {float(r['video_eer']):.4%} |\n"
            )

    print(f"已生成: {csv_path}")
    print(f"已生成: {md_path}")


if __name__ == "__main__":
    main()
