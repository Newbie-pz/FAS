import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs_week3_three_routes")
    return p.parse_args()


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    root = Path(args.root)
    targets = set()

    for base in (
        root / "route1_resnet",
        root / "route2_dinov2_eval",
        root / "route3_ssdg_dino",
    ):
        if base.exists():
            targets.update(p.name for p in base.iterdir() if p.is_dir())

    rows = []
    for target in sorted(targets):
        r1 = read_json(root / "route1_resnet" / target / "target_summary.json")
        r2 = read_json(root / "route2_dinov2_eval" / target / "vfm_summary.json")
        r3 = read_json(root / "route3_ssdg_dino" / target / "target_summary.json")

        entries = []
        if r1:
            entries.append(("Multi-source ResNet18", r1))
        if r2:
            entries.append(("DINOv2-Reg + Multi-source", r2))
        if r3:
            entries.append(("DINOv2-Reg + SSDG-style", r3))

        for name, data in entries:
            frame = data["frame_metrics"]
            video = data["video_metrics"]
            rows.append({
                "target": target,
                "method": name,
                "frame_accuracy": frame["accuracy"],
                "frame_auc": frame["auc"],
                "frame_eer": frame["eer"],
                "video_accuracy": video["accuracy"],
                "video_auc": video["auc"],
                "video_eer": video["eer"],
            })

    if not rows:
        raise RuntimeError(f"No completed route summaries found under {root}")

    csv_path = root / "week3_three_routes_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = root / "week3_three_routes_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Week 3: Three-Route DG Comparison\n\n")
        f.write(
            "| Target | Method | Frame Acc | Frame AUC | Frame EER | "
            "Video Acc | Video AUC | Video EER |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['target']} | {r['method']} | "
                f"{r['frame_accuracy']:.4%} | {r['frame_auc']:.4%} | "
                f"{r['frame_eer']:.4%} | {r['video_accuracy']:.4%} | "
                f"{r['video_auc']:.4%} | {r['video_eer']:.4%} |\n"
            )

    print(f"Generated: {csv_path}")
    print(f"Generated: {md_path}")


if __name__ == "__main__":
    main()
