import argparse
import json
from pathlib import Path
import numpy as np


DINO = ["dino_f12_lr5e6", "dino_f10_lr2e6"]
SSDG = [
    "ssdg_f11_ad02_tri10",
    "ssdg_f11_ad01_tri05",
    "ssdg_f10_ad02_tri05",
    "ssdg_f12_ad02_tri05",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs_week3_parallel_sweep")
    p.add_argument(
        "--targets",
        nargs="+",
        default=["CASIA", "MSU-MFSD", "Replay-Attack"],
    )
    return p.parse_args()


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_source_worst(history_path):
    h = load_json(history_path)
    if not h:
        return None
    vals = []
    for r in h:
        if "val_worst_domain_auc" in r:
            vals.append(float(r["val_worst_domain_auc"]))
        elif "val_video_auc" in r:
            vals.append(float(r["val_video_auc"]))
    return max(vals) if vals else None


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []

    for name in DINO:
        for target in args.targets:
            s = load_json(root / f"{name}_eval" / target / "vfm_summary.json")
            h = root / f"{name}_train" / target / "history.json"
            if not s:
                continue
            rows.append({
                "variant": name,
                "target": target,
                "frame_auc": float(s["frame_metrics"]["auc"]),
                "video_auc": float(s["video_metrics"]["auc"]),
                "video_eer": float(s["video_metrics"]["eer"]),
                "source_worst_auc": best_source_worst(h),
            })

    for name in SSDG:
        for target in args.targets:
            s = load_json(root / name / target / "target_summary.json")
            h = root / name / target / "history.json"
            if not s:
                continue
            rows.append({
                "variant": name,
                "target": target,
                "frame_auc": float(s["frame_metrics"]["auc"]),
                "video_auc": float(s["video_metrics"]["auc"]),
                "video_eer": float(s["video_metrics"]["eer"]),
                "source_worst_auc": best_source_worst(h),
            })

    if not rows:
        raise RuntimeError(f"No completed results found under {root}")

    variants = DINO + SSDG
    md = root / "parallel_sweep_summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Week 3 Parallel Sweep Results\n\n")
        f.write("| Variant | CASIA | MSU-MFSD | Replay-Attack | Mean Video AUC | Mean Video EER | Mean Source-Worst AUC |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")

        for v in variants:
            sub = [r for r in rows if r["variant"] == v]
            if not sub:
                continue
            by_target = {r["target"]: r for r in sub}
            aucs = [r["video_auc"] for r in sub]
            eers = [r["video_eer"] for r in sub]
            src = [r["source_worst_auc"] for r in sub if r["source_worst_auc"] is not None]

            vals = []
            for t in args.targets:
                vals.append(by_target[t]["video_auc"] if t in by_target else None)

            def fmt(x):
                return "NA" if x is None else f"{x:.4%}"

            f.write(
                f"| {v} | "
                + " | ".join(fmt(x) for x in vals)
                + f" | {np.mean(aucs):.4%} | {np.mean(eers):.4%} | "
                + (f"{np.mean(src):.4%}" if src else "NA")
                + " |\n"
            )

        f.write("\n## Per-target details\n\n")
        f.write("| Variant | Target | Frame AUC | Video AUC | Video EER | Source-Worst AUC |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['variant']} | {r['target']} | {r['frame_auc']:.4%} | "
                f"{r['video_auc']:.4%} | {r['video_eer']:.4%} | "
                + ("NA" if r["source_worst_auc"] is None else f"{r['source_worst_auc']:.4%}")
                + " |\n"
            )

    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
