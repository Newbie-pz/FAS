import argparse
import json
from pathlib import Path
import numpy as np


VARIANTS = [
    "baseline_f12_tta",
    "cls_focal_g1_lr1e4",
    "cls_focal_g2_lr1e4",
    "cls_focal_g2_lr2e4",
    "cls_ce_lr1e4",
    "clsmean_focal_g2_lr1e4",
    "clsmeanstd_focal_g2_lr1e4",
    "cls_focal_g2_lr1e4_ls005",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs_week3_frozen_sweep")
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


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []

    for name in VARIANTS:
        for target in args.targets:
            if name == "baseline_f12_tta":
                p = root / "baseline_f12_tta_eval" / target / "vfm_summary.json"
            else:
                p = root / f"{name}_eval" / target / "vfm_summary.json"
            s = load_json(p)
            if not s:
                continue
            rows.append({
                "variant": name,
                "target": target,
                "frame_auc": float(s["frame_metrics"]["auc"]),
                "video_auc": float(s["video_metrics"]["auc"]),
                "video_eer": float(s["video_metrics"]["eer"]),
            })

    if not rows:
        raise RuntimeError(f"No completed results found under {root}")

    md = root / "frozen_sweep_summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Week 3 Frozen-DINO Sweep Results\n\n")
        f.write("| Variant | CASIA | MSU-MFSD | Replay-Attack | Mean Video AUC | Mean Video EER |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")

        for v in VARIANTS:
            sub = [r for r in rows if r["variant"] == v]
            if not sub:
                continue
            by_target = {r["target"]: r for r in sub}
            aucs = [r["video_auc"] for r in sub]
            eers = [r["video_eer"] for r in sub]

            vals = []
            for t in args.targets:
                vals.append(by_target[t]["video_auc"] if t in by_target else None)

            def fmt(x):
                return "NA" if x is None else f"{x:.4%}"

            f.write(
                f"| {v} | "
                + " | ".join(fmt(x) for x in vals)
                + f" | {np.mean(aucs):.4%} | {np.mean(eers):.4%} |\n"
            )

        f.write("\n## Per-target details\n\n")
        f.write("| Variant | Target | Frame AUC | Video AUC | Video EER |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['variant']} | {r['target']} | "
                f"{r['frame_auc']:.4%} | {r['video_auc']:.4%} | "
                f"{r['video_eer']:.4%} |\n"
            )

    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
