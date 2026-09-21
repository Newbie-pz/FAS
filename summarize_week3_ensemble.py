import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs_week3_ensemble")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []

    for target in ("CASIA", "MSU-MFSD", "Replay-Attack"):
        p = root / target / "ensemble_summary.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for method in ("dino_tta", "ssdg_tta", "prob_mean", "rank_mean"):
            m = data["results"][method]
            rows.append({
                "target": target,
                "method": method,
                "frame_auc": float(m["frame_metrics"]["auc"]),
                "video_auc": float(m["video_metrics"]["auc"]),
                "video_eer": float(m["video_metrics"]["eer"]),
            })

    if not rows:
        raise RuntimeError(f"No ensemble summaries found under {root}")

    md = root / "ensemble_summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# DINOv2 + SSDG Ensemble Results\n\n")
        f.write("| Target | Method | Frame AUC | Video AUC | Video EER |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['target']} | {r['method']} | "
                f"{r['frame_auc']:.4%} | {r['video_auc']:.4%} | "
                f"{r['video_eer']:.4%} |\n"
            )

        rank_rows = [r for r in rows if r["method"] == "rank_mean"]
        if rank_rows:
            avg_auc = sum(r["video_auc"] for r in rank_rows) / len(rank_rows)
            avg_eer = sum(r["video_eer"] for r in rank_rows) / len(rank_rows)
            f.write("\n## Primary result\n\n")
            f.write(
                f"Fixed 0.5/0.5 rank fusion average Video AUC: "
                f"{avg_auc:.4%}\n\n"
            )
            f.write(
                f"Fixed 0.5/0.5 rank fusion average Video EER: "
                f"{avg_eer:.4%}\n"
            )

    print(md)


if __name__ == "__main__":
    main()
