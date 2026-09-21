import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from metrics import compute_metrics


EPS = 1e-6


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch-evaluate multiple label-free DINO+SSDG fusion rules from cached frame scores."
    )
    p.add_argument("--root", default="outputs_week3_ensemble")
    p.add_argument(
        "--targets",
        nargs="+",
        default=["CASIA", "MSU-MFSD", "Replay-Attack"],
    )
    p.add_argument("--output", default="outputs_week3_dynamic_fusion")
    return p.parse_args()


def load_scores(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "path": r["path"],
                    "video_id": r["video_id"],
                    "label": int(r["label"]),
                    "dino": float(r["dino_score"]),
                    "ssdg": float(r["ssdg_score"]),
                }
            )
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def entropy_reliability(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0)
    return 1.0 - h


def confidence_reliability(p):
    p = np.asarray(p, dtype=np.float64)
    return 2.0 * np.abs(p - 0.5)


def weighted(a, b, wa, wb):
    denom = wa + wb + EPS
    return (wa * a + wb * b) / denom


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def frame_methods(dino, ssdg):
    cd = confidence_reliability(dino)
    cs = confidence_reliability(ssdg)
    ed = entropy_reliability(dino)
    es = entropy_reliability(ssdg)

    agree = ((dino >= 0.5) == (ssdg >= 0.5))
    more_conf = np.where(cd >= cs, dino, ssdg)
    agree_mean_else_conf = np.where(agree, 0.5 * (dino + ssdg), more_conf)

    return {
        "prob_mean": 0.5 * (dino + ssdg),
        "logit_mean": sigmoid(0.5 * (logit(dino) + logit(ssdg))),
        "frame_confidence": weighted(dino, ssdg, cd + 0.05, cs + 0.05),
        "frame_entropy": weighted(dino, ssdg, ed + 0.05, es + 0.05),
        "agreement_gate": agree_mean_else_conf,
    }


def aggregate_video(rows):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["video_id"]].append(i)

    videos = []
    labels = []
    dino_mean = []
    ssdg_mean = []
    dino_std = []
    ssdg_std = []

    d = np.array([r["dino"] for r in rows], dtype=np.float64)
    s = np.array([r["ssdg"] for r in rows], dtype=np.float64)

    for vid, idx in groups.items():
        idx = np.asarray(idx, dtype=np.int64)
        ys = {rows[i]["label"] for i in idx}
        if len(ys) != 1:
            raise RuntimeError(f"Inconsistent labels inside video {vid}: {ys}")
        videos.append(vid)
        labels.append(next(iter(ys)))
        dino_mean.append(float(d[idx].mean()))
        ssdg_mean.append(float(s[idx].mean()))
        dino_std.append(float(d[idx].std()))
        ssdg_std.append(float(s[idx].std()))

    return {
        "videos": videos,
        "labels": np.asarray(labels, dtype=np.int64),
        "dino_mean": np.asarray(dino_mean),
        "ssdg_mean": np.asarray(ssdg_mean),
        "dino_std": np.asarray(dino_std),
        "ssdg_std": np.asarray(ssdg_std),
    }


def video_methods(v):
    d = v["dino_mean"]
    s = v["ssdg_mean"]
    sd = v["dino_std"]
    ss = v["ssdg_std"]

    cd = confidence_reliability(d)
    cs = confidence_reliability(s)
    ed = entropy_reliability(d)
    es = entropy_reliability(s)

    rd = (cd + 0.05) / (sd + 0.05)
    rs = (cs + 0.05) / (ss + 0.05)

    red = (ed + 0.05) / (sd + 0.05)
    res = (es + 0.05) / (ss + 0.05)
    stab_d = 1.0 / (sd + 0.05)
    stab_s = 1.0 / (ss + 0.05)

    agree = ((d >= 0.5) == (s >= 0.5))
    conf_pick = np.where(cd >= cs, d, s)

    return {
        "dino": d,
        "ssdg": s,
        "prob_mean": 0.5 * (d + s),
        "logit_mean": sigmoid(0.5 * (logit(d) + logit(s))),
        "video_confidence": weighted(d, s, cd + 0.05, cs + 0.05),
        "video_stability": weighted(d, s, stab_d, stab_s),
        "video_entropy_stability": weighted(d, s, red, res),
        "video_reliability": weighted(d, s, rd, rs),
        "video_agreement_gate": np.where(agree, 0.5 * (d + s), conf_pick),
    }


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    per_target = {}

    for target in args.targets:
        score_file = Path(args.root) / target / "frame_scores.csv"
        if not score_file.exists():
            raise FileNotFoundError(score_file)

        rows = load_scores(score_file)
        y_frame = np.asarray([r["label"] for r in rows], dtype=np.int64)
        d = np.asarray([r["dino"] for r in rows], dtype=np.float64)
        s = np.asarray([r["ssdg"] for r in rows], dtype=np.float64)

        frame_res = {}
        for name, scores in frame_methods(d, s).items():
            frame_res[name] = compute_metrics(y_frame.tolist(), scores.tolist())

        v = aggregate_video(rows)
        video_res = {}
        vm = video_methods(v)
        for name, scores in vm.items():
            video_res[name] = compute_metrics(v["labels"].tolist(), scores.tolist())
            all_rows.append(
                {
                    "target": target,
                    "method": name,
                    "video_auc": float(video_res[name]["auc"]),
                    "video_eer": float(video_res[name]["eer"]),
                }
            )

        per_target[target] = {
            "num_frames": len(rows),
            "num_videos": len(v["labels"]),
            "frame_results": frame_res,
            "video_results": video_res,
        }

        target_out = out / target
        target_out.mkdir(parents=True, exist_ok=True)
        with open(target_out / "video_dynamic_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            names = list(vm)
            writer.writerow(["video_id", "label", *names])
            for i, vid in enumerate(v["videos"]):
                writer.writerow([vid, int(v["labels"][i]), *[float(vm[n][i]) for n in names]])

    methods = sorted({r["method"] for r in all_rows})
    averages = {}
    for method in methods:
        subset = [r for r in all_rows if r["method"] == method]
        averages[method] = {
            "mean_video_auc": float(np.mean([r["video_auc"] for r in subset])),
            "mean_video_eer": float(np.mean([r["video_eer"] for r in subset])),
        }

    summary = {
        "protocol": (
            "All fusion rules are fixed and label-free. Target labels are used only "
            "after fusion to report metrics, never to compute fusion weights."
        ),
        "primary_method": "video_reliability",
        "primary_formula": (
            "r_m=(0.05+2*abs(mean_score_m-0.5))/(0.05+std_frame_score_m); "
            "fused=sum(r_m*mean_score_m)/sum(r_m)"
        ),
        "targets": per_target,
        "averages": averages,
    }
    (out / "dynamic_fusion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = out / "dynamic_fusion_summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Week 3 Dynamic Fusion Batch Results\n\n")
        f.write("Primary method: video_reliability (label-free).\n\n")
        f.write("| Method | " + " | ".join(args.targets) + " | Mean Video AUC | Mean Video EER |\n")
        f.write("|---|" + "---:|" * len(args.targets) + "---:|---:|\n")
        for method in methods:
            vals = []
            eers = []
            for target in args.targets:
                m = per_target[target]["video_results"][method]
                vals.append(float(m["auc"]))
                eers.append(float(m["eer"]))
            f.write(
                f"| {method} | "
                + " | ".join(f"{x:.4%}" for x in vals)
                + f" | {np.mean(vals):.4%} | {np.mean(eers):.4%} |\n"
            )

    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
