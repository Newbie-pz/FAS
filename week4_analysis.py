import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


TARGETS = ["CASIA", "MSU-MFSD", "Replay-Attack"]


def parse_args():
    p = argparse.ArgumentParser(description="Week 4 unified comparison, ablation-style analysis and visualization.")
    p.add_argument("--three_routes", default="outputs_week3_three_routes")
    p.add_argument("--ensemble_root", default="outputs_week3_ensemble")
    p.add_argument("--dynamic_root", default="outputs_week3_dynamic_fusion")
    p.add_argument("--parallel_root", default="outputs_week3_parallel_sweep")
    p.add_argument("--output", default="outputs_week4_analysis")
    return p.parse_args()


def pct(s):
    s = str(s).strip()
    if not s or s.upper() == "NA":
        return None
    return float(s.rstrip("%")) / 100.0


def parse_md_table(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    tables = []
    i = 0
    while i < len(lines) - 1:
        if lines[i].strip().startswith("|") and lines[i + 1].strip().startswith("|---"):
            headers = [x.strip() for x in lines[i].strip().strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                vals = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if len(vals) == len(headers):
                    rows.append(dict(zip(headers, vals)))
                i += 1
            tables.append(rows)
        else:
            i += 1
    return tables


def first_table(path):
    tables = parse_md_table(path)
    return tables[0] if tables else []


def save_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def grouped_bar(path, labels, series, ylabel, title, ylim=(0.0, 1.0)):
    x = np.arange(len(labels))
    n = len(series)
    width = min(0.8 / max(n, 1), 0.22)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.7), 5.2))
    offsets = (np.arange(n) - (n - 1) / 2.0) * width
    for off, (name, vals) in zip(offsets, series.items()):
        ax.bar(x + off, vals, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def bar(path, labels, vals, ylabel, title, ylim=(0.0, 1.0), rotate=20):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.35), 5.2))
    bars = ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v*100:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def load_parallel_rows(root):
    table = first_table(Path(root) / "parallel_sweep_summary.md")
    out = []
    for r in table:
        out.append({
            "variant": r["Variant"],
            "CASIA": pct(r["CASIA"]),
            "MSU-MFSD": pct(r["MSU-MFSD"]),
            "Replay-Attack": pct(r["Replay-Attack"]),
            "mean_auc": pct(r["Mean Video AUC"]),
            "mean_eer": pct(r["Mean Video EER"]),
            "source_worst": pct(r["Mean Source-Worst AUC"]),
        })
    return out


def load_three_route_rows(root):
    table = first_table(Path(root) / "week3_three_routes_summary.md")
    grouped = {}
    for r in table:
        method = r["方法"]
        target = r["Target"]
        grouped.setdefault(method, {})[target] = {
            "auc": pct(r["Video AUC"]),
            "eer": pct(r["Video EER"]),
        }
    return grouped


def load_dynamic_rows(root):
    table = first_table(Path(root) / "dynamic_fusion_summary.md")
    out = []
    for r in table:
        out.append({
            "method": r["Method"],
            "CASIA": pct(r["CASIA"]),
            "MSU-MFSD": pct(r["MSU-MFSD"]),
            "Replay-Attack": pct(r["Replay-Attack"]),
            "mean_auc": pct(r["Mean Video AUC"]),
            "mean_eer": pct(r["Mean Video EER"]),
        })
    return out


def load_video_scores(path):
    y, s = [], []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            y.append(int(r["label"]))
            s.append(float(r["live_score"]))
    return np.asarray(y), np.asarray(s)


def plot_final_roc(parallel_root, out):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    aucs = {}
    for target in TARGETS:
        p = Path(parallel_root) / "dino_f12_lr5e6_eval" / target / "video_scores.csv"
        if not p.exists():
            continue
        y, s = load_video_scores(p)
        fpr, tpr, _ = roc_curve(y, s)
        a = auc(fpr, tpr)
        aucs[target] = float(a)
        ax.plot(fpr, tpr, label=f"{target} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Final Frozen DINOv2-Reg: Video-level ROC")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(out) / "06_final_model_video_roc.png", dpi=220)
    plt.close(fig)
    return aucs


def plot_final_confusions(parallel_root, out):
    for target in TARGETS:
        p = Path(parallel_root) / "dino_f12_lr5e6_eval" / target / "vfm_summary.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        cm = np.asarray(data["video_metrics"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4.8, 4.3))
        ax.imshow(cm)
        ax.set_xticks([0, 1], ["Spoof", "Live"])
        ax.set_yticks([0, 1], ["Spoof", "Live"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Frozen DINOv2-Reg: {target}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(Path(out) / f"07_confusion_{target}.png", dpi=220)
        plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    three = load_three_route_rows(args.three_routes)
    parallel = load_parallel_rows(args.parallel_root)
    dynamic = load_dynamic_rows(args.dynamic_root)

    parallel_by_name = {r["variant"]: r for r in parallel}
    final = parallel_by_name.get("dino_f12_lr5e6")
    if final is None:
        raise RuntimeError("Missing dino_f12_lr5e6 in parallel sweep summary.")

    # 1) Unified Week-4 method comparison.
    selected_methods = [
        "Multi-source DG + ResNet18",
        "DINOv2-Reg + Multi-source DG",
        "DINOv2-Reg + SSDG-style",
        "FAS-TD-SF-inspired",
    ]
    unified = []
    for method in selected_methods:
        if method not in three:
            continue
        row = {"method": method}
        vals = []
        eers = []
        for t in TARGETS:
            m = three[method].get(t, {})
            row[t] = m.get("auc")
            if row[t] is not None:
                vals.append(row[t])
            if m.get("eer") is not None:
                eers.append(m["eer"])
        row["mean_auc"] = float(np.mean(vals)) if vals else None
        row["mean_eer"] = float(np.mean(eers)) if eers else None
        unified.append(row)

    unified.append({
        "method": "Frozen DINOv2-Reg (Week 3 Final)",
        "CASIA": final["CASIA"],
        "MSU-MFSD": final["MSU-MFSD"],
        "Replay-Attack": final["Replay-Attack"],
        "mean_auc": final["mean_auc"],
        "mean_eer": final["mean_eer"],
    })

    save_csv(
        out / "week4_method_comparison.csv",
        unified,
        ["method", *TARGETS, "mean_auc", "mean_eer"],
    )

    grouped_bar(
        out / "01_method_comparison_video_auc.png",
        [r["method"] for r in unified],
        {
            "CASIA": [r["CASIA"] for r in unified],
            "MSU-MFSD": [r["MSU-MFSD"] for r in unified],
            "Replay-Attack": [r["Replay-Attack"] for r in unified],
        },
        "Video AUC",
        "Cross-domain FAS Method Comparison",
        ylim=(0.40, 1.00),
    )

    bar(
        out / "02_method_mean_video_auc.png",
        [r["method"] for r in unified],
        [r["mean_auc"] for r in unified],
        "Mean Video AUC",
        "Average Cross-domain Performance",
        ylim=(0.55, 0.90),
    )

    # 2) Tuning-depth diagnostic: f10 / f11 / f12.
    f10 = parallel_by_name.get("dino_f10_lr2e6")
    f11 = three.get("DINOv2-Reg + Multi-source DG")
    if f10 and f11:
        f11_vals = [f11[t]["auc"] for t in TARGETS]
        depth_series = {
            "freeze=10": [f10[t] for t in TARGETS],
            "freeze=11": f11_vals,
            "freeze=12": [final[t] for t in TARGETS],
        }
        grouped_bar(
            out / "03_tuning_depth_comparison.png",
            TARGETS,
            depth_series,
            "Video AUC",
            "DINOv2 Tuning-depth Diagnostic",
            ylim=(0.60, 0.95),
        )

    # 3) SSDG parameter sensitivity.
    ssdg_rows = [r for r in parallel if r["variant"].startswith("ssdg_")]
    if ssdg_rows:
        grouped_bar(
            out / "04_ssdg_parameter_sensitivity.png",
            [r["variant"] for r in ssdg_rows],
            {t: [r[t] for r in ssdg_rows] for t in TARGETS},
            "Video AUC",
            "SSDG-style Parameter Sensitivity",
            ylim=(0.60, 1.00),
        )

    # 4) Fusion strategy analysis.
    if dynamic:
        bar(
            out / "05_fusion_mean_video_auc.png",
            [r["method"] for r in dynamic],
            [r["mean_auc"] for r in dynamic],
            "Mean Video AUC",
            "DINOv2 + SSDG Fusion Strategy Analysis",
            ylim=(0.78, 0.85),
            rotate=30,
        )

    # 5) Source-target gap for parallel sweep variants.
    gap_rows = [r for r in parallel if r["source_worst"] is not None and r["mean_auc"] is not None]
    if gap_rows:
        x = np.arange(len(gap_rows))
        fig, ax = plt.subplots(figsize=(max(9, len(gap_rows) * 1.5), 5.3))
        ax.plot(x, [r["source_worst"] for r in gap_rows], marker="o", label="Source worst-domain AUC")
        ax.plot(x, [r["mean_auc"] for r in gap_rows], marker="o", label="Unseen-target mean AUC")
        ax.set_xticks(x)
        ax.set_xticklabels([r["variant"] for r in gap_rows], rotation=25, ha="right")
        ax.set_ylabel("AUC")
        ax.set_title("Source Validation vs Unseen-target Generalization")
        ax.set_ylim(0.65, 1.01)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "06_source_target_generalization_gap.png", dpi=220)
        plt.close(fig)

    # 6) Final model ROC and confusion matrices.
    final_rocs = plot_final_roc(args.parallel_root, out)
    plot_final_confusions(args.parallel_root, out)

    # 7) Summary markdown.
    report = []
    report.append("# Week 4 Unified Analysis\n")
    report.append("## Week 3 frozen result\n")
    report.append("Final model: **Frozen DINOv2-Reg multi-source DG**.\n")
    report.append("| Target | Video AUC |")
    report.append("|---|---:|")
    for t in TARGETS:
        report.append(f"| {t} | {final[t]:.2%} |")
    report.append(f"| **Average** | **{final['mean_auc']:.2%}** |\n")

    report.append("## Analysis assets\n")
    assets = [
        ("01_method_comparison_video_auc.png", "Unified cross-domain method comparison"),
        ("02_method_mean_video_auc.png", "Average Video AUC comparison"),
        ("03_tuning_depth_comparison.png", "DINOv2 tuning-depth diagnostic"),
        ("04_ssdg_parameter_sensitivity.png", "SSDG-style parameter sensitivity"),
        ("05_fusion_mean_video_auc.png", "Fusion strategy comparison"),
        ("06_source_target_generalization_gap.png", "Source-vs-target generalization gap"),
        ("06_final_model_video_roc.png", "Final model video-level ROC curves"),
        ("07_confusion_CASIA.png", "CASIA confusion matrix"),
        ("07_confusion_MSU-MFSD.png", "MSU-MFSD confusion matrix"),
        ("07_confusion_Replay-Attack.png", "Replay-Attack confusion matrix"),
    ]
    for name, desc in assets:
        if (out / name).exists():
            report.append(f"- `{name}`: {desc}")

    report.append("\n## Interpretation notes\n")
    report.append(
        "- The freeze=10/11/12 comparison is a diagnostic tuning-depth comparison rather than a perfectly controlled ablation, "
        "because the historical runs used different backbone learning rates."
    )
    report.append(
        "- The SSDG-style variants are suitable for parameter-sensitivity analysis; the frozen SSDG configuration is especially useful "
        "to illustrate target-dependent gains and failure on Replay-Attack."
    )
    report.append(
        "- Fusion results should be reported as an analysis rather than replacing the frozen Week-3 main result."
    )
    report.append(
        "- Source validation AUC near 100% together with substantially lower unseen-target AUC directly visualizes the source-domain overfitting/generalization gap."
    )
    if final_rocs:
        report.append(
            "- Video-level ROC curves are regenerated from the saved final-model video scores, so no retraining is required."
        )

    (out / "week4_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print((out / "week4_summary.md").read_text(encoding="utf-8"))
    print(f"\nAll Week-4 assets saved to: {out}")


if __name__ == "__main__":
    main()
