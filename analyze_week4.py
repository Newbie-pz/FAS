import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    'Baseline': {
        'cross_csv': Path('outputs_cross/cross_dataset_summary.csv'),
        'source_json': Path('outputs/OULU-NPU/test_metrics.json'),
    },
    'Strong': {
        'cross_csv': Path('outputs_week3_cross/strong/strong_cross_dataset_summary.csv'),
        'source_json': Path('outputs_week3/strong/OULU-NPU/test_metrics.json'),
    },
    'Appearance': {
        'cross_csv': Path('outputs_week3_cross/appearance/appearance_cross_dataset_summary.csv'),
        'source_json': Path('outputs_week3/appearance/OULU-NPU/test_metrics.json'),
    },
}

TARGET_ORDER = ['CASIA', 'MSU-MFSD', 'Replay-Attack']
OUT = Path('outputs_week4')


def read_cross_csv(path):
    rows = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rows[row['target_dataset']] = {
                'accuracy': float(row['accuracy']),
                'auc': float(row['auc']),
                'eer': float(row['eer']),
                'eer_threshold': float(row['eer_threshold']),
                'num_images': int(row['num_images']),
            }
    return rows


def read_source_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        m = json.load(f)
    return {
        'accuracy': float(m['accuracy']),
        'auc': float(m['auc']),
        'eer': float(m['eer']),
        'eer_threshold': float(m['eer_threshold']),
    }


def save_grouped_bar(metric, results, ylabel, filename):
    x = np.arange(len(TARGET_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, method in enumerate(METHODS):
        values = [results[method][target][metric] * 100 for target in TARGET_ORDER]
        ax.bar(x + (i - 1) * width, values, width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(f'跨数据集 {ylabel} 对比')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    cross = {}
    source = {}
    for method, paths in METHODS.items():
        cross[method] = read_cross_csv(paths['cross_csv'])
        source[method] = read_source_json(paths['source_json'])

    # 统一长表
    rows = []
    for method in METHODS:
        rows.append({
            'method': method,
            'domain': 'OULU-NPU',
            'setting': '同域',
            **source[method],
        })
        for target in TARGET_ORDER:
            rows.append({
                'method': method,
                'domain': target,
                'setting': '跨域',
                **cross[method][target],
            })

    with open(OUT / 'week4_all_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ['method', 'domain', 'setting', 'accuracy', 'auc', 'eer', 'eer_threshold']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    # 宏平均
    macro = {}
    for method in METHODS:
        macro[method] = {
            'accuracy': np.mean([cross[method][t]['accuracy'] for t in TARGET_ORDER]),
            'auc': np.mean([cross[method][t]['auc'] for t in TARGET_ORDER]),
            'eer': np.mean([cross[method][t]['eer'] for t in TARGET_ORDER]),
        }

    baseline = cross['Baseline']

    with open(OUT / 'week4_analysis.md', 'w', encoding='utf-8') as f:
        f.write('# 第四周实验对比与分析\n\n')
        f.write('## 一、同域性能对比\n\n')
        f.write('| 方法 | Accuracy | AUC | EER | EER Threshold |\n')
        f.write('|---|---:|---:|---:|---:|\n')
        for method in METHODS:
            m = source[method]
            f.write(
                f"| {method} | {m['accuracy']:.4%} | {m['auc']:.6f} | "
                f"{m['eer']:.4%} | {m['eer_threshold']:.6f} |\n"
            )

        f.write('\n## 二、跨数据集结果对比\n\n')
        f.write('| 目标域 | 方法 | Accuracy | AUC | EER | ΔAccuracy vs Baseline | ΔAUC vs Baseline | ΔEER vs Baseline |\n')
        f.write('|---|---|---:|---:|---:|---:|---:|---:|\n')
        for target in TARGET_ORDER:
            for method in METHODS:
                m = cross[method][target]
                b = baseline[target]
                f.write(
                    f"| {target} | {method} | {m['accuracy']:.4%} | {m['auc']:.6f} | "
                    f"{m['eer']:.4%} | {(m['accuracy']-b['accuracy'])*100:+.4f} pp | "
                    f"{m['auc']-b['auc']:+.6f} | {(m['eer']-b['eer'])*100:+.4f} pp |\n"
                )

        f.write('\n## 三、三个目标域宏平均\n\n')
        f.write('| 方法 | 平均 Accuracy | 平均 AUC | 平均 EER |\n')
        f.write('|---|---:|---:|---:|\n')
        for method in METHODS:
            m = macro[method]
            f.write(f"| {method} | {m['accuracy']:.4%} | {m['auc']:.6f} | {m['eer']:.4%} |\n")

        f.write('\n## 四、主要观察\n\n')
        f.write('1. Strong Augmentation 在 CASIA 上改善最明显，但在 MSU-MFSD 和 Replay-Attack 上没有形成一致的 AUC/EER 提升。\n')
        f.write('2. Appearance Augmentation 在 Replay-Attack 上同时提升 Accuracy、AUC 并降低 EER，但在 MSU-MFSD 上退化。\n')
        f.write('3. MSU-MFSD 的 AUC 与 EER 仍以 Baseline 最优，说明更强或更复杂的数据增强不一定提高跨域泛化。\n')
        f.write('4. 三种方法在不同目标域上的最优策略不同，表明 FAS 的域偏移具有明显异质性。\n')
        f.write('5. 跨域 EER Threshold 与 0.5 经常存在明显偏差，说明目标域上还存在显著的置信度校准问题。\n')

        f.write('\n## 五、第四周建议结论\n\n')
        f.write('现有实验已经构成一组清晰的泛化增强对比：Baseline、Strong Augmentation 和 Appearance Augmentation。结果表明，数据增强可以改善部分目标域，但不存在对三个目标域均稳定占优的单一策略。后续报告应强调“域依赖性”和“增强策略选择性”，而不是只汇报某个目标数据集上的最好数值。\n')

    save_grouped_bar('accuracy', cross, 'Accuracy (%)', 'accuracy_comparison.png')
    save_grouped_bar('auc', cross, 'AUC (%)', 'auc_comparison.png')
    save_grouped_bar('eer', cross, 'EER (%)', 'eer_comparison.png')

    print(f'已生成: {OUT / "week4_all_results.csv"}')
    print(f'已生成: {OUT / "week4_analysis.md"}')
    print(f'已生成: {OUT / "accuracy_comparison.png"}')
    print(f'已生成: {OUT / "auc_comparison.png"}')
    print(f'已生成: {OUT / "eer_comparison.png"}')


if __name__ == '__main__':
    main()
