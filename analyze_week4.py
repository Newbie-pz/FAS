import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np


METHODS = {
    'Baseline': {
        'cross_csv': Path('outputs_cross/cross_dataset_summary.csv'),
        'source_json': Path('outputs/OULU-NPU/test_metrics.json'),
        'description': '基础 ResNet18 + 基础数据增强',
    },
    'Strong': {
        'cross_csv': Path('outputs_week3_cross/strong/strong_cross_dataset_summary.csv'),
        'source_json': Path('outputs_week3/strong/OULU-NPU/test_metrics.json'),
        'description': '强数据增强',
    },
    'Appearance': {
        'cross_csv': Path('outputs_week3_cross/appearance/appearance_cross_dataset_summary.csv'),
        'source_json': Path('outputs_week3/appearance/OULU-NPU/test_metrics.json'),
        'description': '外观/成像风格随机化',
    },
    'MixStyle': {
        'cross_csv': Path('outputs_week3_cross/mixstyle/mixstyle_cross_dataset_summary.csv'),
        'source_json': Path('outputs_week3/mixstyle/OULU-NPU/test_metrics.json'),
        'description': '浅层特征统计混合',
    },
    'Fourier': {
        'cross_csv': Path('outputs_week3_cross/fourier/fourier_cross_dataset_summary.csv'),
        'source_json': Path('outputs_week3/fourier/OULU-NPU/test_metrics.json'),
        'description': '同类样本低频幅度谱混合',
    },
}

TARGET_ORDER = ['CASIA', 'MSU-MFSD', 'Replay-Attack']
METRICS = ['accuracy', 'auc', 'eer']
OUT = Path('outputs_week4')


def configure_plot_font():
    """Use an installed CJK font when available; otherwise use English plot text."""
    candidates = [
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'Source Han Sans SC',
        'Source Han Sans CN',
        'WenQuanYi Micro Hei',
        'SimHei',
        'Microsoft YaHei',
        'Arial Unicode MS',
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f'绘图字体: {name}（支持中文）')
            return True

    print('未检测到可用中文字体，图表标题与标签自动使用英文；实验数据不受影响。')
    return False


PLOT_CHINESE = configure_plot_font()


def plot_text(chinese, english):
    return chinese if PLOT_CHINESE else english


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'缺少第四周分析所需文件: {path}')


def read_cross_csv(path):
    require_file(path)
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
    missing = [t for t in TARGET_ORDER if t not in rows]
    if missing:
        raise RuntimeError(f'{path} 缺少目标域结果: {missing}')
    return rows


def read_source_json(path):
    require_file(path)
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
    n_methods = len(METHODS)
    width = 0.80 / n_methods
    fig, ax = plt.subplots(figsize=(12, 6))

    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width
    for i, method in enumerate(METHODS):
        values = [results[method][target][metric] * 100 for target in TARGET_ORDER]
        ax.bar(x + offsets[i], values, width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(plot_text(f'跨数据集 {ylabel} 对比', f'Cross-dataset {ylabel} Comparison'))
    ax.legend(ncol=3)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=220)
    plt.close(fig)


def save_macro_bar(metric, macro, ylabel, filename, lower_is_better=False):
    methods = list(METHODS)
    values = [macro[m][metric] * 100 for m in methods]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(methods, values)
    ax.set_ylabel(ylabel)
    ax.set_title(plot_text(f'三个目标域宏平均 {ylabel}', f'Macro-average {ylabel} Across Three Target Domains'))
    ax.grid(axis='y', alpha=0.25)

    best_idx = int(np.argmin(values) if lower_is_better else np.argmax(values))
    for i, (bar, value) in enumerate(zip(bars, values)):
        mark = ' *' if i == best_idx else ''
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{value:.2f}{mark}',
            ha='center',
            va='bottom',
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=220)
    plt.close(fig)


def save_delta_heatmap(cross):
    methods = [m for m in METHODS if m != 'Baseline']
    baseline = cross['Baseline']

    # AUC 越大越好；EER 转换成“降低量”，越大越好。
    data = []
    row_labels = []
    for method in methods:
        auc_delta = [
            (cross[method][t]['auc'] - baseline[t]['auc']) * 100
            for t in TARGET_ORDER
        ]
        eer_gain = [
            (baseline[t]['eer'] - cross[method][t]['eer']) * 100
            for t in TARGET_ORDER
        ]
        data.append(auc_delta)
        row_labels.append(f'{method} ΔAUC')
        data.append(eer_gain)
        row_labels.append(
            f'{method} EER降低' if PLOT_CHINESE else f'{method} EER Gain'
        )

    arr = np.asarray(data, dtype=float)
    bound = max(abs(arr.min()), abs(arr.max()), 1.0)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(arr, aspect='auto', cmap='coolwarm', vmin=-bound, vmax=bound)
    ax.set_xticks(np.arange(len(TARGET_ORDER)))
    ax.set_xticklabels(TARGET_ORDER)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(plot_text(
        '相对 Baseline 的跨域提升矩阵（百分点）',
        'Cross-domain Improvement vs Baseline (percentage points)',
    ))

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f'{arr[i, j]:+.2f}', ha='center', va='center', fontsize=9)

    fig.colorbar(
        im,
        ax=ax,
        label=plot_text('提升百分点（正值为改善）', 'Improvement (pp; positive is better)'),
    )
    fig.tight_layout()
    fig.savefig(OUT / 'improvement_heatmap.png', dpi=220)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    cross = {}
    source = {}
    for method, paths in METHODS.items():
        cross[method] = read_cross_csv(paths['cross_csv'])
        source[method] = read_source_json(paths['source_json'])

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

    macro = {}
    for method in METHODS:
        macro[method] = {
            metric: float(np.mean([cross[method][t][metric] for t in TARGET_ORDER]))
            for metric in METRICS
        }

    with open(OUT / 'week4_macro_summary.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'accuracy', 'auc', 'eer'])
        writer.writeheader()
        for method in METHODS:
            writer.writerow({'method': method, **macro[method]})

    baseline = cross['Baseline']

    consistency = {}
    for method in METHODS:
        if method == 'Baseline':
            consistency[method] = 0
            continue
        count = 0
        for target in TARGET_ORDER:
            m = cross[method][target]
            b = baseline[target]
            if (
                m['accuracy'] > b['accuracy']
                and m['auc'] > b['auc']
                and m['eer'] < b['eer']
            ):
                count += 1
        consistency[method] = count

    domain_best = []
    for target in TARGET_ORDER:
        best_acc = max(METHODS, key=lambda m: cross[m][target]['accuracy'])
        best_auc = max(METHODS, key=lambda m: cross[m][target]['auc'])
        best_eer = min(METHODS, key=lambda m: cross[m][target]['eer'])
        domain_best.append({
            'target_dataset': target,
            'best_accuracy_method': best_acc,
            'best_accuracy': cross[best_acc][target]['accuracy'],
            'best_auc_method': best_auc,
            'best_auc': cross[best_auc][target]['auc'],
            'best_eer_method': best_eer,
            'best_eer': cross[best_eer][target]['eer'],
        })

    with open(OUT / 'week4_domain_best.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(domain_best[0].keys()))
        writer.writeheader()
        writer.writerows(domain_best)

    best_macro_acc = max(METHODS, key=lambda m: macro[m]['accuracy'])
    best_macro_auc = max(METHODS, key=lambda m: macro[m]['auc'])
    best_macro_eer = min(METHODS, key=lambda m: macro[m]['eer'])

    with open(OUT / 'week4_analysis.md', 'w', encoding='utf-8') as f:
        f.write('# 第四周：实验对比、消融与结果分析\n\n')
        f.write('本周不再训练新的模型，而是统一分析前三周已经完成的五组方法。所有方法使用相同的 OULU-NPU 源域、相同 Subject-disjoint 划分和相同三个跨域目标数据集。\n\n')

        f.write('## 一、方法与单变量改动\n\n')
        f.write('| 方法 | 相对 Baseline 的主要改动 |\n')
        f.write('|---|---|\n')
        for method, config in METHODS.items():
            f.write(f"| {method} | {config['description']} |\n")

        f.write('\n这种设计可以从“单变量机制对照”的角度观察不同泛化策略对跨域性能的影响。\n\n')

        f.write('## 二、同域性能对比\n\n')
        f.write('| 方法 | Accuracy | AUC | EER | EER Threshold |\n')
        f.write('|---|---:|---:|---:|---:|\n')
        for method in METHODS:
            m = source[method]
            f.write(
                f"| {method} | {m['accuracy']:.4%} | {m['auc']:.6f} | "
                f"{m['eer']:.4%} | {m['eer_threshold']:.6f} |\n"
            )

        f.write('\n## 三、跨数据集完整对比\n\n')
        f.write('| 目标域 | 方法 | Accuracy | AUC | EER | ΔAccuracy | ΔAUC | EER降低量 |\n')
        f.write('|---|---|---:|---:|---:|---:|---:|---:|\n')
        for target in TARGET_ORDER:
            for method in METHODS:
                m = cross[method][target]
                b = baseline[target]
                f.write(
                    f"| {target} | {method} | {m['accuracy']:.4%} | {m['auc']:.4%} | "
                    f"{m['eer']:.4%} | {(m['accuracy']-b['accuracy'])*100:+.4f} pp | "
                    f"{(m['auc']-b['auc'])*100:+.4f} pp | {(b['eer']-m['eer'])*100:+.4f} pp |\n"
                )

        f.write('\n## 四、三个目标域宏平均\n\n')
        f.write('| 方法 | 平均 Accuracy | 平均 AUC | 平均 EER | 三域同时改善数 |\n')
        f.write('|---|---:|---:|---:|---:|\n')
        for method in METHODS:
            m = macro[method]
            count_text = '-' if method == 'Baseline' else f"{consistency[method]}/3"
            f.write(
                f"| {method} | {m['accuracy']:.4%} | {m['auc']:.4%} | "
                f"{m['eer']:.4%} | {count_text} |\n"
            )

        f.write('\n宏平均最优结果：\n\n')
        f.write(f'- Accuracy 最优：{best_macro_acc}，{macro[best_macro_acc]["accuracy"]:.4%}。\n')
        f.write(f'- AUC 最优：{best_macro_auc}，{macro[best_macro_auc]["auc"]:.4%}。\n')
        f.write(f'- EER 最优：{best_macro_eer}，{macro[best_macro_eer]["eer"]:.4%}。\n')

        f.write('\n## 五、各目标域最优方法\n\n')
        f.write('| 目标域 | Accuracy 最优 | AUC 最优 | EER 最优 |\n')
        f.write('|---|---|---|---|\n')
        for row in domain_best:
            f.write(
                f"| {row['target_dataset']} | {row['best_accuracy_method']} "
                f"({row['best_accuracy']:.4%}) | {row['best_auc_method']} "
                f"({row['best_auc']:.4%}) | {row['best_eer_method']} "
                f"({row['best_eer']:.4%}) |\n"
            )

        f.write('\n## 六、消融式机制分析\n\n')
        f.write('1. **像素级强增强并不等于稳定域泛化。** Strong 在 CASIA 上提升显著，但在 MSU-MFSD 和 Replay-Attack 的 AUC/EER 没有形成一致改善，说明过强裁切、模糊或遮挡可能同时破坏有用的活体纹理。\n')
        f.write('2. **外观随机化主要缓解部分成像风格差异。** Appearance 在 Replay-Attack 上明显改善，但在 MSU-MFSD 上退化，说明颜色、亮度和清晰度随机化只能覆盖一部分域偏移。\n')
        f.write('3. **MixStyle 能改善部分浅层风格偏移，但稳定性仍不足。** 它在 CASIA 和 Replay-Attack 上有改善，但 MSU-MFSD 的 AUC/EER 低于 Baseline。\n')
        f.write('4. **Fourier 是当前唯一实现三域一致改善的方法。** 相比 Baseline，三个目标域均同时满足 Accuracy 上升、AUC 上升、EER 下降，说明针对低频幅度统计进行扰动比单纯扩大像素增强强度更适合当前任务。\n')
        f.write('5. **同域高性能与跨域高泛化并不等价。** 所有方法在 OULU-NPU 同域仍保持较高 AUC，但跨域差异巨大，因此最终评价必须以跨域 AUC/EER 为主。\n')

        f.write('\n## 七、Fourier 相对 Baseline 的关键提升\n\n')
        f.write('| 目标域 | Accuracy 提升 | AUC 提升 | EER 降低 |\n')
        f.write('|---|---:|---:|---:|\n')
        for target in TARGET_ORDER:
            m = cross['Fourier'][target]
            b = baseline[target]
            f.write(
                f"| {target} | {(m['accuracy']-b['accuracy'])*100:+.4f} pp | "
                f"{(m['auc']-b['auc'])*100:+.4f} pp | {(b['eer']-m['eer'])*100:+.4f} pp |\n"
            )

        f.write('\n## 八、第四周结论\n\n')
        f.write('五组方法的统一对比表明，普通像素增强、外观随机化和特征统计混合都只能在部分目标域取得收益，而 Fourier 低频幅度扰动是当前唯一在 CASIA、MSU-MFSD 和 Replay-Attack 三个未见目标域上同时改善 Accuracy、AUC 与 EER 的方案。其三个目标域宏平均 AUC 最高、宏平均 EER 最低，因此将 Fourier 作为本项目当前推荐的泛化增强方法。\n\n')
        f.write('同时需要保留实验边界：Fourier 并未完全解决跨域问题，尤其 CASIA 的绝对 AUC 仍低于 0.5。因此最终报告应表述为“跨域泛化得到稳定改善”，而不是“已经解决跨域泛化”。\n')

    save_grouped_bar('accuracy', cross, 'Accuracy (%)', 'accuracy_comparison.png')
    save_grouped_bar('auc', cross, 'AUC (%)', 'auc_comparison.png')
    save_grouped_bar('eer', cross, 'EER (%)', 'eer_comparison.png')
    save_macro_bar('accuracy', macro, 'Accuracy (%)', 'macro_accuracy.png')
    save_macro_bar('auc', macro, 'AUC (%)', 'macro_auc.png')
    save_macro_bar('eer', macro, 'EER (%)', 'macro_eer.png', lower_is_better=True)
    save_delta_heatmap(cross)

    outputs = [
        'week4_all_results.csv',
        'week4_macro_summary.csv',
        'week4_domain_best.csv',
        'week4_analysis.md',
        'accuracy_comparison.png',
        'auc_comparison.png',
        'eer_comparison.png',
        'macro_accuracy.png',
        'macro_auc.png',
        'macro_eer.png',
        'improvement_heatmap.png',
    ]
    for name in outputs:
        print(f'已生成: {OUT / name}')


if __name__ == '__main__':
    main()
