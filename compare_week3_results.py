import argparse
import csv
from pathlib import Path


def read_csv(path):
    rows = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rows[row['target_dataset']] = row
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', default='outputs_cross/cross_dataset_summary.csv')
    p.add_argument('--strong', default='outputs_week3_cross/strong/strong_cross_dataset_summary.csv')
    p.add_argument('--output', default='outputs_week3_cross/strong/week3_comparison.md')
    args = p.parse_args()

    baseline = read_csv(args.baseline)
    strong = read_csv(args.strong)
    targets = sorted(set(baseline) & set(strong))
    if not targets:
        raise RuntimeError('未找到可比较的目标数据集。')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write('# 第三周泛化增强对比\n\n')
        f.write('| 目标数据集 | Baseline Accuracy | Strong Accuracy | ΔAccuracy | Baseline AUC | Strong AUC | ΔAUC | Baseline EER | Strong EER | ΔEER |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for target in targets:
            b = baseline[target]
            s = strong[target]
            ba, sa = float(b['accuracy']), float(s['accuracy'])
            bauc, sauc = float(b['auc']), float(s['auc'])
            beer, seer = float(b['eer']), float(s['eer'])
            f.write(
                f'| {target} | {ba:.4%} | {sa:.4%} | {(sa-ba)*100:+.4f} pp | '
                f'{bauc:.6f} | {sauc:.6f} | {sauc-bauc:+.6f} | '
                f'{beer:.4%} | {seer:.4%} | {(seer-beer)*100:+.4f} pp |\n'
            )

    print(f'已生成: {out}')


if __name__ == '__main__':
    main()
