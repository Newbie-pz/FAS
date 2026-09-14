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
    p.add_argument('--candidate', default='outputs_week3_cross/strong/strong_cross_dataset_summary.csv')
    p.add_argument('--method_name', default='Strong')
    p.add_argument('--output', default='outputs_week3_cross/strong/week3_comparison.md')
    args = p.parse_args()

    baseline = read_csv(args.baseline)
    candidate = read_csv(args.candidate)
    targets = sorted(set(baseline) & set(candidate))
    if not targets:
        raise RuntimeError('未找到可比较的目标数据集。')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(f'# 第三周泛化增强对比：{args.method_name}\n\n')
        f.write(
            f'| 目标数据集 | Baseline Accuracy | {args.method_name} Accuracy | ΔAccuracy | '
            f'Baseline AUC | {args.method_name} AUC | ΔAUC | '
            f'Baseline EER | {args.method_name} EER | ΔEER |\n'
        )
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for target in targets:
            b = baseline[target]
            c = candidate[target]
            ba, ca = float(b['accuracy']), float(c['accuracy'])
            bauc, cauc = float(b['auc']), float(c['auc'])
            beer, ceer = float(b['eer']), float(c['eer'])
            f.write(
                f'| {target} | {ba:.4%} | {ca:.4%} | {(ca-ba)*100:+.4f} pp | '
                f'{bauc:.6f} | {cauc:.6f} | {cauc-bauc:+.6f} | '
                f'{beer:.4%} | {ceer:.4%} | {(ceer-beer)*100:+.4f} pp |\n'
            )

    print(f'已生成: {out}')


if __name__ == '__main__':
    main()
