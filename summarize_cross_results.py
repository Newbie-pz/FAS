import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='outputs_cross')
    p.add_argument('--csv_name', default='cross_dataset_summary.csv')
    p.add_argument('--md_name', default='cross_dataset_summary.md')
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []

    for path in sorted(root.glob('*/cross_dataset_summary.json')):
        with open(path, 'r', encoding='utf-8') as f:
            m = json.load(f)
        rows.append({
            'source_dataset': m.get('source_dataset', ''),
            'target_dataset': m.get('target_dataset', ''),
            'accuracy': m.get('accuracy', ''),
            'auc': m.get('auc', ''),
            'eer': m.get('eer', ''),
            'eer_threshold': m.get('eer_threshold', ''),
            'num_images': m.get('num_images', ''),
            'num_live': m.get('num_live', ''),
            'num_spoof': m.get('num_spoof', ''),
        })

    if not rows:
        raise RuntimeError(f'在 {root} 下未找到 cross_dataset_summary.json')

    csv_path = root / args.csv_name
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = root / args.md_name
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# 第二周跨数据集实验结果汇总\n\n')
        f.write('| 源数据集 | 目标数据集 | Accuracy | AUC | EER | EER Threshold | 测试图像数 | Live | Spoof |\n')
        f.write('|---|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in rows:
            f.write(
                f"| {r['source_dataset']} | {r['target_dataset']} | "
                f"{float(r['accuracy']):.4%} | {float(r['auc']):.6f} | "
                f"{float(r['eer']):.4%} | {float(r['eer_threshold']):.6f} | "
                f"{r['num_images']} | {r['num_live']} | {r['num_spoof']} |\n"
            )

    print(f'已生成: {csv_path}')
    print(f'已生成: {md_path}')


if __name__ == '__main__':
    main()
