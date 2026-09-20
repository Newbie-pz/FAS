import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='outputs_week3_three_routes')
    return p.parse_args()


def safe_load(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []

    for target in ('CASIA', 'MSU-MFSD', 'Replay-Attack'):
        m = safe_load(root / 'dino_multisource_eval' / target / 'vfm_summary.json')
        if m:
            rows.append({'method':'DINOv2-Reg + Multi-source DG','target':target,'frame_or_sequence_auc':m['frame_metrics']['auc'],'video_auc':m['video_metrics']['auc'],'video_eer':m['video_metrics']['eer']})

        m = safe_load(root / 'ssdg_eval' / target / 'summary.json')
        if m:
            rows.append({'method':'DINOv2-Reg + SSDG-style','target':target,'frame_or_sequence_auc':m['frame_metrics']['auc'],'video_auc':m['video_metrics']['auc'],'video_eer':m['video_metrics']['eer']})

        m = safe_load(root / 'td_sf_eval' / target / 'summary.json')
        if m:
            rows.append({'method':'FAS-TD-SF-inspired','target':target,'frame_or_sequence_auc':m['sequence_metrics']['auc'],'video_auc':m['video_metrics']['auc'],'video_eer':m['video_metrics']['eer']})

    if not rows:
        raise RuntimeError(f'No results found under {root}')

    csv_path = root / 'week3_three_routes_summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = root / 'week3_three_routes_summary.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# 第三周三路线实验汇总\n\n')
        f.write('| 方法 | Target | Frame/Sequence AUC | Video AUC | Video EER |\n')
        f.write('|---|---|---:|---:|---:|\n')
        for r in rows:
            f.write(f"| {r['method']} | {r['target']} | {float(r['frame_or_sequence_auc']):.4%} | {float(r['video_auc']):.4%} | {float(r['video_eer']):.4%} |\n")

        f.write('\n## 说明\n\n')
        f.write('- DINOv2-Reg + Multi-source DG：多源强骨干路线。\n')
        f.write('- SSDG-style：在 DINOv2-Reg 上移植 single-side live-domain adversarial 与 unbalanced triplet。\n')
        f.write('- FAS-TD-SF-inspired：三帧时序 + 空间梯度；缺少 PRNet 虚拟深度，因此不是原论文完整复现。\n')

    print(f'已生成: {csv_path}')
    print(f'已生成: {md_path}')


if __name__ == '__main__':
    main()
