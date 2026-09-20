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
        m = safe_load(root / 'route1_resnet' / target / 'target_summary.json')
        if m:
            rows.append({'method':'Multi-source DG + ResNet18','target':target,'frame_or_sequence_auc':m['frame_metrics']['auc'],'video_auc':m['video_metrics']['auc'],'video_eer':m['video_metrics']['eer']})

        m = safe_load(root / 'route2_dinov2_eval' / target / 'vfm_summary.json')
        if m:
            rows.append({'method':'DINOv2-Reg + Multi-source DG','target':target,'frame_or_sequence_auc':m['frame_metrics']['auc'],'video_auc':m['video_metrics']['auc'],'video_eer':m['video_metrics']['eer']})

        m = safe_load(root / 'route3_ssdg_dino' / target / 'target_summary.json')
        if m:
            rows.append({'method':'DINOv2-Reg + SSDG-style','target':target,'frame_or_sequence_auc':m['frame_metrics']['auc'],'video_auc':m['video_metrics']['auc'],'video_eer':m['video_metrics']['eer']})

        m = safe_load(root / 'route3_td_sf_eval' / target / 'summary.json')
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
        f.write('# 第三周三方向实验汇总\n\n')
        f.write('| 方法 | Target | Frame/Sequence AUC | Video AUC | Video EER |\n')
        f.write('|---|---|---:|---:|---:|\n')
        for r in rows:
            f.write(f"| {r['method']} | {r['target']} | {float(r['frame_or_sequence_auc']):.4%} | {float(r['video_auc']):.4%} | {float(r['video_eer']):.4%} |\n")

        f.write('\n## 说明\n\n')
        f.write('- 方向 1：Multi-source DG，使用 ResNet18 验证多源训练本身的收益。\n')
        f.write('- 方向 2：DINOv2-Reg + Multi-source DG，验证强骨干带来的收益。\n')
        f.write('- 方向 3：高级 FAS 方法，包含 SSDG-style 与 FAS-TD-SF-inspired 两个实现。\n')
        f.write('- FAS-TD-SF-inspired 使用 RGB 三帧短序列和空间梯度，不含原论文 PRNet 虚拟深度监督，因此不是原论文完整复现。\n')

    print(f'已生成: {csv_path}')
    print(f'已生成: {md_path}')


if __name__ == '__main__':
    main()
