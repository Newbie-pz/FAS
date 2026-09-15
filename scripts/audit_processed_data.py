#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import IMG_EXTS, infer_label, infer_subject


DATASET_ORDER = ['OULU-NPU', 'CASIA', 'MSU-MFSD', 'Replay-Attack']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Read-only quality audit for processed FAS image datasets.'
    )
    parser.add_argument(
        '--data_root',
        default='/root/Desktop/code/FAS/ProcessedData',
        help='Root directory containing processed dataset folders.',
    )
    parser.add_argument(
        '--output_dir',
        default='outputs_data_audit',
        help='Directory for JSON/Markdown audit reports.',
    )
    parser.add_argument(
        '--min_side',
        type=int,
        default=64,
        help='Flag images whose shorter side is below this value.',
    )
    return parser.parse_args()


def audit_dataset(dataset_dir: Path, min_side: int):
    dataset_name = dataset_dir.name
    image_paths = sorted(
        p for p in dataset_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

    label_counts = Counter()
    group_counts = Counter()
    resolution_counts = Counter()
    unreadable = []
    unlabeled = []
    small_images = []
    non_rgb_convertible = []
    frame_named = 0
    widths = []
    heights = []

    for path in image_paths:
        label = infer_label(path)
        if label is None:
            unlabeled.append(str(path))
        else:
            label_counts['live' if label == 1 else 'spoof'] += 1

        group = infer_subject(path, dataset_name)
        group_counts[group] += 1

        if 'frame' in path.stem.lower():
            frame_named += 1

        try:
            # verify() should run on an image opened specifically for validation.
            with Image.open(path) as img:
                img.verify()

            # Re-open the file for metadata and RGB conversion after verify().
            with Image.open(path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
                resolution_counts[f'{width}x{height}'] += 1
                if min(width, height) < min_side:
                    small_images.append(str(path))
                try:
                    img.convert('RGB')
                except Exception:
                    non_rgb_convertible.append(str(path))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            unreadable.append({'path': str(path), 'error': str(exc)})

    total = len(image_paths)
    valid_sizes = len(widths)
    top_resolutions = [
        {'resolution': res, 'count': count}
        for res, count in resolution_counts.most_common(10)
    ]

    return {
        'dataset': dataset_name,
        'dataset_dir': str(dataset_dir),
        'total_images': total,
        'live_images': label_counts['live'],
        'spoof_images': label_counts['spoof'],
        'unlabeled_images': len(unlabeled),
        'unreadable_images': len(unreadable),
        'non_rgb_convertible_images': len(non_rgb_convertible),
        'small_images': len(small_images),
        'frame_named_images': frame_named,
        'frame_named_ratio': (frame_named / total) if total else 0.0,
        'group_count': len(group_counts),
        'subject_disjoint_split_possible': len(group_counts) >= 3,
        'min_width': min(widths) if valid_sizes else None,
        'max_width': max(widths) if valid_sizes else None,
        'min_height': min(heights) if valid_sizes else None,
        'max_height': max(heights) if valid_sizes else None,
        'top_resolutions': top_resolutions,
        'unlabeled_examples': unlabeled[:20],
        'small_image_examples': small_images[:20],
        'non_rgb_convertible_examples': non_rgb_convertible[:20],
        'unreadable_examples': unreadable[:20],
    }


def write_markdown(report, output_path: Path):
    lines = [
        '# ProcessedData 数据质量审计',
        '',
        '本报告由 `scripts/audit_processed_data.py` 自动生成。脚本仅执行只读检查，不修改数据。',
        '',
        '## 一、总体结果',
        '',
        '| 数据集 | 图像数 | Live | Spoof | 无标签 | 无法读取 | 异常小图 | Group 数 | frame 命名比例 | 可进行 Subject-disjoint |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]

    for item in report['datasets']:
        lines.append(
            f"| {item['dataset']} | {item['total_images']} | {item['live_images']} | "
            f"{item['spoof_images']} | {item['unlabeled_images']} | "
            f"{item['unreadable_images']} | {item['small_images']} | "
            f"{item['group_count']} | {item['frame_named_ratio']:.2%} | "
            f"{'是' if item['subject_disjoint_split_possible'] else '否'} |"
        )

    lines.extend([
        '',
        '## 二、逐数据集尺寸与常见分辨率',
        '',
    ])

    for item in report['datasets']:
        lines.extend([
            f"### {item['dataset']}",
            '',
            f"- 宽度范围：`{item['min_width']}` ～ `{item['max_width']}`",
            f"- 高度范围：`{item['min_height']}` ～ `{item['max_height']}`",
            f"- 无法读取：`{item['unreadable_images']}`",
            f"- 无法推断标签：`{item['unlabeled_images']}`",
            f"- 短边小于阈值的图像：`{item['small_images']}`",
            '',
            '| 常见分辨率 | 数量 |',
            '|---|---:|',
        ])
        for res in item['top_resolutions']:
            lines.append(f"| {res['resolution']} | {res['count']} |")
        if not item['top_resolutions']:
            lines.append('| - | 0 |')
        lines.append('')

    lines.extend([
        '## 三、异常样本',
        '',
        '以下仅列出每类异常的前 20 个样本，完整统计见 JSON。',
        '',
    ])

    for item in report['datasets']:
        lines.append(f"### {item['dataset']}")
        lines.append('')
        for title, key in [
            ('无法读取', 'unreadable_examples'),
            ('无法推断标签', 'unlabeled_examples'),
            ('异常小图', 'small_image_examples'),
            ('无法转换 RGB', 'non_rgb_convertible_examples'),
        ]:
            lines.append(f'**{title}**')
            values = item[key]
            if not values:
                lines.append('- 无')
            else:
                for value in values:
                    if isinstance(value, dict):
                        lines.append(f"- `{value['path']}`：{value['error']}")
                    else:
                        lines.append(f'- `{value}`')
            lines.append('')

    lines.extend([
        '## 四、解释边界',
        '',
        '- `frame` 文件名比例只能证明当前目录中存在明显的抽帧命名痕迹，不能反推出原始抽帧 FPS 或时间间隔。',
        '- 本审计只能检查最终图像文件的可读性、标签、尺寸和分组，不能反推出历史 Face Detector 型号或 bbox 扩展比例。',
        '- Subject/Group 解析规则来自当前 `datasets.py`；当前实验采用自定义 Subject-disjoint 70/15/15，而不是数据集官方 Protocol。',
        '- 若出现无法读取、无法推断标签或极端尺寸样本，应在重新训练前人工确认。',
    ])

    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f'Data root not found: {data_root}')

    datasets = []
    for name in DATASET_ORDER:
        dataset_dir = data_root / name
        if not dataset_dir.exists():
            datasets.append({
                'dataset': name,
                'dataset_dir': str(dataset_dir),
                'missing': True,
                'total_images': 0,
                'live_images': 0,
                'spoof_images': 0,
                'unlabeled_images': 0,
                'unreadable_images': 0,
                'non_rgb_convertible_images': 0,
                'small_images': 0,
                'frame_named_images': 0,
                'frame_named_ratio': 0.0,
                'group_count': 0,
                'subject_disjoint_split_possible': False,
                'min_width': None,
                'max_width': None,
                'min_height': None,
                'max_height': None,
                'top_resolutions': [],
                'unlabeled_examples': [],
                'small_image_examples': [],
                'non_rgb_convertible_examples': [],
                'unreadable_examples': [],
            })
            continue
        datasets.append(audit_dataset(dataset_dir, args.min_side))

    report = {
        'data_root': str(data_root),
        'min_side_threshold': args.min_side,
        'datasets': datasets,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'processed_data_audit.json'
    md_path = output_dir / 'processed_data_audit.md'

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    write_markdown(report, md_path)

    print(f'已生成: {json_path}')
    print(f'已生成: {md_path}')

    total_images = sum(x['total_images'] for x in datasets)
    total_unreadable = sum(x['unreadable_images'] for x in datasets)
    total_unlabeled = sum(x['unlabeled_images'] for x in datasets)
    print(f'总图像数: {total_images}')
    print(f'无法读取: {total_unreadable}')
    print(f'无法推断标签: {total_unlabeled}')


if __name__ == '__main__':
    main()
