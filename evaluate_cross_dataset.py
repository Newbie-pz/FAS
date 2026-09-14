import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import FASImageDataset, scan_dataset
from metrics import save_plots_and_metrics
from model import build_resnet18


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_root', default='/root/Desktop/code/FAS/ProcessedData')
    p.add_argument('--dataset', required=True, help='目标测试数据集')
    p.add_argument('--source_dataset', default=None, help='训练该 checkpoint 的源数据集；默认从 checkpoint 中读取')
    p.add_argument('--output_dir', default='outputs_cross')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--live_keywords', nargs='+', default=['live', 'real', 'genuine', 'positive'])
    p.add_argument('--spoof_keywords', nargs='+', default=['spoof', 'attack', 'fake', 'negative'])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get('args', {})
    source_dataset = args.source_dataset or ckpt_args.get('dataset', 'unknown_source')

    if source_dataset == args.dataset:
        print(f'[警告] source_dataset 与 target_dataset 相同：{source_dataset}。这不是跨数据集实验。')

    samples = scan_dataset(Path(args.data_root) / args.dataset, args.live_keywords, args.spoof_keywords)
    ds = FASImageDataset(samples, train=False, image_size=args.image_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_resnet18(pretrained=False).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    y_true, y_score = [], []
    with torch.no_grad():
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(y.cpu().numpy().tolist())
            y_score.extend(probs.cpu().numpy().tolist())

    pair_name = f'{source_dataset}_to_{args.dataset}'
    out = Path(args.output_dir) / pair_name
    metrics = save_plots_and_metrics(y_true, y_score, out, prefix='cross_dataset')
    metrics['loss'] = total_loss / max(len(ds), 1)
    metrics['source_dataset'] = source_dataset
    metrics['target_dataset'] = args.dataset
    metrics['num_images'] = len(ds)
    metrics['num_live'] = int(sum(y_true))
    metrics['num_spoof'] = int(len(y_true) - sum(y_true))
    metrics['checkpoint'] = str(args.checkpoint)
    metrics['protocol'] = 'source-only training; target-only testing; no target fine-tuning'

    with open(out / 'cross_dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print('\n===== 跨数据集测试结果 =====')
    print(f'源数据集      : {source_dataset}')
    print(f'目标数据集    : {args.dataset}')
    print(f'测试图像数量  : {len(ds)}')
    print(f'Accuracy      : {metrics["accuracy"]:.6f}')
    print(f'AUC           : {metrics["auc"]:.6f}')
    print(f'EER           : {metrics["eer"]:.6f}')
    print(f'EER threshold : {metrics["eer_threshold"]:.6f}')
    print(f'输出目录      : {out}')


if __name__ == '__main__':
    main()
