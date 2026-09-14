import argparse
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
    p.add_argument('--dataset', required=True)
    p.add_argument('--output_dir', default='outputs_cross')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--live_keywords', nargs='+', default=['live','real','genuine','positive'])
    p.add_argument('--spoof_keywords', nargs='+', default=['spoof','attack','fake','negative'])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samples = scan_dataset(Path(args.data_root) / args.dataset, args.live_keywords, args.spoof_keywords)
    ds = FASImageDataset(samples, train=False, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_resnet18(pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
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

    out = Path(args.output_dir) / args.dataset
    metrics = save_plots_and_metrics(y_true, y_score, out, prefix='cross_dataset')
    metrics['loss'] = total_loss / max(len(ds), 1)
    print(metrics)


if __name__ == '__main__':
    main()
