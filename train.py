import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import FASImageDataset, scan_dataset, split_by_group
from dg_methods import fourier_amplitude_mix
from metrics import save_plots_and_metrics
from model import build_resnet18


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='/root/Desktop/code/FAS/ProcessedData')
    p.add_argument('--dataset', default='OULU-NPU')
    p.add_argument('--output_dir', default='outputs')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--augmentation', choices=['baseline', 'strong', 'appearance'], default='baseline')
    p.add_argument('--method', choices=['baseline', 'mixstyle', 'fourier'], default='baseline')
    p.add_argument('--mixstyle_p', type=float, default=0.5)
    p.add_argument('--mixstyle_alpha', type=float, default=0.1)
    p.add_argument('--fourier_p', type=float, default=0.5)
    p.add_argument('--fourier_max_lambda', type=float, default=0.35)
    p.add_argument('--fourier_low_freq_ratio', type=float, default=0.10)
    p.add_argument('--no_pretrained', action='store_true')
    p.add_argument('--live_keywords', nargs='+', default=['live','real','genuine','positive'])
    p.add_argument('--spoof_keywords', nargs='+', default=['spoof','attack','fake','negative'])
    return p.parse_args()


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train=True,
    method='baseline',
    fourier_p=0.5,
    fourier_max_lambda=0.35,
    fourier_low_freq_ratio=0.10,
):
    model.train(train)
    total_loss, ys, probs = 0.0, [], []
    for x, y, _, _ in loader:
        x, y = x.to(device), y.to(device)

        if train and method == 'fourier':
            x = fourier_amplitude_mix(
                x,
                labels=y,
                p=fourier_p,
                max_lambda=fourier_max_lambda,
                low_freq_ratio=fourier_low_freq_ratio,
            )

        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        prob_live = torch.softmax(logits, dim=1)[:, 1]
        ys.extend(y.detach().cpu().numpy().tolist())
        probs.extend(prob_live.detach().cpu().numpy().tolist())
    acc = float(np.mean((np.asarray(probs) >= 0.5) == np.asarray(ys)))
    return total_loss / max(len(loader.dataset), 1), acc, ys, probs


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_dir = Path(args.data_root) / args.dataset
    samples = scan_dataset(dataset_dir, args.live_keywords, args.spoof_keywords)
    train_s, val_s, test_s = split_by_group(samples, seed=args.seed)

    out = Path(args.output_dir) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'split_summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': args.dataset,
            'augmentation': args.augmentation,
            'method': args.method,
            'total_images': len(samples),
            'train_images': len(train_s), 'val_images': len(val_s), 'test_images': len(test_s),
            'train_groups': len(set(x[2] for x in train_s)),
            'val_groups': len(set(x[2] for x in val_s)),
            'test_groups': len(set(x[2] for x in test_s)),
        }, f, ensure_ascii=False, indent=2)

    train_ds = FASImageDataset(
        train_s,
        train=True,
        image_size=args.image_size,
        augmentation=args.augmentation,
    )
    val_ds = FASImageDataset(train=False, samples=val_s, image_size=args.image_size)
    test_ds = FASImageDataset(train=False, samples=test_s, image_size=args.image_size)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    model = build_resnet18(
        pretrained=not args.no_pretrained,
        mixstyle=(args.method == 'mixstyle'),
        mixstyle_p=args.mixstyle_p,
        mixstyle_alpha=args.mixstyle_alpha,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float('inf')
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            True,
            method=args.method,
            fourier_p=args.fourier_p,
            fourier_max_lambda=args.fourier_max_lambda,
            fourier_low_freq_ratio=args.fourier_low_freq_ratio,
        )
        with torch.no_grad():
            va_loss, va_acc, _, _ = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer,
                device,
                False,
                method=args.method,
                fourier_p=args.fourier_p,
                fourier_max_lambda=args.fourier_max_lambda,
                fourier_low_freq_ratio=args.fourier_low_freq_ratio,
            )
        row = {'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': va_loss, 'val_acc': va_acc}
        history.append(row)
        print(row)
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            bad_epochs = 0
            torch.save({'model': model.state_dict(), 'args': vars(args)}, out / 'best.pth')
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break

    with open(out / 'history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    ckpt = torch.load(out / 'best.pth', map_location=device)
    model.load_state_dict(ckpt['model'])
    with torch.no_grad():
        _, _, y_true, y_score = run_epoch(
            model,
            test_loader,
            criterion,
            optimizer,
            device,
            False,
            method=args.method,
            fourier_p=args.fourier_p,
            fourier_max_lambda=args.fourier_max_lambda,
            fourier_low_freq_ratio=args.fourier_low_freq_ratio,
        )
    metrics = save_plots_and_metrics(y_true, y_score, out, prefix='test')
    metrics['augmentation'] = args.augmentation
    metrics['method'] = args.method
    print('Test metrics:', metrics)


if __name__ == '__main__':
    main()
