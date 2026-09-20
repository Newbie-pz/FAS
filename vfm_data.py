from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from datasets import scan_dataset, strip_frame_suffix


MICO_DATASETS = ("OULU-NPU", "CASIA", "MSU-MFSD", "Replay-Attack")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def infer_video_id(path, dataset_name):
    """Recover the original video key from an extracted-frame file name."""
    return f"{dataset_name}:{strip_frame_suffix(Path(path))}"


def scan_vfm_domain(
    data_root,
    dataset_name,
    live_keywords=("live", "real", "genuine", "positive"),
    spoof_keywords=("spoof", "attack", "fake", "negative"),
):
    samples = scan_dataset(
        Path(data_root) / dataset_name,
        live_keywords=live_keywords,
        spoof_keywords=spoof_keywords,
    )
    return [
        (path, label, group, dataset_name, infer_video_id(path, dataset_name))
        for path, label, group in samples
    ]


def split_train_val_by_group(samples, val_ratio=0.10, seed=42):
    """Subject-disjoint train/validation split within one source domain."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1).")

    groups = [s[2] for s in samples]
    if len(set(groups)) < 2:
        raise RuntimeError("Need at least two groups for train/validation split.")

    idx = np.arange(len(samples))
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(idx, groups=groups))
    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]

    train_groups = {x[2] for x in train}
    val_groups = {x[2] for x in val}
    assert train_groups.isdisjoint(val_groups)
    return train, val


def evenly_subsample_frames_per_video(samples, max_frames_per_video):
    """Keep temporally distributed frames so long videos cannot dominate training."""
    if max_frames_per_video is None or max_frames_per_video <= 0:
        return list(samples)

    videos = defaultdict(list)
    for sample in samples:
        videos[sample[4]].append(sample)

    selected = []
    for video_id in sorted(videos):
        frames = sorted(videos[video_id], key=lambda x: x[0])
        if len(frames) <= max_frames_per_video:
            selected.extend(frames)
            continue
        indices = np.linspace(
            0,
            len(frames) - 1,
            num=max_frames_per_video,
            dtype=int,
        )
        selected.extend(frames[i] for i in indices)

    return selected


def build_domain_class_balanced_sampler(samples):
    """Equalize the sampling mass of every (source-domain, class) stratum."""
    counts = Counter((s[3], s[1]) for s in samples)
    weights = [1.0 / counts[(s[3], s[1])] for s in samples]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(samples),
        replacement=True,
    )


def build_vfm_transform(image_size=224, train=False):
    if not train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # Conservative FAS augmentation: diversify appearance while preserving spoof texture.
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.90, 1.00),
            ratio=(0.95, 1.05),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.15,
                hue=0.03,
            )
        ], p=0.60),
        transforms.RandomGrayscale(p=0.03),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
        ], p=0.08),
        transforms.RandomAdjustSharpness(sharpness_factor=0.8, p=0.08),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class VFMImageDataset(Dataset):
    def __init__(self, samples, train=False, image_size=224):
        self.samples = list(samples)
        self.transform = build_vfm_transform(image_size=image_size, train=train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, group, domain, video_id = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
            image = self.transform(img)
        return image, label, path, group, domain, video_id


def aggregate_video_scores(y_true, y_score, video_ids):
    """Mean-pool frame probabilities to one score per original video."""
    labels = {}
    scores = defaultdict(list)

    for label, score, video_id in zip(y_true, y_score, video_ids):
        label = int(label)
        if video_id in labels and labels[video_id] != label:
            raise RuntimeError(f"Inconsistent labels within video: {video_id}")
        labels[video_id] = label
        scores[video_id].append(float(score))

    ordered = sorted(scores)
    video_true = [labels[v] for v in ordered]
    video_score = [float(np.mean(scores[v])) for v in ordered]
    return ordered, video_true, video_score
