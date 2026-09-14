import re
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEFAULT_LIVE = ('live', 'real', 'genuine', 'positive')
DEFAULT_SPOOF = ('spoof', 'attack', 'fake', 'negative')


def infer_label(path: Path, live_keywords=DEFAULT_LIVE, spoof_keywords=DEFAULT_SPOOF):
    """Return 1 for live and 0 for spoof based on path keywords."""
    text = str(path).lower()
    if any(k.lower() in text for k in live_keywords):
        return 1
    if any(k.lower() in text for k in spoof_keywords):
        return 0
    return None


def strip_frame_suffix(path: Path) -> str:
    """Remove the extracted-frame suffix while retaining the original video identifier."""
    return re.sub(r'([_-]?frame)?[_-]?\d{3,6}$', '', path.stem, flags=re.I)


def infer_subject(path: Path, dataset_name: str) -> str:
    """Infer a subject/client identity for subject-disjoint splitting."""
    stem = strip_frame_suffix(path)
    name = dataset_name.lower().replace('_', '-').strip()

    if name in {'oulu-npu', 'oulu'}:
        parts = stem.split('_')
        if len(parts) >= 3 and parts[2].isdigit():
            return f'oulu_subject_{parts[2]}'

    if name in {'msu-mfsd', 'msu'}:
        m = re.search(r'client[_-]?(\d+)', stem, flags=re.I)
        if m:
            return f'msu_client_{int(m.group(1)):03d}'

    if name in {'replay-attack', 'replayattack', 'replay'}:
        m = re.search(r'client[_-]?(\d+)', stem, flags=re.I)
        if m:
            return f'replay_client_{int(m.group(1)):03d}'

    if name in {'casia', 'casia-fasd', 'casia-fas'}:
        m = re.match(r'(train|test)_(\d+)(?:_|$)', stem, flags=re.I)
        if m:
            return f'casia_{m.group(1).lower()}_subject_{int(m.group(2)):02d}'

    return f'{dataset_name}_video_{stem}'


def scan_dataset(dataset_dir: str, live_keywords=DEFAULT_LIVE, spoof_keywords=DEFAULT_SPOOF):
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f'Dataset directory not found: {root}')

    dataset_name = root.name
    samples = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            label = infer_label(p, live_keywords, spoof_keywords)
            if label is not None:
                samples.append((str(p), label, infer_subject(p, dataset_name)))

    if not samples:
        raise RuntimeError(f'No labeled images found under {root}. Check folder names/keywords.')
    return samples


def split_by_group(samples, seed=42, train_ratio=0.7, val_ratio=0.15):
    """Subject-disjoint train/validation/test split using the sample group key."""
    idx = list(range(len(samples)))
    groups = [s[2] for s in samples]
    unique_groups = set(groups)
    if len(unique_groups) < 3:
        raise RuntimeError(
            f'Need at least 3 distinct subjects/groups for train/val/test, found {len(unique_groups)}.'
        )

    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    tr_idx, temp_idx = next(gss1.split(idx, groups=groups))

    temp_groups = [groups[i] for i in temp_idx]
    val_fraction_of_temp = val_ratio / (1.0 - train_ratio)
    gss2 = GroupShuffleSplit(
        n_splits=1,
        train_size=val_fraction_of_temp,
        random_state=seed + 1,
    )
    va_rel, te_rel = next(gss2.split(temp_idx, groups=temp_groups))
    va_idx = [temp_idx[i] for i in va_rel]
    te_idx = [temp_idx[i] for i in te_rel]

    train = [samples[i] for i in tr_idx]
    val = [samples[i] for i in va_idx]
    test = [samples[i] for i in te_idx]

    train_subjects = {s[2] for s in train}
    val_subjects = {s[2] for s in val}
    test_subjects = {s[2] for s in test}
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)

    return train, val, test


def build_train_transform(image_size: int, augmentation: str):
    """Build the training transform used by the baseline or week-3 generalization experiment."""
    if augmentation == 'baseline':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    if augmentation == 'strong':
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.80, 1.00),
                ratio=(0.90, 1.10),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.40,
                    contrast=0.40,
                    saturation=0.40,
                    hue=0.10,
                )
            ], p=0.80),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value='random',
            ),
        ])

    raise ValueError(f'Unsupported augmentation mode: {augmentation}')


class FASImageDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, int, str]],
        train=False,
        image_size=224,
        augmentation='baseline',
    ):
        self.samples = samples
        if train:
            self.transform = build_train_transform(image_size, augmentation)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, group = self.samples[index]
        with Image.open(path) as img:
            img = img.convert('RGB')
            img = self.transform(img)
        return img, label, path, group
