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
    text = str(path).lower()
    if any(k.lower() in text for k in live_keywords):
        return 1
    if any(k.lower() in text for k in spoof_keywords):
        return 0
    return None


def infer_group(path: Path) -> str:
    """Infer a video/subject group by stripping the frame suffix from filename."""
    stem = path.stem
    stem = re.sub(r'([_-]?frame)?[_-]?\d{3,6}$', '', stem, flags=re.I)
    return f'{path.parent.name}/{stem}'


def scan_dataset(dataset_dir: str, live_keywords=DEFAULT_LIVE, spoof_keywords=DEFAULT_SPOOF):
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f'Dataset directory not found: {root}')
    samples = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            label = infer_label(p, live_keywords, spoof_keywords)
            if label is not None:
                samples.append((str(p), label, infer_group(p)))
    if not samples:
        raise RuntimeError(f'No labeled images found under {root}. Check folder names/keywords.')
    return samples


def split_by_group(samples, seed=42, train_ratio=0.7, val_ratio=0.15):
    idx = list(range(len(samples)))
    groups = [s[2] for s in samples]
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    tr_idx, temp_idx = next(gss1.split(idx, groups=groups))
    temp_groups = [groups[i] for i in temp_idx]
    val_fraction_of_temp = val_ratio / (1.0 - train_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_fraction_of_temp, random_state=seed + 1)
    va_rel, te_rel = next(gss2.split(temp_idx, groups=temp_groups))
    va_idx = [temp_idx[i] for i in va_rel]
    te_idx = [temp_idx[i] for i in te_rel]
    return ([samples[i] for i in tr_idx], [samples[i] for i in va_idx], [samples[i] for i in te_idx])


class FASImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int, str]], train=False, image_size=224):
        self.samples = samples
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
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
