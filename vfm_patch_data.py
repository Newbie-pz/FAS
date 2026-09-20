from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from fas_aug import FASAugment

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_standard_transform(image_size=224, train=False):
    if not train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.92, 1.00),
            ratio=(0.96, 1.04),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            )
        ], p=0.40),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class VFMPatchDataset(Dataset):
    def __init__(
        self,
        samples,
        train=False,
        image_size=224,
        fas_aug_p=0.75,
    ):
        self.samples = list(samples)
        self.train = bool(train)
        self.transform = build_standard_transform(image_size, train=train)
        self.fas_aug = FASAugment(p=fas_aug_p) if train and fas_aug_p > 0 else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, group, domain, video_id = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
            aug_name = "original"
            out_label = int(label)
            if self.fas_aug is not None:
                img, out_label, aug_name = self.fas_aug(img, out_label)
            image = self.transform(img)

        return image, out_label, path, group, domain, video_id, aug_name
