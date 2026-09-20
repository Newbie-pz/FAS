import math
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


class FASAugment:
    """Self-contained FAS-specific augmentation inspired by FAS-Aug.

    Four photography/noise operations preserve the label. Four print/display
    artifact operations convert an augmented live image into a synthetic spoof.
    No external texture packs or ICC profiles are required.
    """

    def __init__(self, p=0.75):
        self.p = float(p)
        self.ops = (
            ("color_diversity", False),
            ("low_resolution", False),
            ("hand_trembling", False),
            ("photography_noise", False),
            ("color_distortion", True),
            ("halftone", True),
            ("moire", True),
            ("reflection", True),
        )

    def __call__(self, img, label):
        if random.random() >= self.p:
            return img, int(label), "original"

        name, spoof_artifact = random.choice(self.ops)
        out = getattr(self, name)(img)
        new_label = 0 if (spoof_artifact and int(label) == 1) else int(label)
        return out, new_label, name

    @staticmethod
    def color_diversity(img):
        img = ImageEnhance.Color(img).enhance(random.uniform(0.55, 1.55))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.30))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.80, 1.20))
        return img

    @staticmethod
    def low_resolution(img):
        w, h = img.size
        ratio = random.uniform(0.20, 0.60)
        sw, sh = max(16, int(w * ratio)), max(16, int(h * ratio))
        small = img.resize((sw, sh), Image.Resampling.BILINEAR)
        interp = random.choice((Image.Resampling.NEAREST, Image.Resampling.BILINEAR))
        return small.resize((w, h), interp)

    @staticmethod
    def hand_trembling(img):
        arr = np.asarray(img, dtype=np.float32)
        length = random.randint(3, 11)
        axis = random.choice((0, 1, 2, 3))
        acc = np.zeros_like(arr)
        for k in range(length):
            offset = k - length // 2
            if axis == 0:
                shifted = np.roll(arr, offset, axis=1)
            elif axis == 1:
                shifted = np.roll(arr, offset, axis=0)
            elif axis == 2:
                shifted = np.roll(np.roll(arr, offset, axis=0), offset, axis=1)
            else:
                shifted = np.roll(np.roll(arr, offset, axis=0), -offset, axis=1)
            acc += shifted
        return Image.fromarray(np.clip(acc / length, 0, 255).astype(np.uint8))

    @staticmethod
    def photography_noise(img):
        if random.random() < 0.5:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))
        arr = np.asarray(img, dtype=np.float32)
        sigma = random.uniform(2.0, 8.0)
        arr += np.random.normal(0.0, sigma, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    @staticmethod
    def color_distortion(img):
        bits = random.choice((3, 4, 5))
        out = ImageOps.posterize(img, bits)
        out = ImageEnhance.Color(out).enhance(random.uniform(0.55, 1.45))
        out = ImageEnhance.Contrast(out).enhance(random.uniform(0.75, 1.35))
        return Image.blend(img, out, random.uniform(0.25, 0.55))

    @staticmethod
    def halftone(img):
        gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        h, w = gray.shape
        yy, xx = np.mgrid[0:h, 0:w]
        period = random.uniform(3.0, 7.0)
        pattern = 0.5 + 0.5 * np.sin(2 * math.pi * xx / period) * np.sin(
            2 * math.pi * yy / period
        )
        threshold = np.clip(gray * 0.75 + pattern * 0.25, 0.0, 1.0)
        dots = (threshold > random.uniform(0.42, 0.58)).astype(np.float32)
        texture = np.repeat((dots * 255.0)[..., None], 3, axis=2)
        tex = Image.fromarray(texture.astype(np.uint8))
        return Image.blend(img, tex, random.uniform(0.08, 0.22))

    @staticmethod
    def moire(img):
        arr = np.asarray(img, dtype=np.float32)
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        theta = random.uniform(0, math.pi)
        freq1 = random.uniform(0.06, 0.14)
        freq2 = freq1 * random.uniform(0.92, 1.08)
        u1 = xx * math.cos(theta) + yy * math.sin(theta)
        u2 = xx * math.cos(theta + random.uniform(0.03, 0.10)) + yy * math.sin(
            theta + random.uniform(0.03, 0.10)
        )
        pat = np.sin(2 * math.pi * freq1 * u1) + np.sin(2 * math.pi * freq2 * u2)
        amp = random.uniform(5.0, 16.0)
        arr = np.clip(arr + amp * pat[..., None], 0, 255)
        return Image.fromarray(arr.astype(np.uint8))

    @staticmethod
    def reflection(img):
        arr = np.asarray(img, dtype=np.float32)
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx = random.uniform(0.15, 0.85) * w
        cy = random.uniform(0.15, 0.85) * h
        sx = random.uniform(0.15, 0.40) * w
        sy = random.uniform(0.08, 0.30) * h
        glow = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2.0)
        strength = random.uniform(25.0, 70.0)
        arr = np.clip(arr + strength * glow[..., None], 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
