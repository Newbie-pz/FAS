import torch
import torch.nn as nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MixStyle(nn.Module):
    """Parameter-free feature-statistics mixing for domain generalization."""

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training or x.ndim != 4 or x.size(0) < 2:
            return x
        if torch.rand(1, device=x.device).item() > self.p:
            return x

        b = x.size(0)
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        sig = torch.sqrt(var + self.eps)
        x_norm = (x - mu) / sig

        beta = torch.distributions.Beta(self.alpha, self.alpha)
        lam = beta.sample((b, 1, 1, 1)).to(device=x.device, dtype=x.dtype)
        perm = torch.randperm(b, device=x.device)

        mu_mix = lam * mu + (1.0 - lam) * mu[perm]
        sig_mix = lam * sig + (1.0 - lam) * sig[perm]
        return x_norm * sig_mix + mu_mix


def fourier_amplitude_mix(
    x,
    p=0.5,
    max_lambda=0.35,
    low_freq_ratio=0.10,
):
    """
    Mix only the low-frequency amplitude spectrum between training samples.

    The original phase is retained, so the spatial/content structure mainly
    comes from the original sample while low-frequency appearance statistics
    are diversified. Input/output tensors follow ImageNet normalization.
    """
    if x.ndim != 4 or x.size(0) < 2:
        return x
    if torch.rand(1, device=x.device).item() > p:
        return x

    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    image = (x * std + mean).clamp(0.0, 1.0)

    spectrum = torch.fft.fft2(image, dim=(-2, -1))
    amplitude = torch.abs(spectrum)
    phase = torch.angle(spectrum)
    amplitude_shift = torch.fft.fftshift(amplitude, dim=(-2, -1))

    b, _, h, w = image.shape
    perm = torch.randperm(b, device=x.device)
    lam = torch.rand((b, 1, 1, 1), device=x.device, dtype=x.dtype) * max_lambda

    half_h = max(1, int(round(h * low_freq_ratio / 2.0)))
    half_w = max(1, int(round(w * low_freq_ratio / 2.0)))
    center_h, center_w = h // 2, w // 2
    h0, h1 = max(0, center_h - half_h), min(h, center_h + half_h + 1)
    w0, w1 = max(0, center_w - half_w), min(w, center_w + half_w + 1)

    mixed_shift = amplitude_shift.clone()
    source_patch = amplitude_shift[:, :, h0:h1, w0:w1]
    donor_patch = amplitude_shift[perm, :, h0:h1, w0:w1]
    mixed_shift[:, :, h0:h1, w0:w1] = (
        (1.0 - lam) * source_patch + lam * donor_patch
    )

    mixed_amplitude = torch.fft.ifftshift(mixed_shift, dim=(-2, -1))
    mixed_spectrum = torch.polar(mixed_amplitude, phase)
    mixed_image = torch.fft.ifft2(mixed_spectrum, dim=(-2, -1)).real.clamp(0.0, 1.0)

    return (mixed_image - mean) / std
