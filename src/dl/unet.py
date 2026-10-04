"""
dl/unet.py
==========
A small three-level U-Net: 8 input channels in, 4 class logits per pixel out.

Why a U-Net and why this small
------------------------------
A U-Net predicts a class for every pixel rather than one label per image. The
encoder shrinks the image while learning features, the decoder grows it back,
and skip connections copy fine detail across so boundaries stay sharp - which is
the whole point when the quantity of interest is a shoreline.

It is trained from scratch rather than fine-tuned from ImageNet because the
input is 8 channels of 20 m surface reflectance and spectral indices, not a
3-channel photograph: the first convolution would have to be rebuilt anyway and
the learned filters would not match the physics. With only six lakes, a small
network is the point, not a compromise - roughly 0.5 M parameters against a
~0.5 M-pixel labelled set.

Two details worth keeping:

* `bias=False` on convolutions followed by BatchNorm - BatchNorm has its own
  shift, so the convolution's bias is redundant parameters.
* Upsampling is bilinear interpolation plus a 3x3 convolution rather than a
  transposed convolution, which avoids the checkerboard artefacts transposed
  convolutions leave on smooth surfaces like open water.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common as C


def _norm(ch: int, groups: int | None):
    """BatchNorm, or GroupNorm when batches are too small for stable statistics."""
    if groups:
        return nn.GroupNorm(min(groups, ch), ch)
    return nn.BatchNorm2d(ch)


class DoubleConv(nn.Module):
    """[conv3x3 - norm - ReLU] x 2, the standard U-Net block."""

    def __init__(self, in_ch: int, out_ch: int, groups=None, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch, groups), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch, groups), nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Bilinear upsample, concatenate the encoder skip, then DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, groups=None):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, groups)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        x = self.reduce(x)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(self, in_ch: int = C.N_CHANNELS, n_classes: int = C.N_CLASSES,
                 base: int = 16, groups: int | None = None, dropout: float = 0.3):
        super().__init__()
        b = base
        self.enc1 = DoubleConv(in_ch, b, groups)              # 128 -> b
        self.enc2 = DoubleConv(b, b * 2, groups)              #  64 -> 2b
        self.enc3 = DoubleConv(b * 2, b * 4, groups)          #  32 -> 4b
        self.bottleneck = DoubleConv(b * 4, b * 8, groups, dropout)   # 16 -> 8b
        self.pool = nn.MaxPool2d(2, 2)
        self.up3 = Up(b * 8, b * 4, b * 4, groups)
        self.up2 = Up(b * 4, b * 2, b * 2, groups)
        self.up1 = Up(b * 2, b, b, groups)
        self.head = nn.Conv2d(b, n_classes, 1)

    def forward(self, x):                   # x: (B, 8, 128, 128)
        s1 = self.enc1(x)                   # (B,  b, 128, 128)
        s2 = self.enc2(self.pool(s1))       # (B, 2b,  64,  64)
        s3 = self.enc3(self.pool(s2))       # (B, 4b,  32,  32)
        z = self.bottleneck(self.pool(s3))  # (B, 8b,  16,  16)
        x = self.up3(z, s3)                 # (B, 4b,  32,  32)
        x = self.up2(x, s2)                 # (B, 2b,  64,  64)
        x = self.up1(x, s1)                 # (B,  b, 128, 128)
        return self.head(x)                 # (B,  4, 128, 128)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TemperatureScaling(nn.Module):
    """
    One scalar dividing the logits, fitted on the inner-validation lake.

    Preserves the ranking (so predicted classes never change) while fixing
    over-confidence, which matters because these probabilities feed the
    per-lake-year composition numbers downstream. scikit-learn only gained
    native temperature scaling in 1.8; this project is on 1.6.1, and the model
    is PyTorch anyway.
    """

    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(self.log_t.detach().exp())

    def forward(self, logits):
        return logits / self.log_t.exp()

    def fit(self, logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 100):
        """logits (N, C), targets (N,) - already flattened and IGNORE-free."""
        opt = torch.optim.LBFGS([self.log_t], lr=0.05, max_iter=max_iter)
        loss_fn = nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = loss_fn(self(logits), targets)
            loss.backward()
            return loss

        opt.step(closure)
        return self
