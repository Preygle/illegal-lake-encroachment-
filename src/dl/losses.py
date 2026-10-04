"""
dl/losses.py
============
Weighted cross-entropy plus soft Dice, both ignoring unlabelled pixels.

Class balance here is severe and lake-dependent: Bellandur's footprint is ~5%
open water against ~83% vegetation in 2019, while Madiwala's collar is mostly
built-up. Inverse-frequency class weights are the first line of defence -
weighting the loss is preferable to resampling, which would either duplicate
patches or throw them away, and SMOTE is meaningless on spatially structured
pixels.

Dice is added because cross-entropy optimises per-pixel likelihood and is
relatively indifferent to a thin class that is spatially coherent. Dice scores
overlap directly, so it pushes on the boundary, which is exactly what a
shoreline is.

If a class is still being swallowed after weighting, `FocalLoss` here is the
documented next step (gamma=2) - but weights first.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common as C


class SoftDiceLoss(nn.Module):
    """
    1 - mean Dice over classes present in the batch.

    Classes absent from a batch are skipped rather than scored 0, so a patch
    that happens to contain no water is not punished for the model correctly
    predicting none.
    """

    def __init__(self, ignore_index: int = C.IGNORE, eps: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, target):
        n_cls = logits.shape[1]
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0

        probs = F.softmax(logits, dim=1)
        tgt = target.clone()
        tgt[~valid] = 0
        onehot = F.one_hot(tgt, n_cls).permute(0, 3, 1, 2).float()

        v = valid.unsqueeze(1).float()
        probs, onehot = probs * v, onehot * v

        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        present = onehot.sum(dims) > 0
        if not present.any():
            return logits.sum() * 0.0
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice[present].mean()


class FocalLoss(nn.Module):
    """Down-weights easy pixels. Fallback for a class that weighting alone misses."""

    def __init__(self, weight=None, gamma: float = 2.0,
                 ignore_index: int = C.IGNORE):
        super().__init__()
        self.register_buffer("weight", weight if weight is None
                             else torch.as_tensor(weight, dtype=torch.float32))
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             ignore_index=self.ignore_index, reduction="none")
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce)[valid].mean()


class SegLoss(nn.Module):
    """weighted CE (or focal) + dice_weight * Dice."""

    def __init__(self, class_weight=None, dice_weight: float = 1.0,
                 focal: bool = False, gamma: float = 2.0,
                 ignore_index: int = C.IGNORE):
        super().__init__()
        w = None if class_weight is None else torch.as_tensor(
            class_weight, dtype=torch.float32)
        if focal:
            self.pixel = FocalLoss(w, gamma, ignore_index)
        else:
            self.pixel = nn.CrossEntropyLoss(weight=w,
                                             ignore_index=ignore_index)
        self.dice = SoftDiceLoss(ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return self.pixel(logits, target) + self.dice_weight * self.dice(logits, target)
