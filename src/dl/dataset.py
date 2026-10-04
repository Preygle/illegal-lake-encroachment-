"""
dl/dataset.py
=============
Patch extraction, fold-local normalisation, and augmentation.

Patches are 128 x 128 at 20 m (2.56 km across), cut with stride 64 from the
bounding box of each lake's footprint-plus-collar and kept when at least 10% of
their pixels carry a label.

Two properties are load-bearing:

* **A patch belongs to exactly one lake.** Labels are masked to that lake's own
  region before the patch is cut, so a patch taken near Bellandur can never
  carry Varthur's pixels. Without this, grouping the folds by lake would not
  actually separate them.

* **Normalisation statistics and class weights are fitted on the training fold
  only.** Fitting them across all six lakes first is the pre-split
  preprocessing leak - the held-out lake would have contributed to the numbers
  used to train on it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import common as C

# 64 px at 20 m is 1.28 km across. The design proposed 128, but the measured
# geometry rules it out: four of the six lakes have a footprint-plus-collar
# bounding box smaller than 128 px (Sankey Tank is 76 x 74), so a 128 px patch
# would be mostly padding for them and each would yield a single patch position
# per year - 7 patches for a whole lake. At 64/32 the index goes from 98 patches
# to roughly 500, and 1.28 km still exceeds this network's receptive field, so
# no context is actually lost.
PATCH = 64
STRIDE = 32
MIN_LABELLED_FRAC = 0.10


# ==========================================================================
# Patch index
# ==========================================================================
def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _starts(lo: int, hi: int, limit: int) -> list[int]:
    """Patch origins covering [lo, hi), clamped inside the grid."""
    span = hi - lo
    if span <= PATCH:
        return [max(0, min(limit - PATCH, lo + (span - PATCH) // 2))]
    out = list(range(lo, hi - PATCH + 1, STRIDE))
    if out[-1] != hi - PATCH:
        out.append(hi - PATCH)
    return [max(0, min(limit - PATCH, s)) for s in out]


def build_index(rasters=None, verbose: bool = True) -> list[dict]:
    """
    Every (lake, year, row, col) patch worth training on.

    Returned records are light - the arrays themselves are loaded once by
    PatchStore and sliced on demand.
    """
    _, shape, _, _ = C.grid()
    rasters = rasters or C.lake_rasters()
    labels = {y: np.load(C.DL / "labels" / f"labels_{y}.npy") for y in C.YEARS}

    index = []
    for lake, r in rasters.items():
        region = r["region"]
        r0, r1, c0, c1 = _bbox(region)
        for year in C.YEARS:
            lab = np.where(region, labels[year], C.IGNORE)
            for rs in _starts(r0, r1, shape[0]):
                for cs in _starts(c0, c1, shape[1]):
                    tile = lab[rs:rs + PATCH, cs:cs + PATCH]
                    frac = float((tile != C.IGNORE).mean())
                    if frac >= MIN_LABELLED_FRAC:
                        index.append({"lake": lake, "year": year,
                                      "row": rs, "col": cs, "labelled": frac})
    if verbose:
        from collections import Counter

        per_lake = Counter(p["lake"] for p in index)
        print(f"  {len(index)} patches over {len(per_lake)} lakes")
        for lake in C.lake_order():
            print(f"    {lake:16s} {per_lake[lake]:4d}")
    return index


# ==========================================================================
# In-memory store
# ==========================================================================
class PatchStore:
    """Holds every channel stack and label raster once, and cuts patches from them."""

    def __init__(self, rasters=None):
        self.rasters = rasters or C.lake_rasters()
        self.chan, self.bad = {}, {}
        for y in C.YEARS:
            arr, bad = C.channels(y)
            self.chan[y] = arr
            self.bad[y] = bad
        self.labels = {}
        for y in C.YEARS:
            lab = np.load(C.DL / "labels" / f"labels_{y}.npy").copy()
            lab[self.bad[y]] = C.IGNORE
            self.labels[y] = lab
        self._masked: dict[tuple[str, int], np.ndarray] = {}

    def label_for(self, lake: str, year: int) -> np.ndarray:
        key = (lake, year)
        if key not in self._masked:
            self._masked[key] = np.where(self.rasters[lake]["region"],
                                         self.labels[year], C.IGNORE)
        return self._masked[key]

    def cut(self, rec: dict):
        r, c, y = rec["row"], rec["col"], rec["year"]
        x = self.chan[y][:, r:r + PATCH, c:c + PATCH]
        m = self.label_for(rec["lake"], y)[r:r + PATCH, c:c + PATCH]
        return x, m


# ==========================================================================
# Fold-local statistics
# ==========================================================================
def fit_norm(store: PatchStore, index: list[dict], lakes: list[str]):
    """Per-channel mean/std over the training lakes' labelled pixels only."""
    recs = [p for p in index if p["lake"] in lakes]
    tot = np.zeros(C.N_CHANNELS, "float64")
    tot2 = np.zeros(C.N_CHANNELS, "float64")
    n = 0
    for rec in recs:
        x, m = store.cut(rec)
        valid = m != C.IGNORE
        if not valid.any():
            continue
        v = x[:, valid]
        tot += v.sum(axis=1)
        tot2 += (v ** 2).sum(axis=1)
        n += v.shape[1]
    mean = tot / max(n, 1)
    var = np.maximum(tot2 / max(n, 1) - mean ** 2, 1e-8)
    return mean.astype("float32"), np.sqrt(var).astype("float32")


def class_counts(store: PatchStore, index: list[dict], lakes: list[str]):
    """Label histogram over the training lakes, for the loss weights."""
    counts = np.zeros(C.N_CLASSES, "int64")
    for rec in (p for p in index if p["lake"] in lakes):
        _, m = store.cut(rec)
        for c in range(C.N_CLASSES):
            counts[c] += int((m == c).sum())
    return counts


def class_weights(counts: np.ndarray) -> np.ndarray:
    """Inverse frequency, normalised to mean 1 so the loss scale stays familiar."""
    w = 1.0 / np.maximum(counts, 1)
    return (w / w.mean()).astype("float32")


# ==========================================================================
# Dataset
# ==========================================================================
def dihedral(x: torch.Tensor, m: torch.Tensor, k: int):
    """
    One of the 8 symmetries of a square, applied identically to image and mask.

    All 8 are valid here because nadir satellite imagery has no natural "up" -
    unlike photographs, where a vertical flip produces an impossible scene. This
    is the only augmentation used: ColorJitter, resizing, MixUp and CutMix would
    all corrupt the reflectance values the spectral indices are computed from.
    """
    if k >= 4:
        x, m = torch.flip(x, dims=[-1]), torch.flip(m, dims=[-1])
    r = k % 4
    if r:
        x, m = torch.rot90(x, r, dims=[-2, -1]), torch.rot90(m, r, dims=[-2, -1])
    return x.contiguous(), m.contiguous()


class LakePatches(Dataset):
    def __init__(self, store: PatchStore, index: list[dict], lakes: list[str],
                 mean, std, augment: bool = False, shuffle_labels: bool = False,
                 seed: int = 0):
        self.store = store
        self.recs = [p for p in index if p["lake"] in lakes]
        self.mean = torch.from_numpy(np.asarray(mean).reshape(-1, 1, 1))
        self.std = torch.from_numpy(np.asarray(std).reshape(-1, 1, 1))
        self.augment = augment
        self.shuffle_labels = shuffle_labels
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        x, m = self.store.cut(self.recs[i])
        x = (torch.from_numpy(np.ascontiguousarray(x)) - self.mean) / self.std
        m = torch.from_numpy(np.ascontiguousarray(m)).long()

        if self.shuffle_labels:
            # Negative control: permute the labelled pixels within the patch.
            # A model trained on this MUST score at chance. If it does not, the
            # pipeline is leaking and no other number from it can be trusted.
            valid = m != C.IGNORE
            vals = m[valid]
            m = m.clone()
            m[valid] = vals[torch.from_numpy(
                self._rng.permutation(len(vals)))]

        if self.augment:
            x, m = dihedral(x, m, int(self._rng.integers(8)))
        return x, m
