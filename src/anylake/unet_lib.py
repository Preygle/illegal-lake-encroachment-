"""
anylake/unet_lib.py
===================
The project's U-Net (dl/unet.py), trained and run on lakes that each have
their own grid, with 16 input channels instead of 8.

Everything that made the original training honest is kept:

  * a patch belongs to one lake (labels masked to that lake's region first);
  * normalisation statistics and class weights come from the training lakes
    only;
  * early stopping uses a separate validation lake, never the test lake;
  * dihedral flips/rotations are the only augmentation;
  * temperature scaling is fitted on the validation lake.

Raw and enhanced runs use the same seeds, folds, patches and settings, so the
only difference between the two models is the input.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import LAKES, YEARS
from .lake import IGNORE, N_CHANNELS, Lake

sys.path.insert(0, str(LAKES.parents[1] / "src"))
from dl import common as C  # noqa: E402
from dl import metrics as M  # noqa: E402
from dl.dataset import MIN_LABELLED_FRAC, PATCH, _bbox, _starts, class_weights, dihedral  # noqa: E402
from dl.losses import SegLoss  # noqa: E402
from dl.unet import TemperatureScaling, UNet  # noqa: E402

DEFAULTS = dict(epochs=100, batch_size=8, lr=1e-3, weight_decay=1e-4, patience=10,
                base=16, dice_weight=1.0, seed=42)


# ==========================================================================
# Store and index
# ==========================================================================
class LibraryStore:
    """Channels, labels and regions for several lakes, one variant (raw | enh)."""

    def __init__(self, slugs: list[str], variant: str, log=print):
        self.variant = variant
        self.lakes = {s: Lake(s) for s in slugs}
        self.chan, self.labels, self.region = {}, {}, {}
        t0 = time.time()
        for s, L in self.lakes.items():
            self.region[s] = L.masks["region"]
            for y in L.years:
                x, bad = L.channels(y, variant)
                lab = L.labels(y).copy()
                lab[bad] = IGNORE
                self.chan[(s, y)] = x
                self.labels[(s, y)] = lab
        log(f"  loaded {len(self.chan)} lake-years ({variant}) in {time.time() - t0:.0f}s")

    def cut(self, rec):
        k = (rec["lake"], rec["year"])
        r, c = rec["row"], rec["col"]
        return (self.chan[k][:, r:r + PATCH, c:c + PATCH],
                self.labels[k][r:r + PATCH, c:c + PATCH])


def build_index(store: LibraryStore) -> list[dict]:
    index = []
    for s, L in store.lakes.items():
        r0, r1, c0, c1 = _bbox(store.region[s])
        for y in L.years:
            lab = store.labels[(s, y)]
            for rs in _starts(r0, r1, L.shape[0]):
                for cs in _starts(c0, c1, L.shape[1]):
                    tile = lab[rs:rs + PATCH, cs:cs + PATCH]
                    if tile.shape == (PATCH, PATCH) and (tile != IGNORE).mean() >= MIN_LABELLED_FRAC:
                        index.append({"lake": s, "year": y, "row": rs, "col": cs})
    return index


def fit_norm(store, index, lakes):
    tot = np.zeros(N_CHANNELS, "float64")
    tot2 = np.zeros(N_CHANNELS, "float64")
    n = 0
    for rec in (p for p in index if p["lake"] in lakes):
        x, m = store.cut(rec)
        v = x[:, m != IGNORE]
        tot += v.sum(1)
        tot2 += (v.astype("float64") ** 2).sum(1)
        n += v.shape[1]
    mean = tot / max(n, 1)
    std = np.sqrt(np.maximum(tot2 / max(n, 1) - mean ** 2, 1e-8))
    return mean.astype("float32"), std.astype("float32")


def class_counts(store, index, lakes):
    counts = np.zeros(4, "int64")
    for rec in (p for p in index if p["lake"] in lakes):
        _, m = store.cut(rec)
        counts += np.bincount(m[m != IGNORE].ravel(), minlength=4)[:4]
    return counts


class Patches(Dataset):
    def __init__(self, store, index, lakes, mean, std, augment, seed):
        self.store = store
        self.recs = [p for p in index if p["lake"] in lakes]
        self.mean = torch.from_numpy(mean.reshape(-1, 1, 1))
        self.std = torch.from_numpy(std.reshape(-1, 1, 1))
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        x, m = self.store.cut(self.recs[i])
        x = (torch.from_numpy(np.ascontiguousarray(x)) - self.mean) / self.std
        m = torch.from_numpy(np.ascontiguousarray(m)).long()
        if self.augment:
            x, m = dihedral(x, m, int(self.rng.integers(8)))
        return x, m


# ==========================================================================
# Train / predict
# ==========================================================================
def train(store, index, train_lakes, val_lake, cfg=None, seed_offset=0, log=print):
    """Train one U-Net. Returns a bundle: weights + statistics + temperature."""
    cfg = {**DEFAULTS, **(cfg or {})}
    C.set_seed(cfg["seed"] + seed_offset)
    mean, std = fit_norm(store, index, train_lakes)
    weights = class_weights(class_counts(store, index, train_lakes))
    g = torch.Generator()
    g.manual_seed(cfg["seed"] + seed_offset)
    tr = Patches(store, index, train_lakes, mean, std, True, cfg["seed"])
    va = Patches(store, index, [val_lake], mean, std, False, cfg["seed"] + 1)
    tr_dl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True, generator=g,
                       drop_last=len(tr) > cfg["batch_size"], num_workers=0)
    va_dl = DataLoader(va, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    model = UNet(in_ch=N_CHANNELS, base=cfg["base"])
    crit = SegLoss(class_weight=weights, dice_weight=cfg["dice_weight"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    log(f"    {len(tr)} train / {len(va)} val patches, {model.n_params:,d} params")

    best, bad, best_epoch, best_state = float("inf"), 0, -1, None
    for epoch in range(cfg["epochs"]):
        model.train()
        for x, m in tr_dl:
            opt.zero_grad()
            loss = crit(model(x), m)
            loss.backward()
            opt.step()
        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for x, m in va_dl:
                vtot += float(crit(model(x), m)) * len(x)
        vloss = vtot / max(len(va), 1)
        sched.step(vloss)
        if vloss < best - 1e-4:
            best, bad, best_epoch = vloss, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    log(f"    stopped at epoch {epoch} (best {best_epoch}, val loss {best:.4f})")

    model.eval()
    L, T = [], []
    with torch.no_grad():
        for x, m in va_dl:
            lg = model(x).permute(0, 2, 3, 1).reshape(-1, 4)
            t = m.reshape(-1)
            keep = t != IGNORE
            L.append(lg[keep])
            T.append(t[keep])
    temp = 1.0
    if L and sum(len(t) for t in T) > 100:
        ts = TemperatureScaling()
        ts.fit(torch.cat(L).float(), torch.cat(T).long())
        temp = ts.temperature
    return {"state": best_state, "mean": mean, "std": std, "temperature": temp,
            "base": cfg["base"], "best_epoch": best_epoch, "val_lake": val_lake,
            "train_lakes": list(train_lakes), "variant": store.variant}


def load_model(bundle):
    model = UNet(in_ch=N_CHANNELS, base=bundle["base"])
    model.load_state_dict(bundle["state"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, bundle, chan: np.ndarray, region: np.ndarray):
    """Overlap-tiled inference over the region's bounding box (as 14_train_unet.py)."""
    H, W = region.shape
    r0, r1, c0, c1 = _bbox(region)
    pad = PATCH // 2
    r0, c0, r1, c1 = max(0, r0 - pad), max(0, c0 - pad), min(H, r1 + pad), min(W, c1 + pad)
    if H < PATCH or W < PATCH:                    # tiny grid: pad the image itself
        ph, pw = max(0, PATCH - H), max(0, PATCH - W)
        chan = np.pad(chan, ((0, 0), (0, ph), (0, pw)), mode="reflect")
        H2, W2 = chan.shape[1:]
    else:
        H2, W2 = H, W
    mu = bundle["mean"].reshape(-1, 1, 1)
    sd = bundle["std"].reshape(-1, 1, 1)
    acc = np.zeros((4, H2, W2), "float32")
    hits = np.zeros((H2, W2), "float32")
    for rs in _starts(r0, max(r1, r0 + PATCH), H2):
        for cs in _starts(c0, max(c1, c0 + PATCH), W2):
            tile = (chan[:, rs:rs + PATCH, cs:cs + PATCH] - mu) / sd
            x = torch.from_numpy(np.ascontiguousarray(tile, dtype="float32"))[None]
            acc[:, rs:rs + PATCH, cs:cs + PATCH] += model(x)[0].numpy() / bundle["temperature"]
            hits[rs:rs + PATCH, cs:cs + PATCH] += 1
    acc, hits = acc[:, :H, :W], hits[:H, :W]
    ok = hits > 0
    acc[:, ok] /= hits[ok]
    e = np.exp(acc - acc.max(0, keepdims=True))
    probs = e / e.sum(0, keepdims=True)
    pred = probs.argmax(0).astype("uint8")
    pred[~ok] = IGNORE
    return pred, probs


def score(pred: np.ndarray, lab: np.ndarray, region: np.ndarray) -> np.ndarray:
    valid = region & (lab != IGNORE) & (pred != IGNORE)
    return M.confusion(pred[valid], lab[valid])


def map_lake(bundle, lake: Lake, variant: str, years=None):
    """Per-year class maps for one lake, plus the confusion matrix against its seeds."""
    model = load_model(bundle)
    region = lake.masks["region"]
    maps, cm = {}, np.zeros((4, 4), "int64")
    for y in years or lake.years:
        x, bad = lake.channels(y, variant)
        pred, _ = predict(model, bundle, x, region)
        pred[~region] = IGNORE
        maps[y] = pred
        lab = lake.labels(y).copy()
        lab[bad] = IGNORE
        cm += score(pred, lab, region)
    return maps, cm


def folds(slugs_by_size: list[str]) -> list[dict]:
    """Nested leave-one-lake-out, same scheme as dl/common.folds()."""
    n = len(slugs_by_size)
    out = []
    for i, test in enumerate(slugs_by_size):
        val = slugs_by_size[(i + 1) % n]
        out.append({"fold": i, "test": test, "val": val,
                    "train": [s for s in slugs_by_size if s not in (test, val)]})
    return out


def save_bundle(bundle, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**bundle, "mean": torch.from_numpy(bundle["mean"]),
                "std": torch.from_numpy(bundle["std"])}, path)


def load_bundle(path):
    b = torch.load(path, map_location="cpu", weights_only=False)
    b["mean"], b["std"] = b["mean"].numpy(), b["std"].numpy()
    return b


