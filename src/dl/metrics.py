"""
dl/metrics.py
=============
Segmentation metrics, computed from one confusion matrix so every model in the
comparison is scored identically.

Accuracy alone is never reported. Bare soil is roughly half the labelled pixels
and open water under 8%, so a model that predicted "bare" everywhere would look
respectable on accuracy while being useless for the question the project asks.
What is reported instead:

* **per-class IoU** - the standard segmentation measure, and the one that shows
  *which* class fails on *which* lake.
* **macro-F1** - unweighted across the four classes, so the rare water class
  counts as much as the common bare class.
* **Cohen's kappa** - the remote-sensing convention; agreement corrected for
  what chance alone would give.
* **the water/vegetation confusion cell** - the hyacinth question, stated as a
  number rather than left in a picture.
* **Expected Calibration Error** - because these probabilities feed the
  per-lake-year composition, so over-confidence propagates.
"""

from __future__ import annotations

import numpy as np

from . import common as C


def confusion(pred: np.ndarray, true: np.ndarray,
              n_classes: int = C.N_CLASSES) -> np.ndarray:
    """Rows = truth, columns = prediction. IGNORE pixels are dropped."""
    valid = true != C.IGNORE
    t, p = true[valid].astype(np.int64), pred[valid].astype(np.int64)
    k = (t >= 0) & (t < n_classes) & (p >= 0) & (p < n_classes)
    return np.bincount(t[k] * n_classes + p[k],
                       minlength=n_classes ** 2).reshape(n_classes, n_classes)


def metrics_from_cm(cm: np.ndarray) -> dict:
    cm = cm.astype("float64")
    n = cm.sum()
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)

    # A class absent from the truth is not scored, rather than scored zero -
    # Sankey Tank has no meaningful bare-bed pixels in a wet year.
    present = support > 0
    out = {
        "n_pixels": int(n),
        "accuracy": float(tp.sum() / n) if n else float("nan"),
        "macro_f1": float(np.nanmean(np.where(present, f1, np.nan))),
        "macro_iou": float(np.nanmean(np.where(present, iou, np.nan))),
        "kappa": kappa_from_cm(cm),
    }
    for i, name in enumerate(C.CLASS_NAMES):
        out[f"iou_{name}"] = float(iou[i]) if present[i] else float("nan")
        out[f"f1_{name}"] = float(f1[i]) if present[i] else float("nan")
        out[f"support_{name}"] = int(support[i])

    # The hyacinth question: of truly-water pixels, what fraction were called
    # vegetation, and vice versa.
    w, v = C.WATER, C.VEG
    out["water_as_veg"] = float(cm[w, v] / support[w]) if support[w] else float("nan")
    out["veg_as_water"] = float(cm[v, w] / support[v]) if support[v] else float("nan")
    return out


def kappa_from_cm(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.diag(cm).sum() / n
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n ** 2)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def expected_calibration_error(probs: np.ndarray, true: np.ndarray,
                               n_bins: int = 10) -> float:
    """
    Mean gap between confidence and accuracy, binned by confidence.

    probs (N, C) already softmaxed, true (N,) with no IGNORE.
    """
    if len(true) == 0:
        return float("nan")
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == true).astype("float64")
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(ece / len(true))


def format_cm(cm: np.ndarray) -> str:
    """Row-normalised confusion matrix as text, truth down and prediction across."""
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    head = "true\\pred   " + "".join(f"{n[:9]:>10s}" for n in C.CLASS_NAMES)
    rows = [head]
    for i, name in enumerate(C.CLASS_NAMES):
        rows.append(f"{name:<11s}" +
                    "".join(f"{cmn[i, j]:10.3f}" for j in range(len(C.CLASS_NAMES))) +
                    f"   (n={int(cm[i].sum()):,d})")
    return "\n".join(rows)


def baseline_predict(chan: np.ndarray) -> np.ndarray:
    """
    The current pipeline's rule, as a four-class predictor: MNDWI > 0 is water,
    else NDVI > 0.30 is vegetation, else bare.

    It can never predict the built class. That is not an oversight in the
    implementation - it is the structural limitation of an index-threshold rule,
    and it is precisely why the comparison is worth running.
    """
    import sys

    sys.path.insert(0, str(C.ROOT / "src"))
    import s2_common as s2c

    mndwi = chan[C.CHANNELS.index("mndwi")]
    ndvi = chan[C.CHANNELS.index("ndvi")]
    pred = np.full(mndwi.shape, C.BARE, dtype="uint8")
    pred[ndvi > s2c.NDVI_VEG] = C.VEG
    pred[mndwi > s2c.MNDWI_WATER] = C.WATER
    return pred
