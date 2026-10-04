"""
13_train_rf.py
==============
The classical baseline the U-Net has to beat, plus the index-threshold rule that
both of them have to beat.

Three predictors, six identical leave-one-lake-out folds:

  1. **MNDWI/NDVI rule** - what the project does today. Cannot predict built-up
     at all, which is the point.
  2. **Random Forest** - the same 8 channels, one pixel at a time, no spatial
     context whatsoever.
  3. (14_train_unet.py) - the U-Net, which adds exactly that context.

Running this first is deliberate: if the folds or the labels are broken, this
catches it in a minute or two rather than after a night of training.

    python src/13_train_rf.py
    python src/13_train_rf.py --max-pixels 400000

Writes outputs/dl/rf/{per_fold.csv, confusion_<lake>.txt, summary.json}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dl import common as C
from dl import dataset as D
from dl import metrics as M

OUT = C.DL / "rf"
# Half the cores by default, so a run does not monopolise the machine.
N_JOBS = max(1, (__import__("os").cpu_count() or 2) // 2)


def pixels_for(store: D.PatchStore, lakes: list[str], rng, max_px: int | None):
    """Flatten every labelled pixel of these lakes into (X, y), optionally subsampled."""
    xs, ys = [], []
    for lake in lakes:
        region = store.rasters[lake]["region"]
        for year in C.YEARS:
            lab = store.label_for(lake, year)
            valid = region & (lab != C.IGNORE)
            if not valid.any():
                continue
            xs.append(store.chan[year][:, valid].T)
            ys.append(lab[valid])
    X = np.concatenate(xs).astype("float32")
    y = np.concatenate(ys).astype("int64")
    if max_px and len(y) > max_px:
        keep = rng.choice(len(y), max_px, replace=False)
        X, y = X[keep], y[keep]
    return X, y


def main(max_pixels: int, n_estimators: int, seed: int):
    from sklearn.ensemble import RandomForestClassifier

    C.set_seed(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("[1/3] loading scenes and labels")
    store = D.PatchStore()

    print("[2/3] folds")
    folds = C.folds()
    for f in folds:
        print(f"  fold {f['fold']}  test={f['test']:16s} val={f['val']}")

    print("[3/3] per fold")
    rows, summaries = [], {}
    for f in folds:
        t0 = time.time()
        test = f["test"]
        train_lakes = f["train"] + [f["val"]]     # the RF needs no early stopping

        Xtr, ytr = pixels_for(store, train_lakes, rng, max_pixels)
        Xte, yte = pixels_for(store, [test], rng, None)

        rf = RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            min_samples_leaf=2, n_jobs=N_JOBS, random_state=seed)
        rf.fit(Xtr, ytr)
        pred_rf = rf.predict(Xte)

        # The index rule, scored on exactly the same pixels.
        base_pred, base_true = [], []
        region = store.rasters[test]["region"]
        for year in C.YEARS:
            lab = store.label_for(test, year)
            valid = region & (lab != C.IGNORE)
            if not valid.any():
                continue
            base_pred.append(M.baseline_predict(store.chan[year])[valid])
            base_true.append(lab[valid])
        base_pred = np.concatenate(base_pred)
        base_true = np.concatenate(base_true)

        for name, cm in (("random_forest", M.confusion(pred_rf, yte)),
                         ("mndwi_rule", M.confusion(base_pred, base_true))):
            m = M.metrics_from_cm(cm)
            m.update(model=name, fold=f["fold"], test_lake=test,
                     n_train_px=len(ytr))
            rows.append(m)
            (OUT / f"confusion_{name}_{test.replace(' ', '_')}.txt").write_text(
                M.format_cm(cm), encoding="utf-8")

        rf_f1 = rows[-2]["macro_f1"]
        base_f1 = rows[-1]["macro_f1"]
        summaries[test] = {"rf_macro_f1": rf_f1, "rule_macro_f1": base_f1}
        print(f"  fold {f['fold']}  {test:16s}  "
              f"RF macro-F1 {rf_f1:.3f}   rule {base_f1:.3f}   "
              f"({len(yte):,d} test px, {time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_fold.csv", index=False)

    print("\n=== held-out means over 6 lakes (spread in brackets) ===")
    for name in ("mndwi_rule", "random_forest"):
        s = df[df.model == name]
        print(f"  {name:14s} macro-F1 {s.macro_f1.mean():.3f} "
              f"[{s.macro_f1.min():.3f}-{s.macro_f1.max():.3f}]   "
              f"kappa {s.kappa.mean():.3f}   "
              f"IoU water {s.iou_water.mean():.3f}  veg {s.iou_vegetation.mean():.3f}  "
              f"bare {s.iou_bare.mean():.3f}  built {s.iou_built.mean():.3f}")

    rf_mean = float(df[df.model == "random_forest"].macro_f1.mean())
    (OUT / "summary.json").write_text(json.dumps(
        {"per_lake": summaries,
         "rf_mean_macro_f1": rf_mean,
         "rule_mean_macro_f1": float(df[df.model == "mndwi_rule"].macro_f1.mean()),
         "note": "The U-Net must exceed rf_mean_macro_f1 by more than 0.02 to be "
                 "recommended over the Random Forest (MODEL_DESIGN.md 2.10)."},
        indent=2), encoding="utf-8")
    print(f"\n  the U-Net has to clear {rf_mean + 0.02:.3f} macro-F1 to be "
          f"recommended (RF {rf_mean:.3f} + the 2-point rule)")
    print(f"  -> {OUT}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pixels", type=int, default=200_000,
                    help="training pixels per fold (0 = all)")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.max_pixels or None, a.n_estimators, a.seed)
