"""
21_anylake_metrics.py
=====================
Training, validation and test accuracy of the any-lake U-Nets (raw and
enhanced bands), side by side.

  train      the deployable model on the 5 lakes it was trained on (in-sample)
  val        the same model on Varthur Lake, used only for early stopping and
             temperature scaling
  test       leave-one-lake-out: each lake scored by the fold model that never
             saw it (from 19_anylake_library.py) - the honest number

All scores are against the seed labels (ESA Scene Classification Layer +
OpenStreetMap buildings), every year 2019-2025, inside lake + 500 m.
The gap between train and test is the overfitting / generalisation gap.

    python src/21_anylake_metrics.py
Writes outputs/anylake/training_metrics.{csv,json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anylake import MODELS, OUT  # noqa: E402
from anylake import unet_lib as U  # noqa: E402
from anylake.lake import Lake  # noqa: E402
from dl import metrics as M  # noqa: E402

COLS = ["accuracy", "macro_f1", "kappa", "macro_iou", "f1_water", "f1_vegetation", "f1_bare", "f1_built"]


def main():
    rows = []
    for v in ("raw", "enh"):
        b = U.load_bundle(MODELS / f"unet_{v}.pt")
        split = {s: "train" for s in b["train_lakes"]} | {b["val_lake"]: "val"}
        pooled = {"train": np.zeros((4, 4), "int64"), "val": np.zeros((4, 4), "int64")}
        for slug, part in split.items():
            _, cm = U.map_lake(b, Lake(slug), v)
            pooled[part] += cm
            rows.append({"variant": v, "split": part, "lake": Lake(slug).name, **M.metrics_from_cm(cm)})
        for part, cm in pooled.items():
            rows.append({"variant": v, "split": part, "lake": "ALL (pooled)", **M.metrics_from_cm(cm)})
        print(f"  {v}: train/val scored")

    exp = pd.read_csv(OUT / "enhancement_experiment.csv")
    for v in ("raw", "enh"):
        e = exp[exp.variant == v]
        for _, r in e.iterrows():
            rows.append({"variant": v, "split": "test", "lake": r["lake"], **{c: r[c] for c in COLS}})
        rows.append({"variant": v, "split": "test", "lake": "ALL (mean of 6)", **{c: e[c].mean() for c in COLS}})

    df = pd.DataFrame(rows)[["variant", "split", "lake"] + COLS]
    df.to_csv(OUT / "training_metrics.csv", index=False)
    summ = df[df.lake.str.startswith("ALL")].set_index(["variant", "split"])[COLS].round(4)
    (OUT / "training_metrics.json").write_text(
        json.dumps({f"{v}/{s}": r.to_dict() for (v, s), r in summ.iterrows()}, indent=1), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(summ.to_string())


if __name__ == "__main__":
    main()
