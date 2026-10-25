"""
19_anylake_library.py
=====================
Builds the model library behind "Any lake in India", and runs the image
enhancement experiment.

  1. Fetch the six Bengaluru lakes again, this time with all 12 Sentinel-2
     bands, each on its own grid (anylake/fetch.py). OpenStreetMap comes from
     the dump the original pipeline already has.
  2. Enhancement experiment: leave-one-lake-out U-Net, trained twice - on raw
     bands and on enhanced bands (anylake/enhance.py). Same folds, seeds,
     patches and settings; only the input differs. Each lake is scored by the
     model that never saw it.
  3. Deployable models: one U-Net per variant trained on all six lakes, used
     to map any lake that is searched later. A searched lake is never in this
     training set, so its score is an honest out-of-lake score.

Decision rule, fixed before the run: the enhanced model is used downstream
only if it beats the raw model by at least 0.01 mean macro-F1 across the six
held-out lakes; otherwise raw is used, since it is simpler.

    python src/19_anylake_library.py                  # everything (~1 hour on CPU)
    python src/19_anylake_library.py --skip-experiment

Writes outputs/anylake/enhancement_experiment.{csv,json},
outputs/anylake/models/unet_{raw,enh}.pt and out-of-fold maps per lake.
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

from anylake import LAKES, MODELS, OUT  # noqa: E402
from anylake import fetch as F  # noqa: E402
from anylake import unet_lib as U  # noqa: E402
from anylake.lake import IGNORE, Lake  # noqa: E402
from anylake.search import slugify  # noqa: E402
from dl import metrics as M  # noqa: E402

VARIANTS = {"raw": "Raw bands", "enh": "Enhanced bands"}
MIN_GAIN = 0.01


def log(*a):
    print(*a, flush=True)


def library_slugs() -> list[str]:
    """Register and fetch the six Bengaluru lakes; return slugs, largest first."""
    fc = json.loads((LAKES.parents[1] / "data" / "processed" / "lake_polygons.geojson")
                    .read_text(encoding="utf-8"))
    slugs = []
    for f in fc["features"]:
        name = f["properties"]["lake"]
        rec = {"name": name, "display": f"{name}, Bengaluru, Karnataka",
               "district": "Bengaluru Urban", "state": "Karnataka", "lat": None, "lon": None,
               "osm_type": "", "osm_id": None, "area_ha": None, "slug": slugify(name),
               "geometry": f["geometry"], "library": True}
        meta = F.register(rec)
        log(f"\n  {name}  grid {meta['grid']['shape']}")
        t0 = time.time()
        F.fetch(meta, log)
        log(f"  {name}: imagery ready ({time.time() - t0:.0f}s)")
        slugs.append(rec["slug"])
    lakes = {s: Lake(s) for s in slugs}
    for s, L in lakes.items():
        for y in L.years:
            L.labels(y)
    return sorted(slugs, key=lambda s: -float(lakes[s].masks["inside"].sum()))


def experiment(slugs: list[str], variants, cfg=None) -> pd.DataFrame:
    rows = []
    for v in variants:
        log(f"\n=== U-Net on {VARIANTS[v].lower()} ===")
        store = U.LibraryStore(slugs, v, log)
        index = U.build_index(store)
        log(f"  {len(index)} patches")
        for fold in U.folds(slugs):
            t0 = time.time()
            log(f"\n  fold {fold['fold']}: test={fold['test']}  val={fold['val']}")
            bundle = U.train(store, index, fold["train"], fold["val"], cfg,
                             seed_offset=fold["fold"], log=log)
            lake = store.lakes[fold["test"]]
            maps, cm = U.map_lake(bundle, lake, v)
            for y, m in maps.items():
                np.save(lake.dir / f"oof_{v}_{y}.npy", m)
            res = M.metrics_from_cm(cm)
            res.update(variant=v, lake=lake.name, slug=fold["test"], fold=fold["fold"],
                       best_epoch=bundle["best_epoch"], temperature=bundle["temperature"],
                       seconds=round(time.time() - t0, 1))
            rows.append(res)
            log(f"    macro-F1 {res['macro_f1']:.3f}  kappa {res['kappa']:.3f}  IoU water "
                f"{res['iou_water']:.3f} veg {res['iou_vegetation']:.3f} bare {res['iou_bare']:.3f} "
                f"built {res['iou_built']:.3f}  ({res['seconds']:.0f}s)")
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    piv = df.pivot_table(index="lake", columns="variant", values="macro_f1")
    out = {"per_lake_macro_f1": {v: piv[v].round(4).to_dict() for v in piv.columns},
           "mean_macro_f1": {v: round(float(piv[v].mean()), 4) for v in piv.columns},
           "mean_kappa": {v: round(float(df[df.variant == v].kappa.mean()), 4)
                          for v in piv.columns},
           "rule": f"enhanced is used only if it beats raw by >= {MIN_GAIN} mean macro-F1"}
    if {"raw", "enh"} <= set(piv.columns):
        gain = float(piv["enh"].mean() - piv["raw"].mean())
        out["gain_enh_minus_raw"] = round(gain, 4)
        out["lakes_improved"] = int((piv["enh"] > piv["raw"]).sum())
        try:
            from scipy.stats import wilcoxon

            out["wilcoxon_p"] = round(float(wilcoxon(piv["enh"], piv["raw"]).pvalue), 4)
        except Exception:                                         # noqa: BLE001
            pass
        out["chosen"] = "enh" if gain >= MIN_GAIN else "raw"
    else:
        out["chosen"] = piv.columns[0]
    return out


def deploy(slugs: list[str], variants, cfg=None) -> None:
    val = slugs[1]
    train_lakes = [s for s in slugs if s != val]
    for v in variants:
        t0 = time.time()
        log(f"\n=== deployable U-Net, {VARIANTS[v].lower()} (all six lakes, val={val}) ===")
        store = U.LibraryStore(slugs, v, log)
        index = U.build_index(store)
        bundle = U.train(store, index, train_lakes, val, cfg, seed_offset=100, log=log)
        bundle["library"] = slugs
        U.save_bundle(bundle, MODELS / f"unet_{v}.pt")
        log(f"  -> {MODELS / f'unet_{v}.pt'}  ({time.time() - t0:.0f}s)")


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("[1/3] library: six Bengaluru lakes, all 12 bands")
    slugs = library_slugs()
    (OUT / "library.json").write_text(json.dumps(slugs), encoding="utf-8")

    if not args.skip_experiment:
        log("\n[2/3] enhancement experiment (leave-one-lake-out)")
        df = experiment(slugs, args.variants, {"epochs": args.epochs})
        df.to_csv(OUT / "enhancement_experiment.csv", index=False)
        summary = summarise(df)
        (OUT / "enhancement_experiment.json").write_text(json.dumps(summary, indent=2),
                                                        encoding="utf-8")
        log("\n=== result ===")
        log(df.pivot_table(index="lake", columns="variant", values="macro_f1").round(3).to_string())
        log(json.dumps({k: summary[k] for k in summary if k != "per_lake_macro_f1"}, indent=1))

    if not args.skip_deploy:
        log("\n[3/3] deployable models")
        deploy(slugs, args.variants, {"epochs": args.epochs})
    log(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["raw", "enh"], choices=list(VARIANTS))
    ap.add_argument("--skip-experiment", action="store_true")
    ap.add_argument("--skip-deploy", action="store_true")
    ap.add_argument("--epochs", type=int, default=100, help="2 for a quick smoke test")
    main(ap.parse_args())
