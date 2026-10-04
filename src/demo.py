"""
demo.py
=======
Live demonstration: map one lake in one year with the trained model, and say
what the change from the previous year looks like.

    python src/demo.py --lake bellandur --year 2021
    python src/demo.py --lake madiwala --year 2020 --rf
    python src/demo.py --list

The U-Net used for each lake is the one from the fold in which that lake was
held out, so the model has never seen the lake it is mapping. That makes the
demo an honest test rather than a replay of training data.

Writes outputs/demo/<lake>_<year>.png. Nothing opens on screen.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dl import common as C
from dl import dataset as D
from dl import metrics as M
from dl import viz as V

OUT = C.ROOT / "outputs" / "demo"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, C.ROOT / "src" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_lake(text: str) -> str:
    t = text.strip().lower()
    for lake in C.lake_order():
        if t in lake.lower():
            return lake
    raise SystemExit(f"no lake matches '{text}'. Choose from: "
                     + ", ".join(C.lake_order()))


def held_out_fold(lake: str) -> dict:
    for f in C.folds():
        if f["test"] == lake:
            return f
    raise SystemExit(f"no fold holds out {lake}")


def unet_map_all_years(lake, fold, store, index):
    """U-Net land cover for every year, from the fold that never saw this lake."""
    import torch
    from dl.unet import UNet

    ckpt = C.DL / "unet" / f"fold{fold['fold']}_best.pt"
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} missing - run python src/14_train_unet.py first")
    model = UNet()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    train = _load("t14", "14_train_unet.py")
    # Normalisation must match training exactly: statistics from the fold's
    # training lakes only.
    mean, std = D.fit_norm(store, index, fold["train"])
    maps = {}
    for y in C.YEARS:
        maps[y], _ = train.predict_region(model, store, lake, y, mean, std,
                                          torch.device("cpu"))
    return maps, ckpt.name


def rf_map(lake, fold, store, year):
    """Random Forest trained live on the same fold's lakes (~30 s)."""
    from sklearn.ensemble import RandomForestClassifier

    rf_mod = _load("t13", "13_train_rf.py")
    rng = np.random.default_rng(42)
    X, y = rf_mod.pixels_for(store, fold["train"] + [fold["val"]], rng, 200_000)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                min_samples_leaf=2, n_jobs=rf_mod.N_JOBS,
                                random_state=42).fit(X, y)
    chan = store.chan[year]
    flat = chan.reshape(chan.shape[0], -1).T
    region = store.rasters[lake]["region"].reshape(-1)
    out = np.full(region.shape, C.IGNORE, dtype="uint8")
    out[region] = rf.predict(flat[region])
    return out.reshape(chan.shape[1:])


def reading_for(lake, year, maps, rasters) -> str:
    """What the desilting rule says about the change into `year`."""
    import pandas as pd

    risk = _load("t16", "16_risk_model.py")
    rows = []
    for y, m in maps.items():
        inside = rasters[lake]["inside"] & (m != C.IGNORE)
        n = max(int(inside.sum()), 1)
        rec = {"lake": lake, "year": y}
        for k, name in enumerate(C.CLASS_NAMES):
            rec[f"{name}_frac"] = float((inside & (m == k)).sum()) / n
        rows.append(rec)
    rule = risk.desilting_rule(pd.DataFrame(rows))
    hit = rule[rule.to_year == year]
    return hit.iloc[0].reading if len(hit) else "first year in the series - nothing to compare"


def fence_counts(lake: str, year: int) -> str:
    import pandas as pd

    parts = []
    ob = C.DL / "gee" / "open_buildings_intrusion.csv"
    if ob.exists():
        d = pd.read_csv(ob)
        r = d[(d.lake == lake) & (d.year == year)]
        if len(r):
            parts.append(f"{r.iloc[0].ob_buildings_ring_30m:.0f} buildings inside the 30 m fence, "
                         f"{r.iloc[0].ob_buildings_ring_75m:.0f} inside 75 m "
                         f"(Open Buildings Temporal, {year})")
        else:
            parts.append(f"no Open Buildings data for {year} - the dataset ends in 2023")
    return "; ".join(parts) or "run src/17_fetch_gee.py for building counts"


def figure(lake, year, panels, rasters, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import viz_style as vs

    t = vs.apply("light")
    win = V.window(lake, rasters)
    h, w = win[1] - win[0], win[3] - win[2]
    n = len(panels) + 1
    pw = 4.4 if w / h < 1.3 else 5.2
    fig, axes = plt.subplots(1, n, figsize=(pw * n, pw * h / w + 1.6))
    V.draw_rgb(axes[0], year, lake, rasters, win)
    axes[0].set_title(f"Sentinel-2, {year}", fontsize=11)
    for ax, (name, cmap_) in zip(axes[1:], panels):
        V.draw_classes(ax, cmap_, lake, rasters, win, year=year)
        ax.set_title(name, fontsize=11)
        ax.text(0.0, -0.04, V.comp_line(V.composition_ha(cmap_, lake, rasters)),
                transform=ax.transAxes, ha="left", va="top", fontsize=8.6,
                color=t.ink2)
    fig.legend(handles=V.legend_handles(), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06), fontsize=9.5)
    fig.suptitle(f"{lake}, {year}", x=0.01, ha="left", fontsize=14,
                 fontweight="600", color=t.ink)
    fig.text(0.01, 0.005, "Solid line: lake footprint (OpenStreetMap).  "
             "Dashed line: 30 m statutory buffer.  Each lake is mapped by a "
             "model that never trained on it.", fontsize=8, color=t.muted)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(a):
    if a.list:
        for f in C.folds():
            print(f"  {f['test']:16s} mapped by fold {f['fold']} "
                  f"(trained on {', '.join(f['train'] + [f['val']])})")
        return

    t0 = time.time()
    lake = resolve_lake(a.lake)
    if a.year not in C.YEARS:
        raise SystemExit(f"year must be one of {C.YEARS}")
    fold = held_out_fold(lake)

    print(f"\n  {lake}, {a.year}")
    print(f"  model: U-Net from fold {fold['fold']} - never trained on {lake}")
    print("  loading 7 years of Sentinel-2 ...", flush=True)
    store = D.PatchStore()
    index = D.build_index(store.rasters, verbose=False)
    rasters = store.rasters

    maps, ckpt = unet_map_all_years(lake, fold, store, index)
    unet = maps[a.year]
    rule = M.baseline_predict(store.chan[a.year])

    panels = [("MNDWI / NDVI rule (old method)", rule),
              ("U-Net (deep learning)", unet)]
    if a.rf:
        print("  training the Random Forest on the same fold (~30 s) ...", flush=True)
        panels.append(("Random Forest", rf_map(lake, fold, store, a.year)))

    print(f"\n  inside the lake footprint ({rasters[lake]['inside'].sum() * C.pixel_area_ha():.1f} ha)")
    print(f"  {'':34s} {'water':>8s} {'hyacinth':>9s} {'bare':>8s} {'built':>8s}")
    for name, m in panels:
        ha = V.composition_ha(m, lake, rasters)
        print(f"  {name:34s} {ha['water']:8.1f} {ha['vegetation']:9.1f} "
              f"{ha['bare']:8.1f} {ha['built']:8.1f}")

    print(f"\n  change from {a.year - 1}: {reading_for(lake, a.year, maps, rasters)}"
          if a.year > C.YEARS[0] else f"\n  {a.year} is the first year - no change to read")
    print(f"  geo-fence: {fence_counts(lake, a.year)}")

    path = OUT / f"{lake.split()[0].lower()}_{a.year}.png"
    figure(lake, a.year, panels, rasters, path)
    print(f"\n  map saved: {path.relative_to(C.ROOT)}")
    print(f"  done in {time.time() - t0:.1f}s  (weights: outputs/dl/unet/{ckpt})\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lake", default="bellandur",
                    help="any part of the name: bellandur, varthur, madiwala, "
                         "hebbal, ulsoor, sankey")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--rf", action="store_true",
                    help="also train and show the Random Forest (~30 s)")
    ap.add_argument("--list", action="store_true",
                    help="show which fold maps each lake")
    main(ap.parse_args())
