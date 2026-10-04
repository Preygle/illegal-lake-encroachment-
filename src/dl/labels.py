"""
dl/labels.py
============
Builds the four-class seed labels that the segmentation model trains on.

The rule that matters
---------------------
**No label is derived from MNDWI or NDVI.** Those two indices are what the
current baseline thresholds, and they are inputs to the model. If they also
produced the labels, the network would be trained to imitate the baseline it is
supposed to be compared against, and every score would measure agreement with a
threshold rather than agreement with the ground.

So the seeds come from two sources that are independent of the project's own
index rules:

  1. Sentinel-2 Scene Classification Layer (SCL) - ESA's own classifier. Same
     sensor, different algorithm, produced before this project existed.
  2. OpenStreetMap building locations - a different data source entirely.

These are *seeds*, not ground truth. They are meant to be corrected by hand in
QGIS before training; `export_for_qgis()` writes the GeoTIFFs for that. The
report must describe the labels as hand-corrected seeds, never as ground truth.

SCL mapping
-----------
    6 (water)               -> WATER
    4 (vegetation)          -> VEG
    5 (not vegetated)       -> BARE
    0,1,3,8,9,10 (bad)      -> IGNORE   (nodata, saturated, shadow, cloud, cirrus)
    2,7,11 (dark/unclass/snow) -> IGNORE  (ambiguous - let the human decide)

Buildings take priority over the SCL class, because a structure standing on a
drained lake bed is exactly the signal this project exists to find. Where a
building falls on an SCL water pixel the two disagree; those pixels are counted
and written to a QA raster rather than silently resolved.
"""

from __future__ import annotations

import numpy as np

from . import common as C

# SCL -> class. Anything not listed becomes IGNORE.
SCL_TO_CLASS = {6: C.WATER, 4: C.VEG, 5: C.BARE}

# A 20 m pixel is 400 m2 and a Bengaluru building footprint is on the order of
# 100 m2, so one centroid in a pixel is already meaningful evidence of built-up
# cover. Raise this to be stricter about what counts as the built class.
MIN_BUILDINGS_PER_PIXEL = 1


def building_raster() -> np.ndarray:
    """
    Count of OpenStreetMap buildings per grid pixel.

    Built from building centroids (data/processed/osm_building_pts.npy). The
    Overpass query in 01_fetch_nonsatellite.py requests `out center`, so the
    dump holds one point per building and no outlines - there is no footprint
    polygon to rasterise, whether or not the JSON is pulled from Git LFS. A
    centroid decides which 20 m pixel a building counts towards; at 400 m2 per
    pixel against ~100 m2 per building, that is adequate for seeding the built
    class, but it is a point count, not a footprint area.
    """
    from pyproj import Transformer

    bounds, shape, crs, _ = C.grid()
    counts = np.zeros(shape, dtype="int32")
    pts = np.load(C.PROC / "osm_building_pts.npy")           # lon, lat
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tf.transform(pts[:, 0], pts[:, 1])
    res = (bounds[2] - bounds[0]) / shape[1]
    col = np.floor((x - bounds[0]) / res).astype("int64")
    row = np.floor((bounds[3] - y) / res).astype("int64")
    ok = (col >= 0) & (col < shape[1]) & (row >= 0) & (row < shape[0])
    np.add.at(counts, (row[ok], col[ok]), 1)
    print(f"  buildings: {len(pts):,} centroids, {int(ok.sum()):,} on grid, "
          f"{int((counts > 0).sum()):,} pixels occupied")
    return counts


def seed_labels(year: int, bldg: np.ndarray, rasters: dict):
    """
    Seed label raster for one year, plus a QA record.

    Returns (labels uint8 (H, W), qa dict). Pixels outside every lake's region
    stay IGNORE, so nothing beyond the six study areas is ever trained on.
    """
    import sys

    sys.path.insert(0, str(C.ROOT / "src"))
    import s2_common as s2c

    st = C.load_stack(year)
    scl = st["scl"]
    _, shape, _, _ = C.grid()

    lab = np.full(shape, C.IGNORE, dtype="uint8")
    for scl_val, cls in SCL_TO_CLASS.items():
        lab[scl == scl_val] = cls

    built = bldg >= MIN_BUILDINGS_PER_PIXEL
    conflict = built & (lab == C.WATER)
    lab[built] = C.BUILT

    bad = np.isin(scl, s2c.SCL_BAD)
    lab[bad] = C.IGNORE

    region = np.zeros(shape, bool)
    for r in rasters.values():
        region |= r["region"]
    lab[~region] = C.IGNORE

    qa = {"year": year,
          "labelled_px": int((lab != C.IGNORE).sum()),
          "region_px": int(region.sum()),
          "cloud_px_in_region": int((bad & region).sum()),
          "building_water_conflicts": int((conflict & region).sum())}
    for cls, name in enumerate(C.CLASS_NAMES):
        qa[f"n_{name}"] = int((lab == cls).sum())
    return lab, qa


def build_all():
    """Seed labels for every year -> outputs/dl/labels/labels_<year>.npy."""
    out_dir = C.DL / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] rasterising OpenStreetMap buildings")
    bldg = building_raster()
    np.save(out_dir / "building_count.npy", bldg)

    print("[2/3] lake regions")
    rasters = C.lake_rasters()
    for lake, r in rasters.items():
        print(f"  {lake:16s} inside={int(r['inside'].sum()):6,d} px  "
              f"collar={int(r['collar'].sum()):7,d} px")

    print("[3/3] seed labels per year")
    qas = []
    for y in C.YEARS:
        lab, qa = seed_labels(y, bldg, rasters)
        np.save(out_dir / f"labels_{y}.npy", lab)
        qas.append(qa)
        frac = qa["labelled_px"] / max(qa["region_px"], 1)
        print(f"  {y}  labelled {qa['labelled_px']:7,d}/{qa['region_px']:,d} "
              f"({frac:5.1%})  " +
              "  ".join(f"{n}={qa['n_' + n]:,d}" for n in C.CLASS_NAMES) +
              f"  conflicts={qa['building_water_conflicts']:,d}")

    import pandas as pd

    df = pd.DataFrame(qas)
    df.to_csv(out_dir / "label_qa.csv", index=False)
    return df


def export_for_qgis():
    """
    Write the seed labels and a true-colour composite per year as GeoTIFFs, so
    they can be opened in QGIS and corrected by hand.

    Correct the label raster in place, save it back as GeoTIFF with the same
    name, then run `python src/12_make_labels.py --import-corrected` to fold the
    corrections back into the .npy files the trainer reads.
    """
    import rasterio
    import sys

    sys.path.insert(0, str(C.ROOT / "src"))
    import s2_common as s2c

    out_dir = C.DL / "labels" / "qgis"
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds, shape, crs, transform = C.grid()
    profile = dict(driver="GTiff", height=shape[0], width=shape[1],
                   crs=crs, transform=transform, compress="deflate")

    for y in C.YEARS:
        lab = np.load(C.DL / "labels" / f"labels_{y}.npy")
        with rasterio.open(out_dir / f"labels_{y}.tif", "w", count=1,
                           dtype="uint8", nodata=C.IGNORE, **profile) as dst:
            dst.write(lab, 1)
            dst.write_colormap(1, {0: (33, 102, 172), 1: (27, 120, 55),
                                   2: (201, 138, 30), 3: (200, 40, 40),
                                   255: (0, 0, 0)})

        st = C.load_stack(y)
        rgb = (np.clip(s2c.rgb(st), 0, 1) * 255).astype("uint8")
        with rasterio.open(out_dir / f"truecolor_{y}.tif", "w", count=3,
                           dtype="uint8", **profile) as dst:
            for i in range(3):
                dst.write(rgb[:, :, i], i + 1)
        print(f"  {y}  labels_{y}.tif + truecolor_{y}.tif")
    print(f"\n  open these in QGIS from {out_dir}")


def import_corrected():
    """Read hand-corrected GeoTIFFs back over the .npy seed labels."""
    import rasterio

    qgis_dir = C.DL / "labels" / "qgis"
    n = 0
    for y in C.YEARS:
        tif = qgis_dir / f"labels_{y}.tif"
        if not tif.exists():
            continue
        with rasterio.open(tif) as src:
            lab = src.read(1)
        before = np.load(C.DL / "labels" / f"labels_{y}.npy")
        changed = int((lab != before).sum())
        np.save(C.DL / "labels" / f"labels_{y}.npy", lab.astype("uint8"))
        print(f"  {y}  imported, {changed:,d} pixels changed")
        n += 1
    if n == 0:
        print("  nothing to import - run --export-qgis first")
