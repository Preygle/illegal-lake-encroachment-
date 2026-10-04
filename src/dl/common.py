"""
dl/common.py
============
Shared plumbing for the segmentation model: the analysis grid, lake geometry on
that grid, the eight input channels, and the leave-one-lake-out folds.

Everything downstream imports the grid from here rather than recomputing it, so
labels, patches, predictions and the geo-fence all sit on identical pixels.

Grid
----
EPSG:32643, 20 m, 843 x 1095, bounds from data/processed/grid_meta.npy. That is
the grid 02_fetch_satellite.py wrote the cached scenes onto.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
S2DIR = RAW / "s2"
DL = ROOT / "outputs" / "dl"

YEARS = list(range(2019, 2026))

# --- classes -------------------------------------------------------------
WATER, VEG, BARE, BUILT = 0, 1, 2, 3
IGNORE = 255
CLASS_NAMES = ["water", "vegetation", "bare", "built"]
N_CLASSES = 4

# --- input channels ------------------------------------------------------
# SCL is deliberately absent: it seeds the labels and masks cloud, so feeding it
# in as well would let the network copy ESA's classifier and make any agreement
# we later report with SCL circular.
CHANNELS = ["blue", "green", "red", "nir", "swir16", "ndwi", "mndwi", "ndvi"]
N_CHANNELS = len(CHANNELS)

# --- geo-fence rings, metres --------------------------------------------
# 30 m is the Bengaluru Master Plan / KTCDA figure; 75 m is the superseded
# National Green Tribunal figure. Both are reported as bracketing cases - see
# research-dl-lake-encroachment/RESEARCH_DEEP_LEARNING.md section 4.
FENCES_M = (30, 75)
COLLAR_M = 500          # labelling and patch-sampling region around each lake


def set_seed(seed: int = 42) -> None:
    """Seed every generator that can affect a run."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding - torch.manual_seed alone does not reach workers."""
    import torch

    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==========================================================================
# Grid
# ==========================================================================
def grid():
    """(bounds, shape, crs, transform) for the shared 20 m analysis grid."""
    from rasterio.transform import from_bounds

    gm = np.load(PROC / "grid_meta.npy")
    minx, miny, maxx, maxy, h, w = gm
    shape = (int(h), int(w))
    bounds = (float(minx), float(miny), float(maxx), float(maxy))
    crs = (PROC / "grid_crs.txt").read_text(encoding="utf-8").strip()
    return bounds, shape, crs, from_bounds(*bounds, shape[1], shape[0])


def pixel_area_ha() -> float:
    bounds, shape, _, _ = grid()
    res = (bounds[2] - bounds[0]) / shape[1]
    return (res ** 2) / 10_000.0


# ==========================================================================
# Lake geometry on the grid
# ==========================================================================
def lake_geoms() -> dict:
    """Lake polygons reprojected from lon/lat to the grid CRS."""
    from pyproj import Transformer
    from shapely.geometry import shape as shp_shape
    from shapely.ops import transform as shp_transform

    _, _, crs, _ = grid()
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    fc = json.loads((PROC / "lake_polygons.geojson").read_text(encoding="utf-8"))
    return {f["properties"]["lake"]: shp_transform(tf, shp_shape(f["geometry"]))
            for f in fc["features"]}


def lake_rasters() -> dict:
    """
    Per lake: the footprint mask, the 500 m labelling collar, and one boolean
    ring per fence distance.

    `region` (footprint + collar) is the only area that ever carries a label or
    contributes a training patch. Everything outside it is IGNORE, which is what
    keeps one lake's pixels out of another lake's fold.
    """
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping

    _, shape, _, transform = grid()

    def burn(geom):
        if geom.is_empty:
            return np.zeros(shape, bool)
        return geometry_mask([mapping(geom)], out_shape=shape,
                             transform=transform, invert=True)

    out = {}
    for lake, poly in lake_geoms().items():
        inside = burn(poly)
        collar = burn(poly.buffer(COLLAR_M).difference(poly))
        rec = {"geom": poly, "inside": inside, "collar": collar,
               "region": inside | collar}
        for m in FENCES_M:
            rec[f"ring_{m}m"] = burn(poly.buffer(m).difference(poly))
        out[lake] = rec
    return out


# ==========================================================================
# Input channels
# ==========================================================================
def load_stack(year: int) -> dict:
    """Cached digital numbers for one year."""
    with np.load(S2DIR / f"s2_{year}.npz") as z:
        return {k: z[k] for k in z.files}


def channels(year: int) -> tuple[np.ndarray, np.ndarray]:
    """
    (C, H, W) float32 feature stack and the bad-pixel mask for one year.

    Indices come from s2_common.indices(), which clips them to [-1, 1]; the
    reflectance bands are left unscaled here because per-channel normalisation
    has to be fitted on the training fold only, not on the whole scene.
    """
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    import s2_common as s2c

    st = load_stack(year)
    ix = s2c.indices(st)
    arr = np.stack([ix[c] for c in CHANNELS]).astype("float32")
    return arr, ix["cloud"]


# ==========================================================================
# Folds
# ==========================================================================
def lake_order() -> list[str]:
    """Lakes in a fixed, reproducible order (largest first)."""
    geoms = lake_geoms()
    return sorted(geoms, key=lambda k: -geoms[k].area)


def folds() -> list[dict]:
    """
    Nested leave-one-lake-out.

    Outer: each lake is held out once and touched exactly once, at test time.
    Inner: one of the five remaining lakes is held out for early stopping, so
    the stopping decision never sees the test lake.

    Patches from one lake share illumination, atmosphere and - across years -
    the very same pixels, so a random patch split would put near-duplicates in
    train and test. Grouping by lake is the only honest split here, and it makes
    the effective sample size for cross-lake generalisation 6, not 42.
    """
    order = lake_order()
    n = len(order)
    out = []
    for i, test in enumerate(order):
        rest = [l for l in order if l != test]
        val = order[(i + 1) % n]
        if val == test:
            val = rest[0]
        out.append({"fold": i, "test": test, "val": val,
                    "train": [l for l in rest if l != val]})
    return out
