"""
s2_common.py
============
Shared Sentinel-2 scaling and index maths.

Raw digital numbers are what gets cached on disk; physical reflectance and every
spectral index are derived here, so a scaling correction never requires
re-downloading imagery.

Scaling
-------
Earth Search serves the AWS `sentinel-2-l2a` archive reprocessed to a uniform
baseline (05.xx as of this extract). Its `raster:bands` metadata declares

    scale = 0.0001,  offset = -0.1

i.e.  reflectance = DN * 0.0001 - 0.1

The offset (introduced with processing baseline 04.00) is *not* optional: over
dark targets like open water it shifts green/SWIR by +0.1 each, which collapses
MNDWI from roughly +0.78 to +0.14 and pushes most genuine water pixels below the
zero threshold. Because the archive is uniformly reprocessed, the same offset
applies to every year here - there is no 2022 discontinuity to correct for.
"""

from __future__ import annotations

import numpy as np

SCALE = 0.0001
OFFSET = -0.1
EPS = 1e-6

# Scene Classification Layer values treated as unusable
SCL_BAD = (0, 1, 3, 8, 9, 10)      # nodata, saturated, shadow, cloud med/high, cirrus
SCL_WATER = 6                       # SCL's own water class, used as a cross-check

# Index thresholds
MNDWI_WATER = 0.0                   # Xu (2006): open water > 0
NDVI_VEG = 0.30                     # dense green vegetation / hyacinth mat


def reflectance(dn: np.ndarray) -> np.ndarray:
    """Digital number -> surface reflectance."""
    return dn.astype("float32") * SCALE + OFFSET


def indices(stack) -> dict:
    """
    Derive reflectance bands, spectral indices and masks from a cached DN stack.

    Returns green/red/nir/swir16 reflectance plus:
      ndwi   McFeeters (1996)  (green - nir)  / (green + nir)    water vs vegetation
      mndwi  Xu (2006)         (green - swir) / (green + swir)   water vs built-up
      ndvi                     (nir - red)    / (nir + red)      vegetation vigour
      cloud  bool              unusable pixels from SCL
      water  bool              mndwi > 0 and not cloudy
      veg    bool              ndvi > 0.30 and not cloudy
    """
    g = reflectance(stack["green"])
    n = reflectance(stack["nir"])
    s = reflectance(stack["swir16"])
    r = reflectance(stack["red"]) if "red" in stack else None

    ndwi = (g - n) / (g + n + EPS)
    mndwi = (g - s) / (g + s + EPS)
    ndvi = ((n - r) / (n + r + EPS)) if r is not None else np.zeros_like(g)

    scl = stack["scl"] if "scl" in stack else None
    cloud = np.isin(scl, SCL_BAD) if scl is not None else np.zeros(g.shape, bool)

    out = {"green": g, "nir": n, "swir16": s, "ndwi": ndwi, "mndwi": mndwi,
           "ndvi": ndvi, "cloud": cloud,
           "water": (mndwi > MNDWI_WATER) & ~cloud,
           "veg": (ndvi > NDVI_VEG) & ~cloud}
    if r is not None:
        out["red"] = r
    if scl is not None:
        out["scl"] = scl
        out["scl_water"] = (scl == SCL_WATER)
    return out


def rgb(stack, pct: tuple[float, float] = (2.0, 98.0),
        gamma: float = 0.85) -> np.ndarray:
    """
    True-colour composite with a per-band percentile stretch.

    A fixed ceiling makes Indian urban scenes read as dark orange, because bare
    soil and terracotta roofs push the red channel far above vegetation and
    water. Stretching each band on its own percentiles restores a natural
    balance; the stretch is computed over whatever window is passed in.
    """
    chans = []
    for band in ("red", "green", "blue"):
        if band not in stack:
            raise KeyError(f"true-colour composite needs the '{band}' band")
        a = reflectance(stack[band])
        lo, hi = np.percentile(a[np.isfinite(a)], pct)
        chans.append(np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) ** gamma)
    return np.dstack(chans)


def panel_grid(ext, n: int, target_w: float = 15.0, ncol: int | None = None):
    """
    Choose a panel layout that matches the map window's aspect ratio.

    Without this a wide corridor laid out four-across produces short, stranded
    panels separated by dead space.
    """
    import math
    w, h = ext[1] - ext[0], ext[3] - ext[2]
    ar = w / h
    if ncol is None:
        ncol = 4 if ar < 1.25 else 3 if ar < 1.9 else 2 if ar < 3.6 else 1
    nrow = math.ceil(n / ncol)
    panel_w = target_w / ncol
    return ncol, nrow, (target_w, nrow * (panel_w / ar) + 1.15)
