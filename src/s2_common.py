"""
s2_common.py
============
Shared Sentinel-2 scaling and index maths.

Raw digital numbers are what gets cached on disk; physical reflectance and every
spectral index are derived here, so a scaling correction never requires
re-downloading imagery.

Scaling
-------
    reflectance = DN * 0.0001          (no offset)

Earth Search serves the AWS `sentinel-2-l2a` archive reprocessed to processing
baseline 05.xx. Since baseline 04.00, ESA's native L2A product stores
reflectance as (DN - 1000) / 10000, i.e. with a -0.1 offset. Every item this
project uses carries `earthsearch:boa_offset_applied: true` in its STAC
properties: Earth Search has ALREADY subtracted that offset from the pixel
values before writing the COGs, so DN * 0.0001 is surface reflectance.

The same items still advertise `scale 0.0001, offset -0.1` under
`raster:bands`. Following that metadata applies the offset a second time. This
project did exactly that until September 2026, and the effect was severe:

  * over open water the green, NIR and SWIR values sit at 50-1200 DN, so a
    second -0.1 made nearly every water pixel's reflectance negative;
  * a normalised difference of two negative numbers flips sign, so water read
    as NDVI ~ +0.9 (dense vegetation) and MNDWI ~ -0.9 (dry land);
  * MNDWI > 0 found only 4-16% of the pixels ESA's own Scene Classification
    Layer marks as water. Without the double offset it finds 82-96%.

Checked on 22 September 2026 against the STAC items for 2019, 2021, 2023 and
2025 (all baseline 05.xx, all boa_offset_applied = true), and against the raw
DN over SCL-water pixels in every year. If a future scene is served with
boa_offset_applied = false, OFFSET must be -0.1 for that scene only.
"""

from __future__ import annotations

import numpy as np

SCALE = 0.0001
OFFSET = 0.0   # already applied by Earth Search - see module docstring
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


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """
    Normalised difference, kept inside its mathematical range [-1, 1].

    A normalised difference is bounded by +-1 only while both bands are
    non-negative. When reflectance goes negative the denominator can pass
    through zero and the ratio blows up. Under the old double offset (see the
    module docstring) that is what produced a scene-mean MNDWI of -61.6 for
    Bellandur 2025 and an NDWI of -111.3 for Sankey Tank 2023. With the offset
    fixed it is rare, but L2A reflectance can still dip just below zero over
    very dark targets, so the guard stays.

    Values are clipped here rather than at every call site. Where the
    denominator is within EPS of zero the index is genuinely undefined and is
    returned as 0.
    """
    den = np.asarray(den, dtype="float32")
    out = np.zeros_like(den, dtype="float32")
    ok = np.abs(den) > EPS
    np.divide(num, den, out=out, where=ok)
    return np.clip(out, -1.0, 1.0)


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
    b = reflectance(stack["blue"]) if "blue" in stack else None

    ndwi = _ratio(g - n, g + n)
    mndwi = _ratio(g - s, g + s)
    ndvi = _ratio(n - r, n + r) if r is not None else np.zeros_like(g)

    scl = stack["scl"] if "scl" in stack else None
    cloud = np.isin(scl, SCL_BAD) if scl is not None else np.zeros(g.shape, bool)

    out = {"green": g, "nir": n, "swir16": s, "ndwi": ndwi, "mndwi": mndwi,
           "ndvi": ndvi, "cloud": cloud,
           "water": (mndwi > MNDWI_WATER) & ~cloud,
           "veg": (ndvi > NDVI_VEG) & ~cloud}
    if r is not None:
        out["red"] = r
    if b is not None:
        out["blue"] = b
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
