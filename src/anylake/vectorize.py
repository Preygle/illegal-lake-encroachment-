"""
anylake/vectorize.py
====================
Image -> vector, in both of the senses the term is used:

1. Feature vectors (machine learning). Every pixel in the lake region becomes
   one row: its 12 band reflectances, the 4 spectral indices, a little local
   texture (3 x 3 mean and standard deviation of NIR and SWIR), its position
   and its class. The table is what the encroachment forecaster learns from,
   and the per-class mean of it is the spectral signature chart.

2. Polygons (GIS). Class maps are rasters - a grid of pixels. Turning them
   into vector polygons (water body outline, built-up patches, growth, the
   predicted encroachment) gives shapes with an area and a boundary that open
   in QGIS or Google Earth. Done with rasterio.features.shapes, which traces
   the outline of every connected group of pixels, then lightly simplified.
"""

from __future__ import annotations

import json

import numpy as np

from .lake import CHANNELS, CLASS_NAMES, IGNORE, Lake

TEXTURE_BANDS = ("nir", "swir16")


def texture(chan: np.ndarray) -> dict:
    import cv2

    out = {}
    for b in TEXTURE_BANDS:
        a = chan[CHANNELS.index(b)].astype("float32")
        m = cv2.blur(a, (3, 3))
        sq = cv2.blur(a * a, (3, 3))
        out[f"{b}_mean3"] = m
        out[f"{b}_std3"] = np.sqrt(np.maximum(sq - m * m, 0))
    return out


def feature_table(lake: Lake, year: int, classmap: np.ndarray | None = None):
    """One row per region pixel: position, 16 channels, texture, class."""
    import pandas as pd
    from pyproj import Transformer

    chan, bad = lake.channels(year, "raw")
    reg = lake.masks["region"] & ~bad
    rows, cols = np.nonzero(reg)
    x = lake.bounds[0] + (cols + 0.5) * lake.res
    y = lake.bounds[3] - (rows + 0.5) * lake.res
    lon, lat = Transformer.from_crs(lake.crs, "EPSG:4326", always_xy=True).transform(x, y)
    d = {"row": rows, "col": cols, "lon": np.round(lon, 6), "lat": np.round(lat, 6),
         "inside_lake": lake.masks["inside"][rows, cols]}
    for i, c in enumerate(CHANNELS):
        d[c] = np.round(chan[i][rows, cols], 4)
    for k, a in texture(chan).items():
        d[k] = np.round(a[rows, cols], 4)
    if classmap is not None:
        cls = classmap[rows, cols]
        d["unet_class"] = np.where(cls == IGNORE, "none",
                                   np.array(CLASS_NAMES + ["none"])[np.minimum(cls, 4)])
    return pd.DataFrame(d)


def signatures(table) -> "pd.DataFrame":
    """Mean reflectance of each band per U-Net class: the spectral signature."""
    from .fetch import BANDS

    t = table[table.unet_class != "none"]
    return t.groupby("unet_class")[BANDS].mean()


def signature_chart(sig, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .fetch import BANDS

    wl = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]   # band centres, nm
    names = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
    colors = {"water": "#2166ac", "vegetation": "#1b7837", "bare": "#c98a1e", "built": "#c82828"}
    x = np.arange(len(BANDS))
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=130)
    for cls in sig.index:
        ax.plot(x, sig.loc[cls, BANDS].values, "-o", ms=3.5, lw=2, color=colors.get(cls, "#555"),
                label=cls.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n{w}" for n, w in zip(names, wl)], fontsize=7)
    ax.set_xlabel("Sentinel-2 band and centre wavelength (nm)", fontsize=8)
    ax.set_ylabel("Surface reflectance", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ==========================================================================
# Raster -> polygons
# ==========================================================================
def polygons(mask: np.ndarray, lake: Lake, min_px: int = 2, simplify_m: float | None = None):
    """
    Shapely polygons (grid CRS) tracing every connected patch of `mask`.
    Patches smaller than `min_px` pixels are dropped as speckle.
    """
    from rasterio.features import shapes
    from shapely.geometry import shape

    out = []
    if not mask.any():
        return out
    tol = lake.res / 2 if simplify_m is None else simplify_m
    for geom, val in shapes(mask.astype("uint8"), mask=mask, transform=lake.transform,
                            connectivity=8):
        p = shape(geom)
        if p.area < min_px * lake.res ** 2:
            continue
        out.append(p.simplify(tol, preserve_topology=True) if tol else p)
    return out


def to_geojson(polys, lake: Lake, props: dict | None = None, path=None) -> dict:
    """FeatureCollection in lon/lat (EPSG:4326), with each polygon's area."""
    from pyproj import Transformer
    from shapely.geometry import mapping
    from shapely.ops import transform

    tf = Transformer.from_crs(lake.crs, "EPSG:4326", always_xy=True).transform
    feats = []
    for i, p in enumerate(polys):
        pr = {"id": i, "area_ha": round(p.area / 1e4, 3), **(props or {})}
        feats.append({"type": "Feature", "properties": pr, "geometry": mapping(transform(tf, p))})
    fc = {"type": "FeatureCollection", "features": feats}
    if path is not None:
        path.write_text(json.dumps(fc), encoding="utf-8")
    return fc
