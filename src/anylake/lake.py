"""
anylake/lake.py
===============
One lake on its own grid: geometry, fence rings, input channels, seed labels.

This is dl/common.py's lake_rasters() and dl/labels.py's seed_labels() with
the hard-coded Bengaluru grid replaced by the lake's own grid, so the same
rules hold everywhere:

  * region = lake footprint + 500 m collar; nothing outside it is labelled
  * fence rings at 30 m and 75 m from the outline
  * seed labels from SCL (6 water, 4 vegetation, 5 bare) and OpenStreetMap
    building centroids (built), never from the spectral indices the model reads

Input channels (16): the 12 Sentinel-2 bands plus four indices
    NDWI  (green - nir) / (green + nir)        water vs vegetation
    MNDWI (green - swir16) / (green + swir16)  water vs built-up
    NDVI  (nir - red) / (nir + red)            vegetation
    NDBI  (swir16 - nir) / (swir16 + nir)      built-up
"""

from __future__ import annotations

import json
import sys
from functools import cached_property

import numpy as np

from . import LAKES, YEARS
from .fetch import BANDS, transform_of

sys.path.insert(0, str(LAKES.parents[1] / "src"))
import s2_common as s2c  # noqa: E402
from dl import common as C  # noqa: E402
from dl.labels import SCL_TO_CLASS  # noqa: E402

INDICES = ["ndwi", "mndwi", "ndvi", "ndbi"]
CHANNELS = BANDS + INDICES
N_CHANNELS = len(CHANNELS)
WATER, VEG, BARE, BUILT, IGNORE = C.WATER, C.VEG, C.BARE, C.BUILT, C.IGNORE
CLASS_NAMES = C.CLASS_NAMES
FENCES_M = C.FENCES_M
COLLAR_M = C.COLLAR_M


def reflectance_bands(stack: dict) -> dict:
    return {b: s2c.reflectance(stack[b]) for b in BANDS}


def add_indices(refl: dict) -> np.ndarray:
    """(16, H, W) float32 from a dict of 12 reflectance bands."""
    g, r, n, s = refl["green"], refl["red"], refl["nir"], refl["swir16"]
    idx = {"ndwi": s2c._ratio(g - n, g + n), "mndwi": s2c._ratio(g - s, g + s),
           "ndvi": s2c._ratio(n - r, n + r), "ndbi": s2c._ratio(s - n, s + n)}
    return np.stack([refl[b] for b in BANDS] + [idx[k] for k in INDICES]).astype("float32")


class Lake:
    def __init__(self, slug: str):
        self.slug = slug
        self.dir = LAKES / slug
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        g = self.meta["grid"]
        self.crs, self.shape, self.res = g["crs"], tuple(g["shape"]), g["res"]
        self.bounds = g["bounds"]
        self.transform = transform_of(g)
        self.px_ha = self.res ** 2 / 1e4
        self.name = self.meta["name"]

    # --- geometry ---------------------------------------------------------
    @cached_property
    def geom(self):
        """Lake outline in the grid CRS."""
        from pyproj import Transformer
        from shapely.geometry import shape
        from shapely.ops import transform

        tf = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True).transform
        return transform(tf, shape(self.meta["geometry"]))

    def burn(self, geom) -> np.ndarray:
        from rasterio.features import geometry_mask
        from shapely.geometry import mapping

        if geom.is_empty:
            return np.zeros(self.shape, bool)
        return geometry_mask([mapping(geom)], out_shape=self.shape,
                             transform=self.transform, invert=True)

    @cached_property
    def masks(self) -> dict:
        inside = self.burn(self.geom)
        collar = self.burn(self.geom.buffer(COLLAR_M).difference(self.geom))
        out = {"inside": inside, "collar": collar, "region": inside | collar}
        for m in FENCES_M:
            out[f"ring_{m}m"] = self.burn(self.geom.buffer(m).difference(self.geom))
        return out

    def to_px(self, x, y):
        """Grid coordinates (col, row) as floats, for drawing."""
        return (x - self.bounds[0]) / self.res, (self.bounds[3] - y) / self.res

    # --- imagery ----------------------------------------------------------
    @cached_property
    def years(self) -> list[int]:
        return [y for y in YEARS if (self.dir / f"s2_{y}.npz").exists()]

    def stack(self, year: int) -> dict:
        with np.load(self.dir / f"s2_{year}.npz") as z:
            return {k: z[k] for k in z.files}

    def bad(self, stack: dict) -> np.ndarray:
        return np.isin(stack["scl"], s2c.SCL_BAD) | (stack["green"] == 0)

    def channels(self, year: int, variant: str = "raw"):
        """(16, H, W) channels and the bad-pixel mask. variant: raw | enh."""
        st = self.stack(year)
        refl = reflectance_bands(st)
        if variant == "enh":
            from .enhance import enhance_bands

            refl = enhance_bands(refl)
        return add_indices(refl), self.bad(st)

    # --- OpenStreetMap ----------------------------------------------------
    @cached_property
    def buildings_xy(self) -> np.ndarray:
        from pyproj import Transformer

        pts = np.load(self.dir / "osm_buildings.npy")
        if not len(pts):
            return np.zeros((0, 2))
        x, y = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True).transform(pts[:, 0], pts[:, 1])
        return np.column_stack([x, y])

    @cached_property
    def building_count(self) -> np.ndarray:
        counts = np.zeros(self.shape, "int32")
        if len(self.buildings_xy):
            col = np.floor((self.buildings_xy[:, 0] - self.bounds[0]) / self.res).astype(int)
            row = np.floor((self.bounds[3] - self.buildings_xy[:, 1]) / self.res).astype(int)
            ok = (col >= 0) & (col < self.shape[1]) & (row >= 0) & (row < self.shape[0])
            np.add.at(counts, (row[ok], col[ok]), 1)
        return counts

    @cached_property
    def road_mask(self) -> np.ndarray:
        from pyproj import Transformer
        from rasterio.features import rasterize
        from shapely.geometry import LineString

        roads = json.loads((self.dir / "osm_roads.json").read_text(encoding="utf-8"))
        if not roads:
            return np.zeros(self.shape, bool)
        tf = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        lines = []
        for r in roads:
            a = np.asarray(r)
            x, y = tf.transform(a[:, 0], a[:, 1])
            lines.append(LineString(np.column_stack([x, y])))
        return rasterize(((l, 1) for l in lines), out_shape=self.shape,
                         transform=self.transform, all_touched=True).astype(bool)

    # --- seed labels ------------------------------------------------------
    def labels(self, year: int) -> np.ndarray:
        """SCL + OpenStreetMap seed labels, cached as labels_<year>.npy."""
        path = self.dir / f"labels_{year}.npy"
        if path.exists():
            return np.load(path)
        st = self.stack(year)
        lab = np.full(self.shape, IGNORE, "uint8")
        for v, cls in SCL_TO_CLASS.items():
            lab[st["scl"] == v] = cls
        lab[self.building_count >= 1] = BUILT
        lab[self.bad(st)] = IGNORE
        lab[~self.masks["region"]] = IGNORE
        np.save(path, lab)
        return lab

    def window(self, margin: int = 6):
        reg = self.masks["region"]
        r = np.where(reg.any(1))[0]
        c = np.where(reg.any(0))[0]
        return (max(0, int(r[0]) - margin), min(self.shape[0], int(r[-1]) + 1 + margin),
                max(0, int(c[0]) - margin), min(self.shape[1], int(c[-1]) + 1 + margin))
