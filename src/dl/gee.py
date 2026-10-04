"""
dl/gee.py
=========
Google Earth Engine access for the three layers that only exist there:

  * GOOGLE/Research/open-buildings-temporal/v1  - Google Open Buildings
    Temporal: yearly building presence, height and fractional count, 2016-2023,
    derived from Sentinel-2. Used for a per-year building count inside each
    fence - the only genuinely time-varying building signal in the project.
  * GOOGLE/DYNAMICWORLD/V1                     - Google Dynamic World: a
    9-class land-cover map for every Sentinel-2 scene. Used as an independent
    cross-check on the segmentation, on the exact seven scene dates.
  * JRC/GSW1_4/GlobalSurfaceWater, JRC/GSW1_4/YearlyHistory - JRC Global
    Surface Water (EC Joint Research Centre / Google), from Landsat 1984-2021.
    A different sensor and algorithm, used as a cross-check on open water.

Setup mirrors the satellite-prediction project (src/urbanintel/data/gee.py):
credentials come from `earthengine authenticate`, stored once per user under
~/.config/earthengine, and the Google Cloud project is `satellite-505815`
unless EE_PROJECT is set. `ee.Initialize()` only reads stored credentials; it
never opens a browser. If they are missing the error says what to run.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import common as C

DEFAULT_PROJECT = os.environ.get("EE_PROJECT", "satellite-505815")

OPEN_BUILDINGS = "GOOGLE/Research/open-buildings-temporal/v1"
DYNAMIC_WORLD = "GOOGLE/DYNAMICWORLD/V1"
GSW = "JRC/GSW1_4/GlobalSurfaceWater"
GSW_YEARLY = "JRC/GSW1_4/YearlyHistory"

_EE = None


class GEEUnavailable(RuntimeError):
    """Earth Engine is not installed or not authenticated."""


def ee_init(project: str | None = None):
    global _EE
    if _EE is not None:
        return _EE
    try:
        import ee
    except ImportError as exc:
        raise GEEUnavailable("pip install earthengine-api") from exc
    try:
        ee.Initialize(project=project or DEFAULT_PROJECT)
    except Exception as exc:                                      # noqa: BLE001
        raise GEEUnavailable(
            "Earth Engine is not authenticated.\n"
            "  run once in a terminal:  earthengine authenticate\n"
            "  and set EE_PROJECT if your Cloud project is not "
            f"{DEFAULT_PROJECT}\n  underlying error: {exc}") from exc
    _EE = ee
    return ee


def ee_geom(shapely_geom):
    """A shapely geometry in the grid CRS as an Earth Engine geometry."""
    from shapely.geometry import mapping

    ee = ee_init()
    _, _, crs, _ = C.grid()
    return ee.Geometry(mapping(shapely_geom), crs, False)


def grid_params() -> dict:
    """
    Download parameters that land an Earth Engine image on *exactly* the
    project's 20 m grid - same CRS, same origin, same pixel count - so it can be
    compared pixel for pixel with the segmentation without any resampling here.
    """
    bounds, shape, crs, _ = C.grid()
    res = (bounds[2] - bounds[0]) / shape[1]
    return {"crs": crs,
            "crs_transform": [res, 0, bounds[0], 0, -res, bounds[3]],
            "dimensions": f"{shape[1]}x{shape[0]}",
            "format": "GEO_TIFF"}


def download(img, dest: Path, force: bool = False) -> np.ndarray:
    """Fetch an image onto the grid as a GeoTIFF and return its first band."""
    import rasterio
    import requests

    dest = Path(dest)
    if not dest.exists() or force:
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = img.getDownloadURL(grid_params())
        r = requests.get(url, stream=True, timeout=600)
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
    with rasterio.open(dest) as src:
        arr = src.read()
    _, shape, _, _ = C.grid()
    if arr.shape[1:] != tuple(shape):
        raise RuntimeError(f"{dest.name}: got {arr.shape[1:]}, grid is {shape}")
    return arr


# ==========================================================================
# Open Buildings Temporal
# ==========================================================================
def open_buildings_years() -> dict[int, object]:
    """One mosaic per inference year, keyed by calendar year."""
    import datetime as dt

    ee = ee_init()
    col = ee.ImageCollection(OPEN_BUILDINGS)
    epochs = col.aggregate_array("inference_time_epoch_s").distinct().getInfo()
    out = {}
    for e in sorted(epochs):
        year = dt.datetime.fromtimestamp(int(e), dt.timezone.utc).year
        out[year] = col.filter(ee.Filter.eq("inference_time_epoch_s", e)).mosaic()
    return out


def open_buildings_scale() -> float:
    ee = ee_init()
    return float(ee.ImageCollection(OPEN_BUILDINGS).first()
                 .projection().nominalScale().getInfo())


# ==========================================================================
# Dynamic World
# ==========================================================================
DW_CLASSES = ["water", "trees", "grass", "flooded_vegetation", "crops",
              "shrub_and_scrub", "built", "bare", "snow_and_ice"]

# Dynamic World -> the project's four classes. Flooded vegetation is mapped to
# vegetation because inside a lake footprint that is what water hyacinth is; it
# is also counted on its own, since it is the closest ready-made proxy for
# hyacinth anywhere in the public catalogue.
DW_TO_PROJECT = {0: C.WATER, 1: C.VEG, 2: C.VEG, 3: C.VEG, 4: C.VEG,
                 5: C.VEG, 6: C.BUILT, 7: C.BARE, 8: C.IGNORE}


def dynamic_world_for(date: str, tile: str):
    """
    The Dynamic World image computed from the same Sentinel-2 scene.

    Dynamic World indexes each image by its source Sentinel-2 product ID, so
    the tile code in the index identifies the exact scene rather than just the
    day. Returns None if that scene has no Dynamic World counterpart.
    """
    import datetime as dt

    ee = ee_init()
    d0 = dt.date.fromisoformat(date)
    bounds, _, crs, _ = C.grid()
    region = ee.Geometry.Rectangle(list(bounds), crs, False)
    col = (ee.ImageCollection(DYNAMIC_WORLD)
           .filterDate(str(d0), str(d0 + dt.timedelta(days=1)))
           .filterBounds(region))
    ids = col.aggregate_array("system:index").getInfo()
    match = [i for i in ids if tile.replace("MGRS-", "T") in i]
    if not match:
        return None, ids
    return ee.Image(f"{DYNAMIC_WORLD}/{match[0]}"), match[0]
