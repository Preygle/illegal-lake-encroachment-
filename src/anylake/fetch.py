"""
anylake/fetch.py
================
Sentinel-2 imagery and OpenStreetMap context for one lake, on its own grid.

Dataset
-------
Earth Search `sentinel-2-l2a` (Element 84, AWS Open Data): ESA Copernicus
Sentinel-2 Level-2A surface reflectance as cloud-optimised GeoTIFFs. No login.

All 12 L2A bands are kept, because the user asked for every band:

    10 m  blue B02, green B03, red B04, nir B08
    20 m  rededge1-3 B05-B07, nir08 B8A, swir16 B11, swir22 B12
    60 m  coastal B01 (aerosol), nir09 B09 (water vapour)
    20 m  scl - ESA's Scene Classification Layer (cloud mask + label seeds)

Everything is warped onto one 20 m UTM grid around the lake (lake + 1 km),
the resolution the U-Net was designed for: 10 m bands are averaged down, 60 m
bands interpolated up (they add no 20 m detail, only their spectral value),
SCL is resampled nearest-neighbour so its class codes survive.

One scene per year, 2019-2025, from January-April (the dry season over most of
India, when clouds are fewest). Scenes are ranked by cloud over the lake itself
rather than over the whole 110 km tile, and two tiles from the same day are
mosaicked when the lake sits on a tile edge.

Scaling is DN x 0.0001. Earth Search has already removed the -0.1 offset that
processing baseline 04.00+ introduced (`earthsearch:boa_offset_applied`); a
scene where that flag is false gets the offset removed here, so every cached
array means the same thing. See s2_common.py for why this matters.
"""

from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from . import LAKES, RES, YEARS
from .search import UA, overpass

sys.path.insert(0, str(LAKES.parents[1] / "src"))
import s2_common as s2c  # noqa: E402

STAC = "https://earth-search.aws.element84.com/v1/search"
BANDS = ["coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3",
         "nir", "nir08", "nir09", "swir16", "swir22"]
NATIVE = {"coastal": 60, "nir09": 60, "blue": 10, "green": 10, "red": 10, "nir": 10}
BUFFER_M = 1000          # imagery around the lake: 500 m labelled collar + context
MAX_SIDE_M = 30000       # bigger lakes are analysed in a centred 30 km window
SEASON = ("01-01", "04-30")
MAX_DATES = 6            # candidate dates tried per year before taking the best seen

GDAL_ENV = dict(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")


# ==========================================================================
# Grid
# ==========================================================================
def utm_crs(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def make_grid(geom_ll) -> dict:
    """A 20 m UTM grid around the lake, snapped to whole pixels."""
    from pyproj import Transformer
    from shapely.ops import transform

    c = geom_ll.centroid
    crs = utm_crs(c.x, c.y)
    poly = transform(Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform, geom_ll)
    minx, miny, maxx, maxy = poly.bounds
    minx, miny, maxx, maxy = minx - BUFFER_M, miny - BUFFER_M, maxx + BUFFER_M, maxy + BUFFER_M
    clipped = False
    if maxx - minx > MAX_SIDE_M or maxy - miny > MAX_SIDE_M:
        cx, cy = poly.centroid.x, poly.centroid.y
        half = MAX_SIDE_M / 2
        if maxx - minx > MAX_SIDE_M:
            minx, maxx = cx - half, cx + half
        if maxy - miny > MAX_SIDE_M:
            miny, maxy = cy - half, cy + half
        clipped = True
    minx, miny = math.floor(minx / RES) * RES, math.floor(miny / RES) * RES
    maxx, maxy = math.ceil(maxx / RES) * RES, math.ceil(maxy / RES) * RES
    return {"crs": crs, "bounds": [minx, miny, maxx, maxy],
            "shape": [int((maxy - miny) / RES), int((maxx - minx) / RES)],
            "res": RES, "clipped": clipped}


def grid_bbox_ll(g: dict) -> list[float]:
    from rasterio.warp import transform_bounds

    return list(transform_bounds(g["crs"], "EPSG:4326", *g["bounds"], densify_pts=21))


def transform_of(g: dict):
    from rasterio.transform import from_bounds

    return from_bounds(*g["bounds"], g["shape"][1], g["shape"][0])


# ==========================================================================
# Reading COGs onto the grid
# ==========================================================================
def read_band(href: str, g: dict, band: str, tries: int = 4) -> np.ndarray:
    """
    One band warped onto the lake grid. WarpedVRT also handles scenes in a
    neighbouring UTM zone, which a plain windowed read would misplace.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    rs = (Resampling.nearest if band == "scl" else
          Resampling.average if NATIVE.get(band) == 10 else
          Resampling.bilinear)
    last = None
    for i in range(tries):
        try:
            with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src, \
                    WarpedVRT(src, crs=g["crs"], transform=transform_of(g),
                              width=g["shape"][1], height=g["shape"][0],
                              resampling=rs, src_nodata=0, nodata=0) as vrt:
                return vrt.read(1)
        except Exception as e:                                    # noqa: BLE001
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"read failed after {tries} tries ({band}): {last}")


def mosaic(items: list[dict], g: dict, band: str) -> np.ndarray:
    """Fill the grid from each item in turn; later items only fill gaps."""
    out = None
    for it in items:
        a = read_band(it["assets"][band]["href"], g, band)
        if not it["properties"].get("earthsearch:boa_offset_applied", True) and band != "scl":
            a = np.where(a > 0, np.clip(a.astype("int32") - 1000, 1, None), 0).astype(a.dtype)
        if out is None:
            out = a
        else:
            gap = out == 0
            out[gap] = a[gap]
        if not (out == 0).any():
            break
    return out


# ==========================================================================
# Scene search
# ==========================================================================
def stac_items(bbox_ll, year: int) -> list[dict]:
    body = {"collections": ["sentinel-2-l2a"], "bbox": bbox_ll,
            "datetime": f"{year}-{SEASON[0]}T00:00:00Z/{year}-{SEASON[1]}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": 40}}, "limit": 200}
    for attempt in range(3):
        try:
            r = requests.post(STAC, json=body, headers=UA, timeout=90)
            r.raise_for_status()
            break
        except Exception:                                         # noqa: BLE001
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    feats = r.json().get("features", [])
    # The archive holds some scenes twice (processing versions _0_ and _1_);
    # keep the later version of each.
    best = {}
    for f in feats:
        key = f["id"].rsplit("_", 2)[0]
        if key not in best or f["id"] > best[key]["id"]:
            best[key] = f
    return list(best.values())


def lake_mask(geom_ll, g: dict) -> np.ndarray:
    """Lake + 500 m on the grid, where cloud matters."""
    from pyproj import Transformer
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping
    from shapely.ops import transform

    poly = transform(Transformer.from_crs("EPSG:4326", g["crs"], always_xy=True).transform, geom_ll)
    return geometry_mask([mapping(poly.buffer(500))], out_shape=tuple(g["shape"]),
                         transform=transform_of(g), invert=True)


def pick_scene(geom_ll, g: dict, year: int, log=print):
    """
    The clearest date for this lake: SCL is read for each candidate date (cheap,
    one band) and scored on cloud and coverage inside lake + 500 m.
    """
    items = stac_items(grid_bbox_ll(g), year)
    if not items:
        return None
    by_date: dict[str, list[dict]] = {}
    for it in items:
        by_date.setdefault(it["properties"]["datetime"][:10], []).append(it)
    dates = sorted(by_date, key=lambda d: np.mean([i["properties"]["eo:cloud_cover"]
                                                   for i in by_date[d]]))
    roi = lake_mask(geom_ll, g)
    best = None
    for d in dates[:MAX_DATES]:
        its = sorted(by_date[d], key=lambda i: i["properties"]["eo:cloud_cover"])
        scl = mosaic(its, g, "scl")
        cover = float((scl[roi] != 0).mean())
        bad = float(np.isin(scl[roi], s2c.SCL_BAD).mean())
        score = bad + (1 - cover)
        if best is None or score < best[0]:
            best = (score, d, its, scl, cover, bad)
        if cover > 0.99 and bad < 0.03:
            break
    score, d, its, scl, cover, bad = best
    log(f"    {year}: {d}  cloud/shadow over lake {bad:5.1%}  coverage {cover:5.1%}"
        f"  ({len(its)} tile{'s' if len(its) > 1 else ''})")
    return {"date": d, "items": its, "scl": scl, "cover": cover, "bad": bad}


# ==========================================================================
# One lake
# ==========================================================================
def lake_dir(slug: str):
    d = LAKES / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def register(rec: dict) -> dict:
    """Write meta.json for a lake chosen from search results (or the library)."""
    from shapely.geometry import shape

    d = lake_dir(rec["slug"])
    path = d / "meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    geom = shape(rec["geometry"])
    meta = {k: rec.get(k) for k in ("name", "display", "district", "state", "lat", "lon",
                                     "osm_type", "osm_id", "area_ha", "slug")}
    meta.update(geometry=rec["geometry"], grid=make_grid(geom), scenes={},
                library=bool(rec.get("library", False)))
    path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def save_meta(meta: dict) -> None:
    (lake_dir(meta["slug"]) / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def fetch_imagery(meta: dict, log=print) -> dict:
    """Seven years of all-band imagery. Cached per year; returns per-year timings."""
    from shapely.geometry import shape

    d = lake_dir(meta["slug"])
    g = meta["grid"]
    geom = shape(meta["geometry"])
    for year in YEARS:
        out = d / f"s2_{year}.npz"
        if out.exists() and str(year) in meta["scenes"]:
            continue
        t0 = time.time()
        sc = pick_scene(geom, g, year, log)
        if sc is None:
            log(f"    {year}: no Sentinel-2 scene found - year skipped")
            continue
        arrays = {"scl": sc["scl"].astype("uint8")}
        with ThreadPoolExecutor(max_workers=len(BANDS)) as ex:
            futs = {b: ex.submit(mosaic, sc["items"], g, b) for b in BANDS}
            for b, f in futs.items():
                arrays[b] = f.result().astype("uint16")
        np.savez_compressed(out, **arrays)
        meta["scenes"][str(year)] = {
            "date": sc["date"], "ids": [i["id"] for i in sc["items"]],
            "tile_cloud_pct": [round(i["properties"]["eo:cloud_cover"], 2) for i in sc["items"]],
            "lake_cloud_frac": round(sc["bad"], 4), "coverage": round(sc["cover"], 4),
            "seconds": round(time.time() - t0, 1)}
        save_meta(meta)
        log(f"    {year}: 12 bands + SCL saved ({time.time() - t0:.0f}s)")
    return meta


def fetch_osm(meta: dict, log=print) -> None:
    """OpenStreetMap building centroids and roads around the lake (Overpass)."""
    d = lake_dir(meta["slug"])
    if (d / "osm_buildings.npy").exists() and (d / "osm_roads.json").exists():
        return
    w, s, e, n = grid_bbox_ll(meta["grid"])
    bb = f"{s:.5f},{w:.5f},{n:.5f},{e:.5f}"
    try:
        b = overpass(f'[out:json][timeout:240];way["building"]({bb});out center;')
        pts = np.array([[el["center"]["lon"], el["center"]["lat"]]
                        for el in b.get("elements", []) if "center" in el], "float64").reshape(-1, 2)
    except Exception as ex:                                       # noqa: BLE001
        log(f"    OSM buildings unavailable ({ex}); the built-up class will have no seed labels")
        pts = np.zeros((0, 2))
    np.save(d / "osm_buildings.npy", pts)
    try:
        r = overpass(f'[out:json][timeout:180];way["highway"~"^(motorway|trunk|primary|'
                     f'secondary|tertiary|residential|unclassified)$"]({bb});out geom;')
        roads = [[[v["lon"], v["lat"]] for v in el["geometry"]]
                 for el in r.get("elements", []) if len(el.get("geometry", [])) >= 2]
    except Exception as ex:                                       # noqa: BLE001
        log(f"    OSM roads unavailable ({ex})")
        roads = []
    (d / "osm_roads.json").write_text(json.dumps(roads), encoding="utf-8")
    log(f"    OSM: {len(pts):,} buildings, {len(roads):,} road segments")


def osm_from_local(meta: dict) -> None:
    """
    Library lakes (the six Bengaluru lakes) reuse the OpenStreetMap dump the
    original pipeline already downloaded (01_fetch_nonsatellite.py), clipped to
    the lake's grid, so the library never depends on Overpass being up.
    """
    d = lake_dir(meta["slug"])
    root = LAKES.parents[1]
    w, s, e, n = grid_bbox_ll(meta["grid"])
    pts = np.load(root / "data" / "processed" / "osm_building_pts.npy")
    keep = (pts[:, 0] >= w) & (pts[:, 0] <= e) & (pts[:, 1] >= s) & (pts[:, 1] <= n)
    np.save(d / "osm_buildings.npy", pts[keep])
    raw = json.loads((root / "data" / "raw" / "osm_roads.json").read_text(encoding="utf-8"))
    roads = []
    for el in raw.get("elements", []):
        g = el.get("geometry") or []
        line = [[v["lon"], v["lat"]] for v in g]
        if len(line) >= 2 and any(w <= x <= e and s <= y <= n for x, y in line):
            roads.append(line)
    (d / "osm_roads.json").write_text(json.dumps(roads), encoding="utf-8")


def fetch(meta: dict, log=print) -> dict:
    d = lake_dir(meta["slug"])
    if meta.get("library") and not (d / "osm_buildings.npy").exists():
        osm_from_local(meta)
    if meta["grid"]["clipped"]:
        log(f"    lake is larger than {MAX_SIDE_M / 1000:.0f} km: analysing a centred "
            f"{MAX_SIDE_M / 1000:.0f} km window")
    fetch_osm(meta, log)
    return fetch_imagery(meta, log)
