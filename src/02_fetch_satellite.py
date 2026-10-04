"""
02_fetch_satellite.py
=====================
Pulls the SATELLITE evidence layers:

  * Sentinel-2 L2A (Earth Search STAC -> free AWS COGs)
        one low-cloud dry-season scene per year, 2019-2025, over the
        Bengaluru lake belt. Bands: green, red, NIR, SWIR16, SCL.
        -> NDWI, MNDWI, NDVI, cloud mask, per-lake water area time series
  * NASA GIBS WMS
        MODIS true colour + VIIRS night lights + MODIS LST for regional context

Outputs
-------
data/raw/s2/s2_<year>.npz            stacked 20 m arrays per year
data/raw/gibs/*.png                  regional context rasters
data/processed/s2_scenes.csv         which scene was used for which year
data/processed/lake_water_area.csv   per-lake water area per year  <- the label signal
data/processed/lake_polygons.geojson reference lake boundaries (OSM)
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, Point, mapping

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
S2DIR, GIBSDIR = RAW / "s2", RAW / "gibs"
for d in (S2DIR, GIBSDIR, PROC):
    d.mkdir(parents=True, exist_ok=True)

STAC = "https://earth-search.aws.element84.com/v1/search"
# lat_min was 12.905 until September 2026, which left 34% of Madiwala Lake
# (one of its two polygon parts entirely) below the grid. 12.890 covers every
# lake plus its 500 m collar. The grid's top-left corner is unchanged, so the
# existing pixels of the other five lakes keep exactly the same positions; the
# grid only gains rows to the south and one column to the east.
AOI = (77.555, 12.890, 77.755, 13.055)          # lon_min, lat_min, lon_max, lat_max
YEARS = list(range(2019, 2026))
TARGET_RES = 20.0                                # metres
# swir22 (B12) is not used by the four-class segmentation model, which reads
# blue/green/red/nir/swir16 plus the three indices. It is listed so that a future
# re-fetch picks it up: the pretrained surface-water networks (DeepWaterMap,
# WatNet) and several Sentinel-2 foundation encoders require B12 and cannot run
# without it. Existing caches stay valid - fetch_year() only re-downloads when
# the cached schema is stale, so nothing is re-fetched until that happens.
BANDS = ["blue", "green", "red", "nir", "swir16", "swir22", "scl"]

GDAL_ENV = dict(AWS_NO_SIGN_REQUEST="YES",
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")

lakes_df = pd.read_csv(PROC / "lakes.csv")


# ==========================================================================
# 1. Reference lake polygons from OpenStreetMap (fallback: circular buffer)
# ==========================================================================
def _osm_candidates() -> list[tuple[str, Polygon]]:
    """
    Every named waterbody in the Overpass dump as a shapely polygon.

    Simple ways carry `geometry` directly. The big Bengaluru lakes are
    multipolygon *relations* whose outer ring arrives as several disjoint way
    fragments, so those are stitched with linemerge + polygonize.
    """
    from shapely.geometry import LineString
    from shapely.ops import linemerge, polygonize, unary_union

    out: list[tuple[str, Polygon]] = []
    path = RAW / "osm_water.json"
    if not path.exists():
        return out

    for el in json.loads(path.read_text(encoding="utf-8")).get("elements", []):
        name = el.get("tags", {}).get("name", "")
        try:
            if el["type"] == "way":
                g = el.get("geometry")
                if not g or len(g) < 4:
                    continue
                p = Polygon([(v["lon"], v["lat"]) for v in g])
                if p.is_valid and p.area > 0:
                    out.append((name, p))

            elif el["type"] == "relation":
                def rings(role):
                    ls = [LineString([(v["lon"], v["lat"]) for v in m["geometry"]])
                          for m in el.get("members", [])
                          if m.get("role") == role and len(m.get("geometry", [])) >= 2]
                    if not ls:
                        return []
                    built = [p for p in polygonize(linemerge(ls))
                             if p.is_valid and p.area > 0]
                    if not built:                   # ring left open -> force-close it
                        merged = linemerge(ls)
                        parts = ([merged] if merged.geom_type == "LineString"
                                 else list(merged.geoms))
                        built = [Polygon(l.coords) for l in parts if len(l.coords) >= 4]
                        built = [p for p in built if p.is_valid and p.area > 0]
                    return built

                outer = rings("outer")
                if not outer:
                    continue
                # A lake relation is frequently several disjoint basins - Bellandur
                # is four. Keeping only the largest would drop two-thirds of it, so
                # the whole multipolygon is retained and islands are cut out.
                geom = unary_union(outer)
                holes = rings("inner")
                if holes:
                    geom = geom.difference(unary_union(holes))
                if not geom.is_empty and geom.area > 0:
                    out.append((name, geom))
        except Exception:                                         # noqa: BLE001
            continue
    return out


def build_lake_polygons() -> dict[str, Polygon]:
    """Match each study lake to its OSM polygon by exact name; circle as fallback."""
    candidates = _osm_candidates()
    by_name: dict[str, list[Polygon]] = {}
    for nm, p in candidates:
        if nm:
            by_name.setdefault(nm.strip().lower(), []).append(p)

    polys: dict[str, Polygon] = {}
    for r in lakes_df.itertuples():
        pt = Point(r.lon, r.lat)
        hit, src = None, "buffer"
        named = by_name.get(str(r.osm_name).strip().lower(), [])
        if named:
            hit, src = max(named, key=lambda p: p.area), "osm"
        else:                                        # nearest polygon containing the point
            near = [p for _, p in candidates if p.contains(pt)]
            if near:
                hit, src = max(near, key=lambda p: p.area), "osm-geo"
        if hit is None:
            radius_deg = math.sqrt(r.nominal_area_ha * 10_000 / math.pi) / 111_320.0
            hit = pt.buffer(radius_deg, 48)

        ha = hit.area * (111_320 ** 2) * math.cos(math.radians(r.lat)) / 10_000
        parts = 1 if hit.geom_type == "Polygon" else len(hit.geoms)
        polys[r.lake] = hit
        print(f"  {r.lake:16s} <- {src:7s} basins={parts:2d} "
              f"area={ha:7.1f} ha  (published ~{r.nominal_area_ha} ha)")
    return polys


def save_polygons(polys: dict[str, Polygon]) -> None:
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"lake": k}, "geometry": mapping(v)}
        for k, v in polys.items()]}
    (PROC / "lake_polygons.geojson").write_text(json.dumps(fc), encoding="utf-8")


# ==========================================================================
# 2. Scene selection - one clear dry-season scene per year, same MGRS tile
# ==========================================================================
def search_scenes() -> pd.DataFrame:
    cache = PROC / "s2_scenes.csv"
    if cache.exists():
        return pd.read_csv(cache)

    rows = []
    for yr in YEARS:
        body = {"collections": ["sentinel-2-l2a"], "bbox": list(AOI),
                "datetime": f"{yr}-01-01T00:00:00Z/{yr}-04-20T23:59:59Z",
                "query": {"eo:cloud_cover": {"lt": 25}}, "limit": 100}
        r = requests.post(STAC, json=body, timeout=90)
        r.raise_for_status()
        for f in r.json().get("features", []):
            p = f["properties"]
            rows.append({"year": yr, "id": f["id"],
                         "datetime": p["datetime"][:10],
                         "cloud": p.get("eo:cloud_cover", 100),
                         "tile": p.get("grid:code", ""),
                         **{b: f["assets"][b]["href"] for b in BANDS if b in f["assets"]},
                         "visual": f["assets"].get("visual", {}).get("href", "")})
    all_scenes = pd.DataFrame(rows)
    if all_scenes.empty:
        raise RuntimeError("STAC returned no scenes")

    # keep the single MGRS tile that covers the AOI in the most years
    tile = (all_scenes.groupby("tile")["year"].nunique().sort_values().index[-1])
    print(f"  tile coverage: {all_scenes.groupby('tile')['year'].nunique().to_dict()}"
          f"  -> using {tile}")
    sub = all_scenes[all_scenes.tile == tile]
    best = (sub.sort_values(["year", "cloud"])
               .groupby("year", as_index=False).first())
    best.to_csv(cache, index=False)
    return best


# ==========================================================================
# 3. Windowed COG reads onto a common 20 m grid
# ==========================================================================
def target_grid(sample_href: str):
    """Common UTM grid derived from the AOI, shared by every year."""
    with rasterio.Env(**GDAL_ENV), rasterio.open(sample_href) as src:
        crs = src.crs
    b = transform_bounds("EPSG:4326", crs, *AOI, densify_pts=21)
    minx = math.floor(b[0] / TARGET_RES) * TARGET_RES
    miny = math.floor(b[1] / TARGET_RES) * TARGET_RES
    maxx = math.ceil(b[2] / TARGET_RES) * TARGET_RES
    maxy = math.ceil(b[3] / TARGET_RES) * TARGET_RES
    w = int((maxx - minx) / TARGET_RES)
    h = int((maxy - miny) / TARGET_RES)
    return crs, (minx, miny, maxx, maxy), (h, w)


def read_band(href: str, bounds, shape, tries: int = 4) -> np.ndarray:
    """Read `href` over `bounds`, decimated/upsampled to `shape`, with retries."""
    last = None
    for i in range(tries):
        try:
            with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
                win = from_bounds(*bounds, src.transform)
                return src.read(1, window=win, out_shape=shape,
                                boundless=True, fill_value=0)
        except Exception as e:                                    # noqa: BLE001
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"read failed after {tries} tries: {last}")


def fetch_year(row, bounds, shape) -> tuple[int, dict] | None:
    """
    Cache the *raw digital numbers* for one year.

    Reflectance scaling and every spectral index are derived later in
    s2_common.indices(), so a scaling fix never costs another download.
    """
    out = S2DIR / f"s2_{int(row.year)}.npz"
    if out.exists():
        with np.load(out) as z:
            if ("green" in z and z["green"].dtype == np.uint16    # current schema
                    and z["green"].shape == tuple(shape)):         # current grid
                print(f"  {int(row.year)}  cached")
                return int(row.year), {k: z[k] for k in z.files}
        print(f"  {int(row.year)}  stale cache (old schema or grid) - refetching")

    t0 = time.time()
    arrays = {}
    try:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(read_band, getattr(row, b), bounds, shape): b
                    for b in BANDS if isinstance(getattr(row, b, None), str)}
            for f in as_completed(futs):
                arrays[futs[f]] = f.result()
    except Exception as e:                                        # noqa: BLE001
        print(f"  {int(row.year)}  FAILED: {type(e).__name__}: {e}")
        return None

    missing = {"green", "nir", "swir16"} - arrays.keys()
    if missing:
        print(f"  {int(row.year)}  missing bands {missing}")
        return None

    payload = {k: (v.astype("uint8") if k == "scl" else v.astype("uint16"))
               for k, v in arrays.items()}
    np.savez_compressed(out, **payload)
    print(f"  {int(row.year)}  {row.datetime}  cloud={row.cloud:5.2f}%  "
          f"{shape}  {len(payload)} bands  ({time.time()-t0:5.1f}s)")
    return int(row.year), payload


# ==========================================================================
# 4. Per-lake water area from the MNDWI mask
# ==========================================================================
def lake_masks(polys, crs, bounds, shape):
    """Rasterise each lake footprint and its 500 m collar onto the target grid."""
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds as t_from_bounds
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    transform = t_from_bounds(*bounds, shape[1], shape[0])
    out = {}
    for lake, poly_ll in polys.items():
        poly = shp_transform(tf, poly_ll)
        inside = geometry_mask([mapping(poly)], out_shape=shape,
                               transform=transform, invert=True)
        ring = poly.buffer(500).difference(poly)
        collar = geometry_mask([mapping(ring)], out_shape=shape,
                               transform=transform, invert=True)
        out[lake] = {"inside": inside, "collar": collar, "geom": poly}
    return out, transform


def lake_areas(stacks: dict[int, dict], masks) -> pd.DataFrame:
    """
    Per lake, per year: how much of the reference footprint is open water,
    how much is vegetation (hyacinth mat / encroached bed), how much is neither.

    Splitting water from vegetation matters here: Bellandur and Varthur are
    chronically covered by water hyacinth, which reads as vegetation, not water.
    Reporting only 'water area' would misattribute that to lake loss.
    """
    import s2_common as s2c

    px_ha = (TARGET_RES ** 2) / 10_000.0
    rows = []
    for yr, st in sorted(stacks.items()):
        ix = s2c.indices(st)
        for lake, m in masks.items():
            inside, collar = m["inside"], m["collar"]
            n_ref = int(inside.sum())
            if n_ref == 0:
                continue
            valid = inside & ~ix["cloud"]
            if valid.sum() < 0.5 * n_ref:
                continue
            water = ix["water"] & valid
            veg = ix["veg"] & valid & ~water
            other = valid & ~water & ~veg
            collar_v = collar & ~ix["cloud"]
            rows.append({
                "lake": lake, "year": yr,
                "ref_ha": round(n_ref * px_ha, 2),
                "water_ha": round(int(water.sum()) * px_ha, 2),
                "veg_ha": round(int(veg.sum()) * px_ha, 2),
                "other_ha": round(int(other.sum()) * px_ha, 2),
                "water_frac": round(float(water.sum() / valid.sum()), 4),
                "veg_frac": round(float(veg.sum() / valid.sum()), 4),
                "wet_frac": round(float((water.sum() + veg.sum()) / valid.sum()), 4),
                "valid_frac": round(float(valid.sum() / n_ref), 3),
                "mean_mndwi": round(float(ix["mndwi"][valid].mean()), 4),
                "mean_ndwi": round(float(ix["ndwi"][valid].mean()), 4),
                "mean_ndvi": round(float(ix["ndvi"][valid].mean()), 4),
                "scl_water_ha": round(int((ix["scl_water"] & valid).sum()) * px_ha, 2)
                if "scl_water" in ix else None,
                "collar_veg_frac": round(
                    float((ix["veg"] & collar_v).sum() / max(collar_v.sum(), 1)), 4),
                "collar_water_frac": round(
                    float((ix["water"] & collar_v).sum() / max(collar_v.sum(), 1)), 4),
            })
    return pd.DataFrame(rows).sort_values(["lake", "year"])


# ==========================================================================
# 5. NASA GIBS regional context via WMS
# ==========================================================================
def fetch_gibs() -> None:
    # WMS 1.3.0 + EPSG:4326 => BBOX is lat_min,lon_min,lat_max,lon_max
    # Only the true-colour layers are relevant here - they place the Bengaluru
    # lake belt in its regional context. (Night lights and land-surface
    # temperature belonged to other candidate projects and are not fetched.)
    jobs = [
        ("modis_truecolor_karnataka", "MODIS_Terra_CorrectedReflectance_TrueColor",
         (11.5, 74.0, 18.5, 81.0), "2024-02-15", 1200, 1200),
        ("modis_truecolor_india", "MODIS_Terra_CorrectedReflectance_TrueColor",
         (6.0, 67.0, 37.0, 98.0), "2024-02-15", 1240, 1240),
    ]
    base = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    for name, layer, bbox, date, w, h in jobs:
        out = GIBSDIR / f"{name}.png"
        if out.exists():
            print(f"  {name:30s} cached")
            continue
        params = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
                  "LAYERS": layer, "CRS": "EPSG:4326",
                  "BBOX": ",".join(str(v) for v in bbox),
                  "WIDTH": w, "HEIGHT": h, "FORMAT": "image/png", "TIME": date}
        try:
            r = requests.get(base, params=params, timeout=120)
            if r.ok and r.content[:4] == b"\x89PNG":
                out.write_bytes(r.content)
                print(f"  {name:30s} {len(r.content):>9,d}B")
            else:
                print(f"  {name:30s} bad response {r.status_code} "
                      f"{r.content[:120]!r}")
        except Exception as e:                                    # noqa: BLE001
            print(f"  {name:30s} FAILED {type(e).__name__}")


# ==========================================================================
if __name__ == "__main__":
    print("[1/5] Reference lake polygons")
    polys = build_lake_polygons()
    save_polygons(polys)

    print("\n[2/5] Sentinel-2 scene search")
    scenes = search_scenes()
    print(scenes[["year", "datetime", "cloud", "tile"]].to_string(index=False))

    print("\n[3/5] Common target grid")
    crs, bounds, shape = target_grid(scenes.iloc[0]["green"])
    print(f"  CRS={crs}  shape={shape}  res={TARGET_RES}m")

    print("\n[4/5] Sentinel-2 band reads")
    stacks = {}
    for row in scenes.itertuples():
        got = fetch_year(row, bounds, shape)
        if got:
            stacks[got[0]] = got[1]

    if stacks:
        masks, _ = lake_masks(polys, crs, bounds, shape)
        df = lake_areas(stacks, masks)
        df.to_csv(PROC / "lake_water_area.csv", index=False)
        print(f"\n  lake_water_area.csv  {len(df)} rows")
        print("\n  open water (ha)")
        print(df.pivot_table(index="lake", columns="year",
                             values="water_ha").to_string())
        print("\n  vegetation inside footprint (ha)")
        print(df.pivot_table(index="lake", columns="year",
                             values="veg_ha").to_string())
        np.save(PROC / "grid_meta.npy",
                np.array([*bounds, shape[0], shape[1]], dtype="float64"))
        Path(PROC / "grid_crs.txt").write_text(str(crs), encoding="utf-8")

    print("\n[5/5] NASA GIBS regional context")
    fetch_gibs()
    print("\nDone.")
