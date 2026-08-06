"""
08_build_dashboard.py
=====================
Builds reports/dashboard.html - a single self-contained interactive dashboard.

Everything is embedded: the source catalogue, the lake time series, the weather
climatology, the endpoint probe, and base64 JPEG crops of the Bellandur-Varthur
corridor for each Sentinel-2 scene (true colour + MNDWI). No server, no network,
no build step - open the file and it works.
"""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import s2_common as s2c

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REP = ROOT / "reports"
S2DIR = RAW / "s2"
REP.mkdir(parents=True, exist_ok=True)

IMG_W = 900                     # crop width in px before JPEG encoding
JPEG_Q = 78


# --------------------------------------------------------------------------
# Imagery crops
# --------------------------------------------------------------------------
def grid():
    minx, miny, maxx, maxy, h, w = np.load(PROC / "grid_meta.npy")
    crs = (PROC / "grid_crs.txt").read_text(encoding="utf-8").strip()
    return (minx, miny, maxx, maxy), (int(h), int(w)), crs


def corridor_window(pad: float = 900.0):
    """Row/col slice covering Bellandur + Varthur, in the shared 20 m grid."""
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform

    (minx, miny, maxx, maxy), (h, w), crs = grid()
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    fc = json.loads((PROC / "lake_polygons.geojson").read_text(encoding="utf-8"))
    geoms = [shp_transform(tf, shape(f["geometry"])) for f in fc["features"]
             if f["properties"]["lake"] in ("Bellandur Lake", "Varthur Lake")]
    b = [g.bounds for g in geoms]
    x0, y0 = min(v[0] for v in b) - pad, min(v[1] for v in b) - pad
    x1, y1 = max(v[2] for v in b) + pad, max(v[3] for v in b) + pad
    rx, ry = (maxx - minx) / w, (maxy - miny) / h
    return (slice(int(max(0, (maxy - y1) / ry)), int(min(h, (maxy - y0) / ry))),
            slice(int(max(0, (x0 - minx) / rx)), int(min(w, (x1 - minx) / rx))))


def to_datauri(arr_rgb: np.ndarray) -> str:
    im = Image.fromarray(arr_rgb, mode="RGB")
    if im.width != IMG_W:
        im = im.resize((IMG_W, max(1, round(im.height * IMG_W / im.width))),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_Q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def diverging_lut() -> np.ndarray:
    """256-step blue<->red ramp with a neutral midpoint, matching the figures."""
    import matplotlib.pyplot as plt                              # noqa: F401
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("div", [
        "#0d366b", "#256abf", "#3987e5", "#86b6ef", "#f0efec",
        "#f0a0a0", "#e34948", "#d03b3b", "#8f2020"])
    return (np.asarray([cm(i / 255)[:3] for i in range(256)]) * 255).astype("uint8")


def build_images(years: list[int]) -> dict:
    win = corridor_window()
    lut = diverging_lut()
    out = {"rgb": [], "mndwi": []}
    for yr in years:
        with np.load(S2DIR / f"s2_{yr}.npz") as z:
            sub = {k: z[k][win] for k in z.files}
        out["rgb"].append(to_datauri((s2c.rgb(sub) * 255).astype("uint8")))
        m = s2c.indices(sub)["mndwi"]
        # MNDWI -0.8..0.8 -> reversed ramp so blue = water
        idx = np.clip((m + 0.8) / 1.6, 0, 1)
        out["mndwi"].append(to_datauri(lut[::-1][(idx * 255).astype("uint8")]))
        print(f"  {yr}  rgb {len(out['rgb'][-1])//1024:>4d} KB   "
              f"mndwi {len(out['mndwi'][-1])//1024:>4d} KB")
    return out


# --------------------------------------------------------------------------
# Tabular payload
# --------------------------------------------------------------------------
# This repository covers ONE project: Illegal Lake Encroachment Prediction.
# The catalogue built by 00_dataset_inventory.py is already scoped to it.
STUDY_CITY = "Bengaluru"


def build_data() -> dict:
    ds = pd.read_csv(PROC / "dataset_inventory.csv")
    health = pd.read_csv(PROC / "endpoint_health.csv")
    feats = pd.read_csv(PROC / "features_lake_year.csv")
    scenes = pd.read_csv(PROC / "s2_scenes.csv", parse_dates=["datetime"])
    wx = pd.read_csv(PROC / "weather_daily.csv", parse_dates=["date"])
    wx["PRECTOTCORR"] = pd.to_numeric(wx["PRECTOTCORR"], errors="coerce")

    datasets = [{
        "id": r.dataset_id, "name": r.dataset, "provider": r.provider,
        "modality": r.modality, "theme": r.theme, "access": r.access_tier,
        "res_m": None if pd.isna(r.res_m) else float(r.res_m),
        "y0": int(r.year_start), "y1": int(r.year_end), "cost": r.cost,
        "used": bool(r.used),
    } for r in ds.itertuples()]

    health = health[health.dataset_id.isin(set(ds.dataset_id))].copy()

    # lake series
    f = feats.sort_values(["lake", "year"])
    lake_area = [{
        "lake": r.lake, "year": int(r.year), "ref_ha": float(r.ref_ha),
        "water_ha": float(r.water_ha), "veg_ha": float(r.veg_ha),
        "other_ha": float(r.other_ha), "water_frac": float(r.water_frac),
        "mean_mndwi": float(r.mean_mndwi), "rain_365d": float(r.rain_365d),
        "scene_date": str(r.scene_date)[:10],
    } for r in f.itertuples()]

    last = f.sort_values("year").groupby("lake").last()
    lake_meta = {i: {"bldg_per_ha_250m": float(row.bldg_per_ha_250m),
                     "road_km_500m": float(row.road_km_500m),
                     "drain_km_500m": float(row.drain_km_500m)}
                 for i, row in last.iterrows()}
    lakes = (f.groupby("lake")["ref_ha"].first()
              .sort_values(ascending=False).index.tolist())

    # monthly rainfall for the study city
    b = wx[wx.city == STUDY_CITY].copy()
    b["y"], b["m"] = b.date.dt.year, b.date.dt.month
    clim = b.groupby(["y", "m"])["PRECTOTCORR"].sum().groupby("m").mean()
    climatology = [round(float(clim.loc[m]), 1) for m in range(1, 13)]

    h = health.merge(ds[["dataset_id", "dataset"]], on="dataset_id", how="left")
    health_rows = [{
        "name": (r.dataset if isinstance(r.dataset, str) else r.dataset_id),
        "ok": bool(r.reachable),
        "status": None if pd.isna(r.http_status) else int(r.http_status),
        "latency": float(r.latency_s), "url": r.url,
    } for r in h.itertuples()]

    scene_rows = [{"year": int(r.year), "date": f"{r.datetime:%d %b %Y}",
                   "cloud": round(float(r.cloud), 3)} for r in scenes.itertuples()]
    years = [int(y) for y in scenes.year]

    return {
        "datasets": datasets,
        "lakes": lakes, "lakeArea": lake_area, "lakeMeta": lake_meta,
        "city": STUDY_CITY, "climatology": climatology,
        "health": health_rows, "scenes": scene_rows,
        "meta": {"health_ok": int(health.reachable.sum()),
                 "health_n": int(len(health)),
                 "n_used": sum(1 for d in datasets if d["used"])},
    }, years


def clean(o):
    """NaN / inf are not valid JSON - carry them across as null."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.floating, np.integer)):
        v = o.item()
        return None if isinstance(v, float) and not math.isfinite(v) else v
    return o


if __name__ == "__main__":
    print("[1/3] tabular payload  (lake project only)")
    data, years = build_data()
    print(f"  {len(data['datasets'])} sources "
          f"({data['meta']['n_used']} actually used), "
          f"{len(data['lakeArea'])} lake-years, "
          f"{len(data['health'])} endpoints, city = {data['city']}")

    print("[2/3] imagery crops")
    data["images"] = build_images(years)

    print("[3/3] writing dashboard")
    tpl = (Path(__file__).parent / "dashboard_template.html").read_text(encoding="utf-8")
    payload = json.dumps(clean(data), separators=(",", ":"), allow_nan=False)
    out = REP / "dashboard.html"
    out.write_text(tpl.replace("/*__DATA__*/null", payload), encoding="utf-8")
    print(f"  {out.relative_to(ROOT)}   {out.stat().st_size/1e6:.2f} MB")
