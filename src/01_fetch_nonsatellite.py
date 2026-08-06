"""
01_fetch_nonsatellite.py
========================
Pulls the NON-SATELLITE evidence layers used in the EDA:

  * OpenStreetMap (Overpass API)  - lake polygons, roads, buildings, drains
  * NASA POWER                    - daily rainfall / temperature per study city
  * Live endpoint health check    - proves which catalogued sources are reachable

Everything is cached under data/raw/ so re-runs are cheap.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
for d in (RAW, PROC):
    d.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "PredictiveDA-EDA/1.0 (university project; contact via github)"}
OVERPASS = "https://overpass-api.de/api/interpreter"

lakes = pd.read_csv(PROC / "lakes.csv")
cities = pd.read_csv(PROC / "study_area.csv")      # Bengaluru only

# Bengaluru lake-belt bounding box (covers all six study lakes)
BBOX = (12.905, 77.555, 13.055, 77.755)  # S, W, N, E  (Overpass order)


# ==========================================================================
# 1. OpenStreetMap via Overpass
# ==========================================================================
def overpass(query: str, cache: Path, tries: int = 3) -> dict:
    """Run an Overpass query with on-disk caching and retry/backoff."""
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    last = None
    for i in range(tries):
        try:
            r = requests.post(OVERPASS, data={"data": query}, headers=UA, timeout=180)
            if r.status_code == 200:
                cache.write_text(r.text, encoding="utf-8")
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as e:                                    # noqa: BLE001
            last = f"{type(e).__name__}"
        wait = 10 * (i + 1)
        print(f"    retry {i+1}/{tries} after {last}; sleeping {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"Overpass failed: {last}")


def fetch_osm() -> None:
    s, w, n, e = BBOX
    bb = f"{s},{w},{n},{e}"

    jobs = {
        # water bodies as full geometry (polygons)
        "osm_water": f"""[out:json][timeout:180];
            (way["natural"="water"]({bb});
             relation["natural"="water"]({bb});
             way["landuse"="reservoir"]({bb}););
            out geom;""",
        # road network (centrelines only - keep payload small)
        "osm_roads": f"""[out:json][timeout:180];
            way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential)$"]({bb});
            out geom;""",
        # buildings -> urban pressure proxy (centroids via 'out center')
        "osm_buildings": f"""[out:json][timeout:240];
            way["building"]({bb});
            out center;""",
        # drains and channels feeding / draining the lakes
        "osm_drains": f"""[out:json][timeout:180];
            way["waterway"~"^(drain|ditch|stream|canal)$"]({bb});
            out geom;""",
    }

    for name, q in jobs.items():
        cache = RAW / f"{name}.json"
        t0 = time.time()
        try:
            d = overpass(q, cache)
            print(f"  {name:16s} {len(d.get('elements', [])):>7,d} elements "
                  f"({time.time()-t0:5.1f}s){'  [cached]' if time.time()-t0 < 0.5 else ''}")
        except Exception as e:                                    # noqa: BLE001
            print(f"  {name:16s} FAILED: {e}")


# ==========================================================================
# 2. NASA POWER daily meteorology
# ==========================================================================
def fetch_power(lat: float, lon: float, tag: str,
                start: str = "20180101", end: str = "20251231") -> pd.DataFrame | None:
    """Daily rainfall / temperature / humidity / wind for one point."""
    cache = RAW / f"power_{tag}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])

    url = ("https://power.larc.nasa.gov/api/temporal/daily/point"
           "?parameters=PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M,WS10M"
           f"&community=AG&longitude={lon}&latitude={lat}"
           f"&start={start}&end={end}&format=JSON")
    try:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        p = r.json()["properties"]["parameter"]
        df = pd.DataFrame(p)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df = df.rename_axis("date").reset_index()
        df = df.replace(-999.0, pd.NA)
        df["city"] = tag
        df.to_csv(cache, index=False)
        return df
    except Exception as e:                                        # noqa: BLE001
        print(f"  POWER {tag:12s} FAILED: {type(e).__name__}: {e}")
        return None


def fetch_all_power() -> None:
    frames = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fetch_power, r.lat, r.lon, r.city): r.city
                for r in cities.itertuples()}
        for f in as_completed(futs):
            df = f.result()
            if df is not None:
                frames.append(df)
                print(f"  POWER {futs[f]:12s} {len(df):,d} days  "
                      f"{df.date.min():%Y-%m-%d} -> {df.date.max():%Y-%m-%d}")
    if frames:
        out = pd.concat(frames, ignore_index=True).sort_values(["city", "date"])
        out.to_csv(PROC / "weather_daily.csv", index=False)
        print(f"  -> weather_daily.csv  {len(out):,d} rows, {out.city.nunique()} cities")


# ==========================================================================
# 3. Live endpoint health check (evidence for the provenance chart)
# ==========================================================================
PROBES = [
    ("POWER",     "https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR"
                  "&community=AG&longitude=77.6&latitude=12.9&start=20240101&end=20240102&format=JSON"),
    ("OSM",       "https://overpass-api.de/api/status"),
    ("GIBS",      "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/"
                  "MODIS_Terra_CorrectedReflectance_TrueColor/default/2024-01-15/250m/6/13/45.jpg"),
    ("S2",        "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a"),
    ("OPENMETEO", "https://archive-api.open-meteo.com/v1/archive?latitude=12.97&longitude=77.59"
                  "&start_date=2024-01-01&end_date=2024-01-02&daily=precipitation_sum"),
    ("LANDSAT",   "https://landsatlook.usgs.gov/stac-server/"),
    ("GHSL",      "https://ghsl.jrc.ec.europa.eu/download.php"),
    ("WORLDPOP",  "https://www.worldpop.org"),
    ("DATAGOVIN", "https://data.gov.in"),
    ("CENSUS",    "https://censusindia.gov.in"),
    ("IMD",       "https://mausam.imd.gov.in"),
    ("BHUVAN",    "https://bhuvan.nrsc.gov.in"),
    ("NWA",       "https://vedas.sac.gov.in"),
]


def probe_one(item):
    ds_id, url = item
    t0 = time.time()
    try:
        r = requests.get(url, timeout=30, headers=UA, allow_redirects=True)
        return {"dataset_id": ds_id, "url": url, "http_status": r.status_code,
                "reachable": bool(r.status_code < 400), "latency_s": round(time.time() - t0, 2),
                "bytes": len(r.content)}
    except Exception as e:                                        # noqa: BLE001
        return {"dataset_id": ds_id, "url": url, "http_status": None,
                "reachable": False, "latency_s": round(time.time() - t0, 2),
                "bytes": 0, "error": type(e).__name__}


def health_check() -> None:
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(probe_one, PROBES))
    df = pd.DataFrame(rows).sort_values("dataset_id")
    df.to_csv(PROC / "endpoint_health.csv", index=False)
    ok = int(df.reachable.sum())
    print(f"  {ok}/{len(df)} endpoints reachable")
    for r in df.itertuples():
        flag = "OK " if r.reachable else "DOWN"
        print(f"    {flag} {r.dataset_id:11s} {str(r.http_status):>5s}  {r.latency_s:5.2f}s")


# ==========================================================================
if __name__ == "__main__":
    print("[1/3] OpenStreetMap via Overpass")
    fetch_osm()
    print("\n[2/3] NASA POWER daily meteorology")
    fetch_all_power()
    print("\n[3/3] Endpoint health check")
    health_check()
    print("\nDone.")
