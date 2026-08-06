"""
04_eda_nonsatellite.py
======================
Exploratory analysis of the NON-SATELLITE layers.

F08  rainfall seasonality heatmap, Bengaluru        (NASA POWER, ground-equivalent)
F10  daily rainfall + the Sentinel-2 acquisition dates actually used
F11  OpenStreetMap layer inventory over the lake belt
F12  building-footprint density with lake outlines
F13  built-up pressure by distance band from each lake shoreline
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import viz_style as vs

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

AOI = (77.555, 12.905, 77.755, 13.055)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

wx = pd.read_csv(PROC / "weather_daily.csv", parse_dates=["date"])
wx["PRECTOTCORR"] = pd.to_numeric(wx["PRECTOTCORR"], errors="coerce")
lakes = pd.read_csv(PROC / "lakes.csv")

SRC_POWER = "Source: NASA POWER daily (MERRA-2 / GPM-IMERG), 2018-01-01 to 2025-12-31"
SRC_OSM = "Source: OpenStreetMap via Overpass API, extracted 2026-08-03 (ODbL)"


# --------------------------------------------------------------------------
# OSM loaders (cached to .npy so the 114 MB JSON is parsed only once)
# --------------------------------------------------------------------------
CENTER_RE = re.compile(
    r'"center":\s*\{\s*"lat":\s*(-?[\d.]+),\s*"lon":\s*(-?[\d.]+)')
LATLON_RE = re.compile(r'"lat":\s*(-?[\d.]+),\s*"lon":\s*(-?[\d.]+)')


def building_points() -> np.ndarray:
    cache = PROC / "osm_building_pts.npy"
    if cache.exists():
        return np.load(cache)
    txt = (RAW / "osm_buildings.json").read_text(encoding="utf-8")
    pts = np.array([(float(lo), float(la))
                    for la, lo in CENTER_RE.findall(txt)], dtype="float32")
    np.save(cache, pts)
    return pts


def way_vertices(name: str) -> np.ndarray:
    """All vertices of a `out geom` layer, as (lon, lat)."""
    cache = PROC / f"osm_{name}_pts.npy"
    if cache.exists():
        return np.load(cache)
    txt = (RAW / f"osm_{name}.json").read_text(encoding="utf-8")
    pts = np.array([(float(lo), float(la))
                    for la, lo in LATLON_RE.findall(txt)], dtype="float32")
    np.save(cache, pts)
    return pts


def load_lake_polys():
    p = PROC / "lake_polygons.geojson"
    if not p.exists():
        return {}
    from shapely.geometry import shape
    fc = json.loads(p.read_text(encoding="utf-8"))
    return {f["properties"]["lake"]: shape(f["geometry"]) for f in fc["features"]}


# ============================== F08 =======================================
def f08_seasonality(t: vs.Theme):
    b = wx[wx.city == "Bengaluru"].copy()
    b["year"], b["month"] = b.date.dt.year, b.date.dt.month
    piv = (b.groupby(["year", "month"])["PRECTOTCORR"].sum()
             .unstack().reindex(columns=range(1, 13)))

    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    im = ax.imshow(piv.values, aspect="auto", cmap=t.seq_cmap,
                   origin="upper", vmin=0, vmax=np.nanmax(piv.values))
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index)
    ax.grid(False)

    vmax = np.nanmax(piv.values)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color=("white" if v > vmax * 0.55 else t.ink2))

    # the dry window the satellite scenes come from
    ax.add_patch(Rectangle((-0.5, -0.5), 4, len(piv), fill=False,
                           edgecolor=t.color(1), lw=2.2, zorder=5))
    ax.text(1.5, -0.72, "Jan–Apr dry window used for Sentinel-2",
            ha="center", fontsize=9, color=t.color(1), fontweight="600")

    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.036)
    cb.set_label("Monthly rainfall (mm)", color=t.ink2, fontsize=9.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=t.muted, labelcolor=t.ink2, labelsize=8.5)

    vs.title(ax, "Bengaluru rainfall seasonality, 2018–2025",
             "Two wet peaks (SW monsoon Jun–Sep, NE monsoon Oct–Nov) bracket a reliable Jan–Apr dry window.", t)
    vs.source(fig, SRC_POWER, t)
    return vs.save(fig, "F08_rainfall_seasonality", FIG, t)


# ============================== F10 =======================================
def f10_daily_with_scenes(t: vs.Theme):
    b = wx[wx.city == "Bengaluru"].set_index("date")["PRECTOTCORR"]
    fig, ax = plt.subplots(figsize=(13.4, 5.4))
    ax.fill_between(b.index, b.values, color=t.color(0), lw=0, alpha=0.9, zorder=2)

    scenes_p = PROC / "s2_scenes.csv"
    if scenes_p.exists():
        sc = pd.read_csv(scenes_p, parse_dates=["datetime"])
        for i, r in enumerate(sc.itertuples()):
            ax.axvline(r.datetime, color=t.color(1), lw=1.6, ls=(0, (3, 2)),
                       zorder=4, alpha=0.95)
            ax.annotate(f"{r.datetime:%Y-%m-%d}", (r.datetime, b.max() * 0.99),
                        rotation=90, fontsize=8, color=t.color(1), ha="right",
                        va="top", fontweight="600")
        ax.plot([], [], color=t.color(1), ls=(0, (3, 2)), lw=1.6,
                label="Sentinel-2 scene used")
    ax.plot([], [], color=t.color(0), lw=6, label="Daily rainfall")
    ax.legend(loc="upper left", ncol=2)

    ax.set_ylabel("Daily rainfall (mm)")
    ax.set_xlabel("")
    ax.set_ylim(0, b.max() * 1.10)
    ax.grid(axis="x", visible=False)
    vs.title(ax, "Every satellite scene was captured in a rain-free window",
             "Bengaluru daily rainfall with the seven Sentinel-2 acquisition dates overlaid — "
             "this is what makes the annual water-area comparison fair.", t)
    vs.source(fig, SRC_POWER + " | Scene dates: Earth Search STAC (Sentinel-2 L2A)", t)
    return vs.save(fig, "F10_rainfall_with_scene_dates", FIG, t)


# ============================== F11 =======================================
def f11_osm_inventory(t: vs.Theme):
    rows = []
    for f, label in [("osm_water", "Waterbodies"), ("osm_roads", "Road segments"),
                     ("osm_buildings", "Building footprints"),
                     ("osm_drains", "Drains / streams")]:
        p = RAW / f"{f}.json"
        if not p.exists():
            continue
        n = len(json.loads(p.read_text(encoding="utf-8")).get("elements", []))
        rows.append({"layer": label, "n": n, "mb": p.stat().st_size / 1e6})
    inv = pd.DataFrame(rows).sort_values("n")
    inv.to_csv(TAB / "osm_layer_inventory.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.0, 4.9))
    y = np.arange(len(inv))
    ax.barh(y, inv.n, color=t.modality("Non-satellite"), height=0.6, zorder=2)
    for i, r in enumerate(inv.itertuples()):
        ax.text(r.n * 1.06, i, f"{r.n:,}   ({r.mb:.0f} MB)", va="center",
                fontsize=9.5, color=t.ink2, fontweight="600")
    ax.set_yticks(y)
    ax.set_yticklabels(inv.layer, fontsize=10.5)
    ax.set_xscale("log")
    ax.set_xlim(inv.n.min() * 0.55, inv.n.max() * 4.2)
    ax.set_xlabel("Elements retrieved (log scale)")
    ax.grid(axis="y", visible=False)
    vs.title(ax, "OpenStreetMap extract over the Bengaluru lake belt",
             "A single 22 × 17 km bounding box, four Overpass queries — no licence or login required.", t)
    vs.source(fig, SRC_OSM, t)
    return vs.save(fig, "F11_osm_inventory", FIG, t)


# ============================== F12 =======================================
def f12_building_density(t: vs.Theme):
    pts = building_points()
    m = ((pts[:, 0] >= AOI[0]) & (pts[:, 0] <= AOI[2]) &
         (pts[:, 1] >= AOI[1]) & (pts[:, 1] <= AOI[3]))
    pts = pts[m]

    nx, ny = 110, 92
    H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=[nx, ny],
                               range=[[AOI[0], AOI[2]], [AOI[1], AOI[3]]])

    fig, ax = plt.subplots(figsize=(11.4, 8.4))
    im = ax.imshow(H.T, origin="lower", cmap=t.seq_cmap, aspect="auto",
                   extent=(AOI[0], AOI[2], AOI[1], AOI[3]),
                   vmin=0, vmax=np.percentile(H[H > 0], 99))

    polys = load_lake_polys()
    for name, poly in polys.items():
        gs = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        for g in gs:
            xs, ys = g.exterior.xy
            ax.plot(xs, ys, color=t.color(1), lw=2.0, zorder=4)
        c = poly.centroid
        ax.annotate(name.replace(" Lake", "").replace(" Tank", ""), (c.x, c.y),
                    fontsize=9.5, color="white", fontweight="600", ha="center",
                    zorder=5,
                    bbox=dict(boxstyle="round,pad=0.26", fc=t.color(1), ec="none"))

    cb = fig.colorbar(im, ax=ax, pad=0.014, fraction=0.036)
    cb.set_label("Building footprints per ~200 m cell", color=t.ink2, fontsize=9.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=t.muted, labelcolor=t.ink2, labelsize=8.5)

    ax.set_xlim(AOI[0], AOI[2])
    ax.set_ylim(AOI[1], AOI[3])
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(False)
    ax.set_title(f"Urban fabric around the study lakes — {len(pts):,} building footprints",
                 loc="left", pad=26)
    ax.text(0.0, 1.012,
            "Lake outlines in orange. Dense fabric pressing directly against a shoreline "
            "is the encroachment signal.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, color=t.muted)
    vs.source(fig, SRC_OSM, t)
    return vs.save(fig, "F12_building_density", FIG, t)


# ============================== F13 =======================================
def f13_pressure(t: vs.Theme):
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    polys = load_lake_polys()
    if not polys:
        raise RuntimeError("lake_polygons.geojson missing - run 02_fetch_satellite.py")

    pts = building_points()
    bands_m = [0, 100, 250, 500, 1000]
    labels = ["0–100 m", "100–250 m", "250–500 m", "500–1000 m"]
    deg = 1.0 / 111_320.0

    rows = []
    for name, poly in polys.items():
        minx, miny, maxx, maxy = poly.buffer(1000 * deg).bounds
        sel = pts[(pts[:, 0] >= minx) & (pts[:, 0] <= maxx) &
                  (pts[:, 1] >= miny) & (pts[:, 1] <= maxy)]
        if len(sel) == 0:
            continue
        tree = STRtree([Point(xy) for xy in sel])
        prev = poly
        for lo, hi, lab in zip(bands_m[:-1], bands_m[1:], labels):
            ring = poly.buffer(hi * deg).difference(poly.buffer(lo * deg))
            idx = tree.query(ring, predicate="intersects")
            area_ha = ring.area * (111_320 ** 2) * np.cos(np.radians(12.97)) / 1e4
            rows.append({"lake": name, "band": lab, "n": int(len(idx)),
                         "area_ha": area_ha,
                         "per_ha": len(idx) / max(area_ha, 1e-6)})
        _ = prev
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "building_pressure_by_band.csv", index=False)

    piv = df.pivot(index="lake", columns="band", values="per_ha")[labels]
    piv = piv.reindex(piv[labels[0]].sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    x = np.arange(len(piv))
    w = 0.20
    for i, lab in enumerate(labels):
        off = (i - 1.5) * (w + 0.012)
        ax.bar(x + off, piv[lab], w, color=t.color(i),
               label=lab, zorder=2)
    for i, lab in enumerate(labels):
        off = (i - 1.5) * (w + 0.012)
        for xi, v in enumerate(piv[lab].values):
            if np.isfinite(v):
                ax.text(xi + off, v + piv.values.max() * 0.018, f"{v:.0f}",
                        ha="center", fontsize=7.8, color=t.ink2)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" Lake", "").replace(" Tank", "") for n in piv.index],
                       fontsize=10.5)
    ax.set_ylabel("Buildings per hectare")
    ax.legend(title="Distance from shoreline", ncol=4, loc="upper right")
    ax.grid(axis="x", visible=False)
    vs.title(ax, "Built-up pressure by distance from each lake shoreline",
             "Buildings per hectare in concentric bands. A high innermost bar means development has reached the water's edge.", t)
    vs.source(fig, SRC_OSM + " | Lake outlines: OSM multipolygon relations", t)
    return vs.save(fig, "F13_encroachment_pressure", FIG, t)


if __name__ == "__main__":
    print("  parsing OSM caches ...")
    print(f"    buildings: {len(building_points()):,} centroids")
    for mode in ("light", "dark"):
        t = vs.apply(mode)
        for fn in (f08_seasonality, f10_daily_with_scenes,
                   f11_osm_inventory, f12_building_density, f13_pressure):
            try:
                p = fn(t)
                print(f"  [{mode}] {p.name}")
            except Exception as e:                                # noqa: BLE001
                print(f"  [{mode}] {fn.__name__} FAILED: {type(e).__name__}: {e}")
    print("Done.")
