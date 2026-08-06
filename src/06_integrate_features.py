"""
06_integrate_features.py
========================
Joins the satellite and non-satellite evidence into one analysis-ready table and
explores the relationships between them.

Builds  data/processed/features_lake_year.csv   (6 lakes x 7 years)
   satellite     : water_ha, veg_ha, other_ha, water_frac, mean MNDWI/NDWI/NDVI,
                   collar water & vegetation fractions
   non-satellite : antecedent rainfall (30/90/365 d), mean max temperature,
                   building counts and road/drain length in distance bands

F23  antecedent rainfall vs open-water area - the confound that must be controlled
F24  built-up pressure vs change in open water
F25  correlation structure of the assembled feature table
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import viz_style as vs

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

DEG = 1.0 / 111_320.0

area = pd.read_csv(PROC / "lake_water_area.csv")
scenes = pd.read_csv(PROC / "s2_scenes.csv", parse_dates=["datetime"])
wx = pd.read_csv(PROC / "weather_daily.csv", parse_dates=["date"])
wx["PRECTOTCORR"] = pd.to_numeric(wx["PRECTOTCORR"], errors="coerce")
wx["T2M_MAX"] = pd.to_numeric(wx["T2M_MAX"], errors="coerce")
lakes = pd.read_csv(PROC / "lakes.csv")


def lake_polys():
    from shapely.geometry import shape
    fc = json.loads((PROC / "lake_polygons.geojson").read_text(encoding="utf-8"))
    return {f["properties"]["lake"]: shape(f["geometry"]) for f in fc["features"]}


# --------------------------------------------------------------------------
# 1. Non-satellite features
# --------------------------------------------------------------------------
def weather_features() -> pd.DataFrame:
    """Antecedent rainfall / temperature at Bengaluru for each scene date."""
    b = wx[wx.city == "Bengaluru"].set_index("date").sort_index()
    rows = []
    for r in scenes.itertuples():
        d = r.datetime
        rows.append({
            "year": int(r.year),
            "scene_date": d,
            "rain_30d": float(b.loc[d - pd.Timedelta(days=30):d, "PRECTOTCORR"].sum()),
            "rain_90d": float(b.loc[d - pd.Timedelta(days=90):d, "PRECTOTCORR"].sum()),
            "rain_365d": float(b.loc[d - pd.Timedelta(days=365):d, "PRECTOTCORR"].sum()),
            "rain_dry_since_oct": float(
                b.loc[pd.Timestamp(year=int(r.year) - 1, month=10, day=1):d,
                      "PRECTOTCORR"].sum()),
            "tmax_30d": float(b.loc[d - pd.Timedelta(days=30):d, "T2M_MAX"].mean()),
        })
    return pd.DataFrame(rows)


def osm_features() -> pd.DataFrame:
    """Static built-environment pressure around each lake (single OSM snapshot)."""
    from shapely.geometry import Point, LineString
    from shapely.strtree import STRtree

    polys = lake_polys()
    bpts = np.load(PROC / "osm_building_pts.npy")
    btree = STRtree([Point(xy) for xy in bpts])

    def line_km(path: str, geom):
        """Total length of an OSM `out geom` layer inside `geom`, in km."""
        data = json.loads((RAW / path).read_text(encoding="utf-8"))
        segs = []
        for el in data.get("elements", []):
            g = el.get("geometry")
            if g and len(g) >= 2:
                segs.append(LineString([(v["lon"], v["lat"]) for v in g]))
        if not segs:
            return 0.0
        tree = STRtree(segs)
        idx = tree.query(geom, predicate="intersects")
        tot = 0.0
        for i in idx:
            tot += segs[i].intersection(geom).length
        return tot * 111.32 * np.cos(np.radians(12.97))

    rows = []
    for name, poly in polys.items():
        rec = {"lake": name}
        for band in (100, 250, 500):
            ring = poly.buffer(band * DEG)
            n = len(btree.query(ring, predicate="intersects"))
            ha = ring.difference(poly).area * (111_320 ** 2) * np.cos(np.radians(12.97)) / 1e4
            rec[f"bldg_{band}m"] = int(n)
            rec[f"bldg_per_ha_{band}m"] = round(n / max(ha, 1e-6), 3)
        buf500 = poly.buffer(500 * DEG)
        rec["road_km_500m"] = round(line_km("osm_roads.json", buf500), 2)
        rec["drain_km_500m"] = round(line_km("osm_drains.json", buf500), 2)
        rows.append(rec)
    return pd.DataFrame(rows)


def build_features() -> pd.DataFrame:
    f = (area.merge(weather_features(), on="year", how="left")
             .merge(osm_features(), on="lake", how="left")
             .merge(lakes[["lake", "lat", "lon", "nominal_area_ha"]], on="lake",
                    how="left"))
    # per-lake change relative to the first observed year
    f = f.sort_values(["lake", "year"])
    base = f.groupby("lake")["water_ha"].transform("first")
    f["water_ha_delta"] = (f.water_ha - base).round(2)
    f["water_ha_pct"] = np.where(base > 0, (f.water_ha / base - 1) * 100, np.nan).round(1)
    f.to_csv(PROC / "features_lake_year.csv", index=False)
    f.to_csv(TAB / "features_lake_year.csv", index=False)
    return f


# ============================== F23 =======================================
def f23_rain_vs_water(t: vs.Theme):
    f = pd.read_csv(PROC / "features_lake_year.csv")
    order = f.groupby("lake")["water_ha"].mean().sort_values(ascending=False).index

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.4))
    for ax, lake in zip(axes.ravel(), order):
        s = f[f.lake == lake]
        sc = ax.scatter(s.rain_365d, s.water_ha, s=95, c=s.year,
                        cmap=t.seq_cmap, edgecolor=t.surface, linewidth=1.5,
                        zorder=3)
        for r in s.itertuples():
            ax.annotate(str(int(r.year)), (r.rain_365d, r.water_ha),
                        xytext=(0, 10), textcoords="offset points", ha="center",
                        fontsize=8, color=t.ink2)
        if len(s) > 2 and s.water_ha.std() > 0 and s.rain_365d.std() > 0:
            r_ = float(np.corrcoef(s.rain_365d, s.water_ha)[0, 1])
            # only draw a trend line where there is a trend worth drawing -
            # a fitted line through r ~ 0 implies a relationship that is not there
            if abs(r_) >= 0.40:
                m, b = np.polyfit(s.rain_365d, s.water_ha, 1)
                xs = np.linspace(s.rain_365d.min(), s.rain_365d.max(), 20)
                ax.plot(xs, m * xs + b, color=t.color(1), lw=1.8, ls=(0, (4, 2)),
                        zorder=2)
            ax.text(0.03, 0.95,
                    f"r = {r_:+.2f}" + ("" if abs(r_) >= 0.40 else "   (no trend)"),
                    transform=ax.transAxes, va="top", fontsize=9.5,
                    color=(t.ink2 if abs(r_) >= 0.40 else t.muted), fontweight="700")
        ax.set_title(lake, fontsize=11)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Open water (ha)")
    for ax in axes[1, :]:
        ax.set_xlabel("Rainfall in preceding 365 days (mm)")

    fig.suptitle("Rainfall explains one of these lakes, and none of the others",
                 x=0.005, y=1.005, ha="left", fontsize=15, fontweight="600", color=t.ink)
    fig.text(0.005, 0.977,
             "Sankey Tank — small, clean and rain-fed — tracks antecedent rainfall (r = +0.74). "
             "The large sewage-fed lakes do not: their level is set by inflow and management "
             "works, not the monsoon. The relationship has to be tested per lake, not assumed.",
             ha="left", va="top", fontsize=9.5, color=t.muted)
    fig.tight_layout(rect=(0, 0.015, 1, 0.955))
    vs.source(fig, "Sources: Sentinel-2 L2A (water area) + NASA POWER (rainfall)", t)
    return vs.save(fig, "F23_rainfall_vs_water", FIG, t)


# ============================== F24 =======================================
def f24_pressure_vs_change(t: vs.Theme):
    f = pd.read_csv(PROC / "features_lake_year.csv")
    last = f.sort_values("year").groupby("lake").last().reset_index()

    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    ax.axhline(0, color=t.baseline, lw=1.4, zorder=1)
    ax.scatter(last.bldg_per_ha_250m, last.water_ha_pct,
               s=60 + last.ref_ha * 1.6, color=t.color(0), alpha=0.85,
               edgecolor=t.surface, linewidth=2.0, zorder=3)
    for r in last.itertuples():
        ax.annotate(f"{r.lake.replace(' Lake','').replace(' Tank','')}\n"
                    f"{r.water_ha_pct:+.0f}%",
                    (r.bldg_per_ha_250m, r.water_ha_pct),
                    xytext=(0, 16 + r.ref_ha * 0.05), textcoords="offset points",
                    ha="center", fontsize=9.5, color=t.ink2, fontweight="600")
    ax.set_xlabel("Building density within 250 m of the shoreline (buildings / ha)")
    ax.set_ylabel(f"Change in open water, first → last year (%)")
    ax.grid(axis="x", visible=False)
    ax.text(0.99, 0.03, "Bubble size = lake footprint area",
            transform=ax.transAxes, ha="right", fontsize=9, color=t.muted,
            style="italic")
    vs.title(ax, "Built-up pressure against observed water change",
             "Six lakes is far too small a sample to fit a model on — this is a hypothesis-"
             "generating view, and it is exactly why the real study needs hundreds of lakes.", t)
    vs.source(fig, "Sources: Sentinel-2 L2A (water change) + OpenStreetMap (buildings)", t)
    return vs.save(fig, "F24_pressure_vs_change", FIG, t)


# ============================== F25 =======================================
def f25_corr(t: vs.Theme):
    f = pd.read_csv(PROC / "features_lake_year.csv")
    cols = ["water_ha", "veg_ha", "water_frac", "mean_mndwi", "mean_ndvi",
            "collar_veg_frac", "collar_water_frac",
            "rain_30d", "rain_90d", "rain_365d", "tmax_30d",
            "bldg_per_ha_250m", "road_km_500m", "drain_km_500m", "ref_ha"]
    cols = [c for c in cols if c in f.columns and f[c].notna().any()]
    C = f[cols].corr(numeric_only=True)

    sat = {"water_ha", "veg_ha", "water_frac", "mean_mndwi", "mean_ndvi",
           "collar_veg_frac", "collar_water_frac", "ref_ha"}

    fig, ax = plt.subplots(figsize=(10.6, 9.0))
    im = ax.imshow(C.values, cmap=t.div_cmap.reversed(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    lab = [f"{c}  {'[S]' if c in sat else '[N]'}" for c in cols]
    ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=8.6)
    ax.set_yticklabels(lab, fontsize=8.6)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = C.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.4,
                        color=("white" if abs(v) > 0.55 else t.ink2))
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label("Pearson correlation", color=t.ink2, fontsize=9.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=t.muted, labelcolor=t.ink2, labelsize=8.5)

    vs.title(ax, "Correlation structure of the assembled feature table",
             "[S] = satellite-derived, [N] = non-satellite. 42 lake-years. "
             "Strong within-block correlation means these features are not independent.", t)
    vs.source(fig, "Source: features_lake_year.csv (Sentinel-2 + NASA POWER + OpenStreetMap)", t)
    return vs.save(fig, "F25_feature_correlation", FIG, t)


if __name__ == "__main__":
    print("[1/2] building feature table")
    f = build_features()
    print(f"  features_lake_year.csv  {f.shape[0]} rows x {f.shape[1]} cols")
    print(f"  lakes={f.lake.nunique()}  years={sorted(f.year.unique())}")
    print(f"  columns: {', '.join(f.columns)}")

    print("\n[2/2] figures")
    for mode in ("light", "dark"):
        t = vs.apply(mode)
        for fn in (f23_rain_vs_water, f24_pressure_vs_change, f25_corr):
            try:
                p = fn(t)
                print(f"  [{mode}] {p.name}")
            except Exception as e:                                # noqa: BLE001
                print(f"  [{mode}] {fn.__name__} FAILED: {type(e).__name__}: {e}")
    print("Done.")
