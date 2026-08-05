"""
09_simple_eda.py
================
Simple figures for the student-facing EDA report.

Deliberately plainer than 03-06: bigger text, fewer annotations, one idea per
chart. Same underlying data, just presented so it can be explained in a viva.

S01  where my data comes from
S02  my 6 lakes on the map
S03  raw satellite image, 2019 vs 2025
S04  what MNDWI looks like
S05  rainfall in Bengaluru
S06  water area of each lake, year by year
S07  what is inside Bellandur lake
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import s2_common as s2c

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures_simple"
FIG.mkdir(parents=True, exist_ok=True)
S2DIR = RAW / "s2"

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#008300", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"

area = pd.read_csv(PROC / "lake_water_area.csv")
scenes = pd.read_csv(PROC / "s2_scenes.csv", parse_dates=["datetime"])
wx = pd.read_csv(PROC / "weather_daily.csv", parse_dates=["date"])
wx["PRECTOTCORR"] = pd.to_numeric(wx["PRECTOTCORR"], errors="coerce")
YEARS = sorted(int(p.stem.split("_")[1]) for p in S2DIR.glob("s2_*.npz"))
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def style():
    mpl.rcParams.update({
        "figure.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff", "axes.edgecolor": GREY,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.grid": True, "axes.axisbelow": True, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#eceae4", "grid.linewidth": 1.0,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "600",
        "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 12,
        "legend.frameon": False, "legend.fontsize": 11,
        "lines.linewidth": 2.5, "lines.markersize": 8,
        "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.3,
    })


def save(fig, name):
    p = FIG / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")
    return p


def load(yr):
    with np.load(S2DIR / f"s2_{yr}.npz") as z:
        return {k: z[k] for k in z.files}


def polys_utm():
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform
    crs = (PROC / "grid_crs.txt").read_text(encoding="utf-8").strip()
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    fc = json.loads((PROC / "lake_polygons.geojson").read_text(encoding="utf-8"))
    return {f["properties"]["lake"]: shp_transform(tf, shape(f["geometry"]))
            for f in fc["features"]}


def polys_ll():
    from shapely.geometry import shape
    fc = json.loads((PROC / "lake_polygons.geojson").read_text(encoding="utf-8"))
    return {f["properties"]["lake"]: shape(f["geometry"]) for f in fc["features"]}


def grid():
    minx, miny, maxx, maxy, h, w = np.load(PROC / "grid_meta.npy")
    return (minx, miny, maxx, maxy), (int(h), int(w))


def corridor(pad=900):
    (minx, miny, maxx, maxy), (h, w) = grid()
    P = polys_utm()
    g = [P[k] for k in ("Bellandur Lake", "Varthur Lake") if k in P]
    b = [x.bounds for x in g]
    x0, y0 = min(v[0] for v in b) - pad, min(v[1] for v in b) - pad
    x1, y1 = max(v[2] for v in b) + pad, max(v[3] for v in b) + pad
    rx, ry = (maxx - minx) / w, (maxy - miny) / h
    c0, c1 = int(max(0, (x0 - minx) / rx)), int(min(w, (x1 - minx) / rx))
    r0, r1 = int(max(0, (maxy - y1) / ry)), int(min(h, (maxy - y0) / ry))
    ext = (minx + c0 * rx, minx + c1 * rx, maxy - r1 * ry, maxy - r0 * ry)
    return (slice(r0, r1), slice(c0, c1)), ext


def plain(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ============================== S01 ==============================
def s01_sources():
    src = [("Sentinel-2 satellite\n(ESA, via AWS)", 7, "7 images"),
           ("OpenStreetMap\n(Overpass API)", 5, "5 map layers"),
           ("NASA POWER\n(weather)", 8, "8 cities"),
           ("NASA GIBS\n(satellite maps)", 4, "4 map images")]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = np.arange(len(src))[::-1]
    ax.barh(y, [s[1] for s in src], color=BLUE, height=0.55, zorder=2)
    for yi, (name, n, lab) in zip(y, src):
        ax.text(n + 0.18, yi, lab, va="center", fontsize=12,
                color=INK2, fontweight="600")
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in src], fontsize=11.5)
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("How many things I downloaded")
    ax.grid(axis="y", visible=False)
    ax.set_title("The 4 places I got my data from")
    save(fig, "S01_sources")


# ============================== S02 ==============================
def s02_lake_map():
    P = polys_ll()
    fig, ax = plt.subplots(figsize=(9.5, 7.4))
    for i, (name, g) in enumerate(P.items()):
        parts = [g] if g.geom_type == "Polygon" else list(g.geoms)
        for p in parts:
            xs, ys = p.exterior.xy
            ax.fill(xs, ys, color=BLUE, alpha=0.85, zorder=3)
        c = g.centroid
        short = name.replace(" Lake", "").replace(" Tank", "")
        ax.annotate(short, (c.x, c.y), xytext=(0, 16), textcoords="offset points",
                    ha="center", fontsize=12, fontweight="600", color=INK,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff",
                              ec=GREY, lw=1), zorder=5)
    ax.plot([77.5946], [12.9716], marker="*", ms=20, color=ORANGE, zorder=4)
    ax.annotate("Bengaluru city centre", (77.5946, 12.9716),
                xytext=(0, -26), textcoords="offset points", ha="center",
                fontsize=11, color=ORANGE, fontweight="600")
    ax.set_xlim(77.55, 77.76)
    ax.set_ylim(12.90, 13.06)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1 / np.cos(np.radians(12.97)))
    ax.set_title("The 6 lakes I picked, all in Bengaluru")
    save(fig, "S02_lake_map")


# ============================== S03 ==============================
def s03_raw_image():
    win, ext = corridor()
    W = 13.0
    ar = (ext[1] - ext[0]) / (ext[3] - ext[2])
    fig, axes = plt.subplots(2, 1, figsize=(W, 2 * (W / ar) + 1.3))
    for ax, yr in zip(axes, (YEARS[0], YEARS[-1])):
        sub = {k: v[win] for k, v in load(yr).items()}
        ax.imshow(s2c.rgb(sub), extent=ext, origin="upper")
        d = scenes.loc[scenes.year == yr, "datetime"].iloc[0]
        ax.set_title(f"{yr}   ({d:%d %B %Y})", fontsize=13.5)
        plain(ax)
    fig.suptitle("Raw satellite photo of Bellandur and Varthur lake",
                 x=0.005, y=0.995, ha="left", fontsize=15.5, fontweight="600",
                 color=INK)
    fig.text(0.005, 0.962,
             "This is just the downloaded image, nothing is processed. "
             "The two dark patches are the lakes.",
             ha="left", va="top", fontsize=11.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    save(fig, "S03_raw_image")


# ============================== S04 ==============================
def s04_mndwi():
    from matplotlib.colors import LinearSegmentedColormap
    win, ext = corridor()
    yr = YEARS[-1]
    ix = s2c.indices({k: v[win] for k, v in load(yr).items()})
    cmap = LinearSegmentedColormap.from_list(
        "w", ["#8f2020", "#d03b3b", "#f0efec", "#86b6ef", "#2a78d6", "#0d366b"])
    # size the figure to the image so the panels fill it instead of floating
    W = 13.0
    ar = (ext[1] - ext[0]) / (ext[3] - ext[2])
    fig, axes = plt.subplots(2, 1, figsize=(W, 2 * (W / ar) + 1.1))
    axes[0].imshow(ix["mndwi"], extent=ext, origin="upper", cmap=cmap,
                   vmin=-0.8, vmax=0.8)
    axes[0].set_title("Step 1 — calculate MNDWI for every pixel "
                      "(blue = wet, red = dry)", fontsize=13)
    plain(axes[0])
    axes[1].imshow(ix["water"], extent=ext, origin="upper",
                   cmap=LinearSegmentedColormap.from_list("b", ["#f4f3ef", BLUE]))
    axes[1].set_title("Step 2 — keep only the pixels where MNDWI is above 0. "
                      "That is my water.", fontsize=13)
    plain(axes[1])
    fig.suptitle(f"How I find water in the image  ({yr})",
                 x=0.005, y=0.995, ha="left", fontsize=15.5, fontweight="600", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, "S04_mndwi")


# ============================== S05 ==============================
def s05_rainfall():
    b = wx[wx.city == "Bengaluru"].copy()
    b["m"] = b.date.dt.month
    b["y"] = b.date.dt.year
    monthly = b.groupby(["y", "m"])["PRECTOTCORR"].sum().groupby("m").mean()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    colors = [ORANGE if i < 4 else BLUE for i in range(12)]
    ax.bar(range(12), monthly.values, color=colors, width=0.62, zorder=2)
    for i, v in enumerate(monthly.values):
        ax.text(i, v + 4, f"{v:.0f}", ha="center", fontsize=10.5, color=INK2)
    ax.set_xticks(range(12))
    ax.set_xticklabels(MON)
    ax.set_ylabel("Rainfall (mm)")
    ax.set_ylim(0, monthly.max() * 1.22)
    ax.grid(axis="x", visible=False)
    ax.set_title("Average rainfall in Bengaluru, month by month")
    ax.text(1.5, monthly.max() * 1.12, "I take my satellite images\nin these dry months",
            ha="center", fontsize=11, color=ORANGE, fontweight="600")
    save(fig, "S05_rainfall")


# ============================== S06 ==============================
def s06_water_series():
    order = area.groupby("lake")["ref_ha"].first().sort_values(ascending=False).index
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 6.6), sharex=True)
    for ax, lake in zip(axes.ravel(), order):
        s = area[area.lake == lake].sort_values("year")
        ax.plot(s.year, s.water_ha, color=BLUE, marker="o", zorder=3)
        ax.fill_between(s.year, s.water_ha, color=BLUE, alpha=0.15)
        ax.set_title(lake.replace(" Lake", "").replace(" Tank", " Tank"), fontsize=13)
        ax.set_xticks(s.year.tolist())
        ax.set_xticklabels([str(y)[2:] for y in s.year], fontsize=10.5)
        ax.set_ylim(0, max(s.water_ha.max() * 1.25, 1))
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Water (hectare)")
    fig.suptitle("How much water is in each lake, every year",
                 x=0.005, y=1.03, ha="left", fontsize=15.5, fontweight="600", color=INK)
    fig.text(0.005, 0.995, "Year is shown in short form (19 means 2019). "
                           "Each lake has its own scale because sizes are very different.",
             ha="left", va="top", fontsize=11.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "S06_water_series")


# ============================== S07 ==============================
def s07_inside_lake():
    s = area[area.lake == "Bellandur Lake"].sort_values("year")
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(s))
    ax.bar(x, s.water_ha, 0.62, color=BLUE, label="Water", zorder=2)
    ax.bar(x, s.veg_ha, 0.62, bottom=s.water_ha, color=GREEN,
           label="Plants (water hyacinth)", zorder=2)
    ax.bar(x, s.other_ha, 0.62, bottom=s.water_ha + s.veg_ha, color=GREY,
           label="Dry / empty ground", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(s.year)
    ax.set_ylabel("Area (hectare)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.14))
    ax.set_title("What is actually inside Bellandur lake", pad=34)
    save(fig, "S07_inside_lake")


if __name__ == "__main__":
    style()
    print("simple figures ->", FIG.relative_to(ROOT))
    for fn in (s01_sources, s02_lake_map, s03_raw_image, s04_mndwi,
               s05_rainfall, s06_water_series, s07_inside_lake):
        try:
            fn()
        except Exception as e:                                    # noqa: BLE001
            print(f"  {fn.__name__} FAILED: {type(e).__name__}: {e}")
    print("Done.")
