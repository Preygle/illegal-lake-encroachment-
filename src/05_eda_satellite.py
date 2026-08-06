"""
05_eda_satellite.py
===================
Exploratory analysis of the SATELLITE layers (Sentinel-2 L2A + NASA GIBS).

F14  true-colour composite per year, Bellandur–Varthur corridor
F15  MNDWI index maps per year (diverging, zero = the water threshold)
F16  footprint composition: open water / vegetation / bare, per lake
F17  open-water area time series, small multiples
F18  why MNDWI: index distributions and the separability of water from land
F19  spectral signature of water vs vegetation vs built-up
F20  water change 2019 -> latest: stable, lost, gained
F21  NASA GIBS regional context (MODIS true colour, night lights, land-surface temp)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import s2_common as s2c
import viz_style as vs

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
S2DIR = RAW / "s2"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

SRC_S2 = ("Source: Sentinel-2 L2A (ESA Copernicus) via Earth Search STAC / AWS open data. "
          "20 m grid, EPSG:32643, dry-season scenes")

lakes = pd.read_csv(PROC / "lakes.csv")
area = pd.read_csv(PROC / "lake_water_area.csv")
scenes = pd.read_csv(PROC / "s2_scenes.csv", parse_dates=["datetime"])
YEARS = sorted(int(p.stem.split("_")[1]) for p in S2DIR.glob("s2_*.npz"))


# --------------------------------------------------------------------------
def load(year: int) -> dict:
    with np.load(S2DIR / f"s2_{year}.npz") as z:
        return {k: z[k] for k in z.files}


def grid():
    meta = np.load(PROC / "grid_meta.npy")
    minx, miny, maxx, maxy, h, w = meta
    crs = (PROC / "grid_crs.txt").read_text(encoding="utf-8").strip()
    return (minx, miny, maxx, maxy), (int(h), int(w)), crs


def polys_utm():
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform
    _, _, crs = grid()
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    fc = json.loads((PROC / "lake_polygons.geojson").read_text(encoding="utf-8"))
    return {f["properties"]["lake"]: shp_transform(tf, shape(f["geometry"]))
            for f in fc["features"]}


def window_for(geoms, pad: float = 700.0):
    """Row/col slice covering `geoms` (UTM) plus a padding collar."""
    (minx, miny, maxx, maxy), (h, w), _ = grid()
    xs = [g.bounds for g in geoms]
    x0 = min(b[0] for b in xs) - pad
    y0 = min(b[1] for b in xs) - pad
    x1 = max(b[2] for b in xs) + pad
    y1 = max(b[3] for b in xs) + pad
    res_x, res_y = (maxx - minx) / w, (maxy - miny) / h
    c0 = int(max(0, (x0 - minx) / res_x))
    c1 = int(min(w, (x1 - minx) / res_x))
    r0 = int(max(0, (maxy - y1) / res_y))
    r1 = int(min(h, (maxy - y0) / res_y))
    ext = (minx + c0 * res_x, minx + c1 * res_x,
           maxy - r1 * res_y, maxy - r0 * res_y)
    return (slice(r0, r1), slice(c0, c1)), ext


def scale_bar(ax, ext, km: float = 2.0, t: vs.Theme = None):
    x0, x1, y0, y1 = ext
    L = km * 1000
    xa = x0 + (x1 - x0) * 0.06
    ya = y0 + (y1 - y0) * 0.065
    ax.plot([xa, xa + L], [ya, ya], color="white", lw=4, solid_capstyle="butt", zorder=9)
    ax.plot([xa, xa + L], [ya, ya], color="#0b0b0b", lw=2, solid_capstyle="butt", zorder=10)
    ax.text(xa + L / 2, ya + (y1 - y0) * 0.022, f"{km:g} km", ha="center",
            fontsize=8.5, color="white", fontweight="700", zorder=11,
            bbox=dict(boxstyle="round,pad=0.15", fc="#0b0b0b99", ec="none"))


def bare_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ============================== F14 =======================================
def f14_truecolor(t: vs.Theme):
    P = polys_utm()
    target = [P[k] for k in ("Bellandur Lake", "Varthur Lake") if k in P]
    win, ext = window_for(target, pad=900)

    n = len(YEARS)
    ncol, nrow, figsize = s2c.panel_grid(ext, n, target_w=15.0)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax, yr in zip(axes, YEARS):
        st = load(yr)
        sub = {k: v[win] for k, v in st.items()}
        ax.imshow(s2c.rgb(sub), extent=ext, origin="upper")
        for name in ("Bellandur Lake", "Varthur Lake"):
            if name in P:
                g = P[name]
                gs = [g] if g.geom_type == "Polygon" else list(g.geoms)
                for gg in gs:
                    ax.plot(*gg.exterior.xy, color=t.color(3), lw=1.5, zorder=4)
        d = scenes.loc[scenes.year == yr, "datetime"]
        ax.set_title(f"{yr}   ·   {d.iloc[0]:%d %b}" if len(d) else str(yr),
                     fontsize=11.5)
        bare_axes(ax)
        scale_bar(ax, ext, 2, t)
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Bellandur–Varthur corridor in true colour, one dry-season scene per year",
                 x=0.005, y=0.998, ha="left", fontsize=15, fontweight="600", color=t.ink)
    fig.text(0.005, 0.972,
             "Sentinel-2 bands B04/B03/B02, per-band 2–98 percentile stretch. Yellow outlines are "
             "the OpenStreetMap lake footprints used as fixed reference boundaries. "
             "Note the 2025 scene is early February, the rest are March–April.",
             ha="left", va="top", fontsize=9.5, color=t.muted)
    fig.tight_layout(rect=(0, 0.012, 1, 0.955))
    vs.source(fig, SRC_S2, t)
    return vs.save(fig, "F14_truecolor_by_year", FIG, t)


# ============================== F15 =======================================
def f15_mndwi(t: vs.Theme):
    P = polys_utm()
    target = [P[k] for k in ("Bellandur Lake", "Varthur Lake") if k in P]
    win, ext = window_for(target, pad=900)

    n = len(YEARS)
    ncol, nrow, figsize = s2c.panel_grid(ext, n, target_w=15.0)
    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize[0], figsize[1] + 0.5))
    axes = np.atleast_1d(axes).ravel()
    im = None
    for ax, yr in zip(axes, YEARS):
        ix = s2c.indices(load(yr))
        im = ax.imshow(ix["mndwi"][win], extent=ext, origin="upper",
                       cmap=t.div_cmap.reversed(), vmin=-0.8, vmax=0.8)
        for name in ("Bellandur Lake", "Varthur Lake"):
            if name in P:
                g = P[name]
                gs = [g] if g.geom_type == "Polygon" else list(g.geoms)
                for gg in gs:
                    ax.plot(*gg.exterior.xy, color=t.ink, lw=1.3, zorder=4)
        d = scenes.loc[scenes.year == yr, "datetime"]
        ax.set_title(f"{yr}   ·   {d.iloc[0]:%d %b}" if len(d) else str(yr),
                     fontsize=11.5)
        bare_axes(ax)
        scale_bar(ax, ext, 2, t)
    for ax in axes[n:]:
        ax.set_visible(False)

    cb = fig.colorbar(im, ax=list(axes), orientation="horizontal",
                      fraction=0.026, pad=0.03, aspect=55)
    cb.set_label("MNDWI  —  negative = dry land / built-up      0 = water threshold      positive = open water",
                 color=t.ink2, fontsize=9.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=t.muted, labelcolor=t.ink2, labelsize=8.5)

    fig.suptitle("Modified Normalised Difference Water Index, year by year",
                 x=0.005, y=0.998, ha="left", fontsize=15, fontweight="600", color=t.ink)
    fig.text(0.005, 0.972,
             "MNDWI = (green − SWIR16) / (green + SWIR16). Diverging scale centred on the "
             "zero threshold that separates water from land. Most of the Bellandur and Varthur "
             "beds sit just below the threshold — that is hyacinth mat, not open water.",
             ha="left", va="top", fontsize=9.5, color=t.muted)
    vs.source(fig, SRC_S2, t)
    return vs.save(fig, "F15_mndwi_by_year", FIG, t)


# ============================== F16 =======================================
def f16_composition(t: vs.Theme):
    order = (area.groupby("lake")["ref_ha"].first().sort_values(ascending=False).index)
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.4), sharex=True)
    for ax, lake in zip(axes.ravel(), order):
        s = area[area.lake == lake].sort_values("year")
        ax.stackplot(s.year, s.water_ha, s.veg_ha, s.other_ha,
                     colors=[t.color(0), t.color(5), t.baseline],
                     labels=["Open water", "Vegetation", "Bare / built"],
                     edgecolor=t.surface, linewidth=1.2)
        ax.set_title(f"{lake}   ({s.ref_ha.iloc[0]:.0f} ha footprint)", fontsize=11)
        ax.set_xticks(s.year.tolist())
        ax.set_xticklabels(s.year.tolist(), fontsize=8.5, rotation=45)
        ax.grid(axis="x", visible=False)
        w0, w1 = s.water_frac.iloc[0], s.water_frac.iloc[-1]
        ax.text(0.98, 0.94, f"water {w0:.0%} → {w1:.0%}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=t.ink2, fontweight="600")
    for ax in axes[:, 0]:
        ax.set_ylabel("Area (ha)")
    axes.ravel()[0].legend(loc="lower left", ncol=3, fontsize=8.6,
                           bbox_to_anchor=(0, 1.16))

    fig.suptitle("What is actually inside each lake footprint",
                 x=0.005, y=1.005, ha="left", fontsize=15, fontweight="600", color=t.ink)
    fig.text(0.005, 0.977,
             "Classifying the fixed OSM footprint each year. Vegetation inside a lake bed is "
             "hyacinth mat or reclaimed land — not open water, and not the same as lake loss.",
             ha="left", va="top", fontsize=9.5, color=t.muted)
    fig.tight_layout(rect=(0, 0.015, 1, 0.955))
    vs.source(fig, SRC_S2, t)
    return vs.save(fig, "F16_footprint_composition", FIG, t)


# ============================== F17 =======================================
def f17_timeseries(t: vs.Theme):
    order = (area.groupby("lake")["water_ha"].mean().sort_values(ascending=False).index)
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.0), sharex=True)
    for ax, lake in zip(axes.ravel(), order):
        s = area[area.lake == lake].sort_values("year")
        ax.plot(s.year, s.water_ha, color=t.color(0), lw=2.2, marker="o",
                ms=7, mec=t.surface, mew=1.6, zorder=3)
        ax.fill_between(s.year, s.water_ha, color=t.color(0), alpha=0.16, zorder=2)
        for x, y in zip(s.year, s.water_ha):
            ax.annotate(f"{y:.0f}", (x, y), xytext=(0, 9),
                        textcoords="offset points", ha="center",
                        fontsize=8.2, color=t.ink2)
        ax.set_title(lake, fontsize=11.5)
        ax.set_xticks(s.year.tolist())
        ax.set_xticklabels(s.year.tolist(), fontsize=8.5, rotation=45)
        ax.set_ylim(0, max(s.water_ha.max() * 1.30, 1))
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Open water (ha)")

    fig.suptitle("Open-water area per lake, 2019 → 2025",
                 x=0.005, y=1.005, ha="left", fontsize=15, fontweight="600", color=t.ink)
    fig.text(0.005, 0.977,
             "Each panel has its own y-scale — these lakes differ by an order of magnitude in size. "
             "This series is the candidate label for the encroachment model.",
             ha="left", va="top", fontsize=9.5, color=t.muted)
    fig.tight_layout(rect=(0, 0.015, 1, 0.955))
    vs.source(fig, SRC_S2, t)
    return vs.save(fig, "F17_water_area_timeseries", FIG, t)


# ============================== F18 =======================================
def f18_separability(t: vs.Theme):
    yr = YEARS[-1]
    ix = s2c.indices(load(yr))
    if "scl_water" not in ix:
        raise RuntimeError("scene has no SCL band - cannot score indices independently")
    ok = ~ix["cloud"]
    ref_water = ix["scl_water"] & ok              # independent reference label
    ref_land = (~ref_water) & ok

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4))

    bins = np.linspace(-1, 1, 121)
    for ax, key, name in [(axes[0], "mndwi", "MNDWI  (green − SWIR16)"),
                          (axes[1], "ndwi", "NDWI  (green − NIR)")]:
        v = ix[key]
        ax.hist(v[ref_land], bins=bins, color=t.baseline, alpha=0.95,
                label="Land (SCL ≠ water)", zorder=2)
        ax.hist(v[ref_water], bins=bins, color=t.color(0), alpha=0.9,
                label="Water (SCL = water)", zorder=3)
        ax.axvline(0, color=t.color(1), lw=2.0, ls=(0, (4, 2)), zorder=4)
        ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel("Pixels (log)")
        ax.grid(axis="x", visible=False)

        # separability: how cleanly does the zero threshold split the two classes?
        tp = float((v[ref_water] > 0).mean())
        fp = float((v[ref_land] > 0).mean())
        mw, ml = float(v[ref_water].mean()), float(v[ref_land].mean())
        sw, sl = float(v[ref_water].std()), float(v[ref_land].std())
        jm = abs(mw - ml) / np.sqrt(sw ** 2 + sl ** 2)
        ax.text(0.03, 0.96,
                f"water above 0: {tp:.1%}\nland above 0: {fp:.1%}\nseparability: {jm:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
                color=t.ink2, fontweight="600",
                bbox=dict(boxstyle="round,pad=0.4", fc=t.surface,
                          ec=t.baseline, lw=0.8))
        ax.annotate("threshold = 0", (0, ax.get_ylim()[1] * 0.30),
                    xytext=(14, 0), textcoords="offset points",
                    fontsize=9, color=t.color(1), fontweight="600")
        ax.legend(loc="upper right")

    fig.suptitle(f"Why MNDWI is the better water index here  ({yr} scene)",
                 x=0.005, y=1.005, ha="left", fontsize=15, fontweight="600", color=t.ink)
    fig.text(0.005, 0.972,
             "Both indices scored against Sentinel-2's own Scene Classification Layer as an "
             "independent reference. Higher separability and lower land false-positives win.",
             ha="left", va="top", fontsize=9.5, color=t.muted)
    fig.tight_layout(rect=(0, 0.02, 1, 0.945))
    vs.source(fig, SRC_S2, t)
    return vs.save(fig, "F18_index_separability", FIG, t)


# ============================== F19 =======================================
def f19_spectra(t: vs.Theme):
    yr = YEARS[-1]
    st = load(yr)
    ix = s2c.indices(st)
    ok = ~ix["cloud"]
    cls = {"Open water": ix["water"] & ok,
           "Vegetation": ix["veg"] & ok & ~ix["water"],
           "Built-up / bare": ok & ~ix["water"] & ~ix["veg"]}
    bands = [("blue", 490), ("green", 560), ("red", 665), ("nir", 842), ("swir16", 1610)]
    bands = [(b, wl) for b, wl in bands if b in st]

    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    for i, (name, m) in enumerate(cls.items()):
        if m.sum() < 50:
            continue
        mu = [float(np.mean(s2c.reflectance(st[b])[m])) for b, _ in bands]
        lo = [float(np.percentile(s2c.reflectance(st[b])[m], 25)) for b, _ in bands]
        hi = [float(np.percentile(s2c.reflectance(st[b])[m], 75)) for b, _ in bands]
        wl = [w for _, w in bands]
        ax.fill_between(wl, lo, hi, color=t.color(i), alpha=0.16, zorder=2)
        ax.plot(wl, mu, color=t.color(i), lw=2.4, marker="o", ms=7.5,
                mec=t.surface, mew=1.6, zorder=3, label=f"{name}  ({m.sum()/m.size:.0%})")
        ax.annotate(name, (wl[-1], mu[-1]), xytext=(10, 0),
                    textcoords="offset points", fontsize=10, color=t.color(i),
                    fontweight="600", va="center")

    ax.set_xticks([w for _, w in bands])
    ax.set_xticklabels([f"{b.upper()}\n{w} nm" for b, w in bands], fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Surface reflectance")
    ax.set_xlim(430, 1900)
    ax.legend(loc="upper left", title=None)
    vs.title(ax, f"Spectral signatures that make water separable  ({yr})",
             "Mean reflectance with the interquartile band. Water absorbs strongly in NIR and SWIR — "
             "the gap at 1610 nm is what MNDWI exploits.", t)
    vs.source(fig, SRC_S2, t)
    return vs.save(fig, "F19_spectral_signatures", FIG, t)


# ============================== F20 =======================================
def f20_change(t: vs.Theme):
    """
    Water change confined to the lake footprints.

    Run city-wide, this comparison reports thousands of hectares of spurious
    'water gain': the February 2025 scene has a lower sun elevation than the
    March/April scenes, and building shadows read as MNDWI > 0. Restricting the
    comparison to the reference footprints - and dropping isolated pixels with a
    morphological opening - removes that artefact and answers the actual
    question, which is only ever about the lakes.
    """
    from matplotlib.colors import ListedColormap
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds as t_from_bounds
    from scipy import ndimage
    from shapely.geometry import mapping

    y0, y1 = YEARS[0], YEARS[-1]
    a, b = s2c.indices(load(y0)), s2c.indices(load(y1))
    ok = (~a["cloud"]) & (~b["cloud"])
    k = np.ones((3, 3), bool)
    w0 = ndimage.binary_opening(a["water"] & ok, structure=k)
    w1 = ndimage.binary_opening(b["water"] & ok, structure=k)

    P = polys_utm()
    (minx, miny, maxx, maxy), (h, w), _ = grid()
    tr = t_from_bounds(minx, miny, maxx, maxy, w, h)
    inside = geometry_mask([mapping(g) for g in P.values()], out_shape=(h, w),
                           transform=tr, invert=True)

    cat = np.zeros(w0.shape, dtype="uint8")        # 0 = outside / never water
    cat[inside & w0 & w1] = 1                       # stable
    cat[inside & w0 & ~w1] = 2                      # lost
    cat[inside & ~w0 & w1] = 3                      # gained

    target = [P[k_] for k_ in ("Bellandur Lake", "Varthur Lake") if k_ in P]
    win, ext = window_for(target, pad=900)
    px_ha = 0.04

    fig, ax = plt.subplots(figsize=(15.0, 6.4))
    grey = s2c.rgb({k_: v[win] for k_, v in load(y1).items()}).mean(axis=2)
    ax.imshow(grey, extent=ext, origin="upper", cmap="gray", vmin=-0.15, vmax=1.5)
    m = np.ma.masked_where(cat[win] == 0, cat[win])
    ax.imshow(m, extent=ext, origin="upper", vmin=0, vmax=3,
              cmap=ListedColormap([t.surface, t.color(0), t.status["critical"],
                                   t.status["good"]]),
              interpolation="nearest")
    for name in ("Bellandur Lake", "Varthur Lake"):
        g = P[name]
        for gg in ([g] if g.geom_type == "Polygon" else list(g.geoms)):
            ax.plot(*gg.exterior.xy, color="#ffffff", lw=1.4, alpha=0.85, zorder=4)
    bare_axes(ax)
    scale_bar(ax, ext, 2, t)

    n_st, n_lo, n_ga = (int(((cat == c) & inside).sum()) for c in (1, 2, 3))
    ax.legend(handles=[
        Patch(fc=t.color(0), label=f"Water in both years — {n_st*px_ha:,.0f} ha"),
        Patch(fc=t.status["critical"], label=f"Water lost by {y1} — {n_lo*px_ha:,.0f} ha"),
        Patch(fc=t.status["good"], label=f"Water gained by {y1} — {n_ga*px_ha:,.0f} ha")],
        loc="lower right", ncol=1, frameon=True, framealpha=0.94,
        facecolor=t.surface, edgecolor=t.baseline, labelcolor=t.ink)

    rows = []
    for name, g in P.items():
        mk = geometry_mask([mapping(g)], out_shape=(h, w), transform=tr, invert=True)
        rows.append({"lake": name,
                     "stable_ha": round(int((mk & (cat == 1)).sum()) * px_ha, 2),
                     "lost_ha": round(int((mk & (cat == 2)).sum()) * px_ha, 2),
                     "gained_ha": round(int((mk & (cat == 3)).sum()) * px_ha, 2)})
    pd.DataFrame(rows).to_csv(TAB / "water_change_by_lake.csv", index=False)

    ax.set_title(f"Surface-water change inside the lake footprints, {y0} → {y1}",
                 loc="left", pad=26)
    ax.text(0.0, 1.012,
            "Change classes are shown only inside the reference footprints, over a greyscale "
            f"{y1} basemap. Totals cover all six lakes; the view is the Bellandur–Varthur corridor.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, color=t.muted)
    vs.source(fig, SRC_S2, t)
    return vs.save(fig, "F20_water_change_map", FIG, t)


# ============================== F21 =======================================
def f21_gibs(t: vs.Theme):
    import matplotlib.image as mpimg
    panels = [("modis_truecolor_india", "MODIS Terra true colour — India",
               "250 m · 15 Feb 2024", (67.0, 98.0, 6.0, 37.0)),
              ("modis_truecolor_karnataka", "Zoomed to Karnataka",
               "250 m · 15 Feb 2024", (74.0, 81.0, 11.5, 18.5))]
    avail = [p for p in panels if (RAW / "gibs" / f"{p[0]}.png").exists()]
    if not avail:
        raise RuntimeError("no GIBS rasters cached")

    fig, axes = plt.subplots(1, len(avail), figsize=(5.4 * len(avail), 6.0))
    axes = np.atleast_1d(axes)
    for ax, (name, title_, sub, ext) in zip(axes, avail):
        ax.imshow(mpimg.imread(RAW / "gibs" / f"{name}.png"), extent=ext,
                  origin="upper")
        ax.scatter([77.5946], [12.9716], s=90, marker="o", color=t.color(1),
                   edgecolor="white", linewidth=1.8, zorder=5)
        ax.annotate("Bengaluru", (77.5946, 12.9716), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=9,
                    color="white", fontweight="600", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.24", fc=t.color(1), ec="none"))
        ax.set_title(f"{title_}\n{sub}", fontsize=10.5)
        bare_axes(ax)

    fig.suptitle("Where the study area sits — NASA GIBS regional context",
                 x=0.005, y=1.006, ha="left", fontsize=15, fontweight="600", color=t.ink)
    fig.text(0.005, 0.968,
             "MODIS Terra true colour at 250 m, from a keyless WMS endpoint. The coarse tier "
             "that frames the 10 m Sentinel-2 work over the Bengaluru lake belt.",
             ha="left", va="top", fontsize=9.5, color=t.muted)
    fig.tight_layout(rect=(0, 0.015, 1, 0.945))
    vs.source(fig, "Source: NASA GIBS / Worldview WMS (EOSDIS) — MODIS Terra, VIIRS SNPP", t)
    return vs.save(fig, "F21_gibs_regional_context", FIG, t)


if __name__ == "__main__":
    print(f"  years available: {YEARS}")
    area.to_csv(TAB / "lake_water_area.csv", index=False)
    for mode in ("light", "dark"):
        t = vs.apply(mode)
        for fn in (f14_truecolor, f15_mndwi, f16_composition, f17_timeseries,
                   f18_separability, f19_spectra, f20_change, f21_gibs):
            try:
                p = fn(t)
                print(f"  [{mode}] {p.name}")
            except Exception as e:                                # noqa: BLE001
                print(f"  [{mode}] {fn.__name__} FAILED: {type(e).__name__}: {e}")
    print("Done.")
