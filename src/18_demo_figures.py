"""
18_demo_figures.py
==================
Presentation figures for the model, from the out-of-fold land-cover maps and the
result tables. Every map shows a lake through a model that never trained on it.

M01  Bellandur 2019 / 2021 / 2025 - drained, not built on
M02  Madiwala 2019 vs 2020 - a hyacinth mat, not a lost lake
M03  per-lake macro-F1 for the index rule, Random Forest and U-Net
M04  buildings inside the 75 m fence, 2016-2023

    python src/18_demo_figures.py

Writes outputs/figures/{light,dark}/M0*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import viz_style as vs
from dl import common as C
from dl import viz as V

FIG = C.ROOT / "outputs" / "figures"
PRED = C.DL / "pred"


def pred(year: int) -> np.ndarray:
    return np.load(PRED / f"landcover_{year}.npy")


def map_grid(t, lake, years, rasters, headline, sub):
    """Top row true colour, bottom row the U-Net map, one column per year."""
    win = V.window(lake, rasters)
    h, w = win[1] - win[0], win[3] - win[2]
    n = len(years)
    pw = 15.0 / n
    fig, axes = plt.subplots(2, n, figsize=(15.0, 2 * pw * h / w + 1.7))
    for j, y in enumerate(years):
        V.draw_rgb(axes[0, j], y, lake, rasters, win)
        axes[0, j].set_title(str(y), fontsize=12)
        m = pred(y)
        V.draw_classes(axes[1, j], m, lake, rasters, win, year=y,
                       fade_to=t.surface, line_color=t.ink)
        axes[1, j].text(0.0, -0.03, V.comp_line(V.composition_ha(m, lake, rasters)),
                        transform=axes[1, j].transAxes, ha="left", va="top",
                        fontsize=8.8, color=t.ink2)
    axes[0, 0].text(-0.03, 0.5, "Sentinel-2", transform=axes[0, 0].transAxes,
                    rotation=90, ha="right", va="center", fontsize=10, color=t.ink2)
    axes[1, 0].text(-0.03, 0.5, "U-Net land cover", transform=axes[1, 0].transAxes,
                    rotation=90, ha="right", va="center", fontsize=10, color=t.ink2)
    fig.legend(handles=V.legend_handles(), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.012), fontsize=10)
    fig.suptitle(headline, x=0.01, y=0.995, ha="left", fontsize=15,
                 fontweight="600", color=t.ink)
    fig.text(0.01, 0.948, sub, ha="left", va="top", fontsize=10, color=t.muted)
    fig.tight_layout(rect=(0.012, 0.07, 1, 0.92))
    vs.source(fig, "Sentinel-2 L2A via Earth Search / AWS Open Data. Solid line: "
              "lake footprint (OpenStreetMap). Dashed: 30 m statutory buffer. "
              "Hectares are inside the footprint.", t)
    return fig


def m01(t, rasters):
    fig = map_grid(
        t, "Bellandur Lake", [2019, 2021, 2025], rasters,
        "Bellandur in 2021: the water was gone, and the bed was bare, not built on",
        "Open water fell to zero, but 146 ha of the footprint became dry lake bed and the water "
        "returned by 2025. That is a drain for desilting. A water-only measure would have called "
        "it encroachment.")
    return vs.save(fig, "M01_bellandur_drain_and_refill", FIG, t)


def m02(t, rasters):
    fig = map_grid(
        t, "Madiwala Lake", [2019, 2020], rasters,
        "Madiwala in 2020: the lake was not lost, a hyacinth mat covered it",
        "Open water fell from 65 to 11 ha while vegetation inside the footprint rose from 5 to 71 ha. "
        "The four-class model separates a weed mat from lake loss.")
    return vs.save(fig, "M02_madiwala_hyacinth", FIG, t)


def m03(t):
    un = pd.read_csv(C.DL / "unet" / "per_fold.csv").set_index("test_lake").macro_f1
    rf = pd.read_csv(C.DL / "rf" / "per_fold.csv")
    rule = rf[rf.model == "mndwi_rule"].set_index("test_lake").macro_f1
    rfm = rf[rf.model == "random_forest"].set_index("test_lake").macro_f1
    lakes = C.lake_order()[::-1]                         # largest at the top
    y = np.arange(len(lakes))

    series = [("MNDWI / NDVI rule", rule, t.muted),
              ("Random Forest", rfm, t.color(1)),
              ("U-Net", un, t.color(0))]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for lake_i, lake in enumerate(lakes):
        vals = [s[1][lake] for s in series]
        ax.plot([min(vals), max(vals)], [lake_i, lake_i], color=t.grid, lw=2.0,
                zorder=1, solid_capstyle="round")
    for name, s, col in series:
        ax.scatter([s[l] for l in lakes], y, s=95, color=col, zorder=3,
                   edgecolor=t.surface, linewidth=2.0,
                   label=f"{name}  (mean {s.mean():.3f})")
    ax.set_yticks(y)
    ax.set_yticklabels([l.replace(" Lake", "") for l in lakes])
    ax.set_xlabel("Macro-F1 on the held-out lake  (each lake scored by a model that never saw it)")
    ax.set_xlim(0.55, 0.88)
    ax.grid(axis="y", visible=False)
    # Inside the plot, top-left: every lake's scores sit right of 0.69 in the
    # top rows, so this corner is empty. Above the plot it collided with the
    # title's subtitle.
    ax.legend(loc="upper left", ncol=1, fontsize=9.5, handletextpad=0.3,
              borderaxespad=0.6)
    d = un.mean() - rfm.mean()
    verdict = "U-Net recommended" if d > 0.02 else "Random Forest recommended"
    ax.text(0.99, 0.02, f"U-Net − Random Forest = {d:+.3f}\nmargin set in advance: +0.020\n"
            f"→ {verdict}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color=t.ink2, linespacing=1.5)
    vs.title(ax, "U-Net leads on 4 of 6 lakes, but by less than the margin set in advance",
             "Both learned models clearly beat the index rule. The decision rule was fixed before "
             "any numbers existed, so the closer result does not get rounded up.", t)
    fig.subplots_adjust(top=0.84)
    vs.source(fig, "Nested leave-one-lake-out, 6 folds. outputs/dl/unet/per_fold.csv, "
              "outputs/dl/rf/per_fold.csv", t)
    return vs.save(fig, "M03_model_comparison", FIG, t)


def m04(t):
    ob = pd.read_csv(C.DL / "gee" / "open_buildings_intrusion.csv")
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.axhline(100, color=t.baseline, lw=1.2, zorder=1)
    for i, lake in enumerate(C.lake_order()):
        s = ob[ob.lake == lake].sort_values("year")
        v = s.ob_buildings_ring_75m.to_numpy()
        idx = v / v[0] * 100
        ax.plot(s.year, idx, color=t.color(i), lw=2.0, zorder=2,
                label=f"{lake.replace(' Lake', '')}  ({v[0]:.0f} → {v[-1]:.0f})")
        ax.scatter([s.year.iloc[-1]], [idx[-1]], s=70, color=t.color(i), zorder=3,
                   edgecolor=t.surface, linewidth=2.0)
    ax.set_xticks(sorted(ob.year.unique()))
    ax.set_ylabel("Buildings inside the 75 m fence  (2016 = 100)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=9.3, ncol=2, title="Lake  (buildings 2016 → 2023)",
              title_fontsize=9.3, alignment="left")
    vs.title(ax, "Buildings inside the 75 m buffer grew at every lake",
             "Pooled across the six lakes: +5.4% a year (95% CI +4.1% to +6.7%), "
             "from a mixed-effects model with a separate baseline per lake.", t)
    vs.source(fig, "GOOGLE/Research/open-buildings-temporal/v1 (Google Open Buildings "
              "Temporal), 2016-2023. Counts from the fractional-count band.", t)
    return vs.save(fig, "M04_building_growth", FIG, t)


if __name__ == "__main__":
    rasters = C.lake_rasters()
    for mode in ("light", "dark"):
        t = vs.apply(mode)
        for fn, args in ((m01, (t, rasters)), (m02, (t, rasters)),
                         (m03, (t,)), (m04, (t,))):
            p = fn(*args)
            print(f"  [{mode}] {p.relative_to(C.ROOT)}")
