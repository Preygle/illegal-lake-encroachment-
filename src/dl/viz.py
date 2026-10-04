"""
dl/viz.py
=========
Map rendering shared by the presentation figures (18_demo_figures.py) and the
live demo (demo.py), so a land-cover map looks identical wherever it appears.

Class colours
-------------
    water       #2166ac
    vegetation  #1b7837   (inside a lake footprint: water hyacinth)
    bare        #c98a1e
    built       #c82828

Checked with the dataviz skill's palette validator against the light surface:
lightness band, chroma floor, colour-blind separation (worst adjacent pair
dE 12.1, target >= 8) and normal-vision separation all pass. The bare-bed colour
was previously #bfa06e, which failed the chroma floor - it read as grey, on the
one class this project's argument depends on. Bare's contrast against the
surface is 2.87:1, below 3:1, so every map carries a labelled legend with
hectare values rather than relying on colour alone.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from . import common as C

CLASS_COLORS = ["#2166ac", "#1b7837", "#c98a1e", "#c82828"]
CLASS_LABELS = ["Open water", "Hyacinth / vegetation", "Bare lake bed", "Built-up"]
CLASS_CMAP = ListedColormap(CLASS_COLORS)

FENCE_STYLE = dict(color="#ffffff", lw=1.0, ls=(0, (3, 2)))
SHORE_STYLE = dict(color="#ffffff", lw=1.4)


def window(lake: str, rasters=None, margin_px: int = 6):
    """Row/column slice around a lake's footprint-plus-collar."""
    from .dataset import _bbox

    _, shape, _, _ = C.grid()
    rasters = rasters or C.lake_rasters()
    r0, r1, c0, c1 = _bbox(rasters[lake]["region"])
    return (max(0, r0 - margin_px), min(shape[0], r1 + margin_px),
            max(0, c0 - margin_px), min(shape[1], c1 + margin_px))


def extent(win):
    """imshow extent, in grid metres, for a window - so outlines can be drawn in metres."""
    bounds, shape, _, _ = C.grid()
    res = (bounds[2] - bounds[0]) / shape[1]
    r0, r1, c0, c1 = win
    return (bounds[0] + c0 * res, bounds[0] + c1 * res,
            bounds[3] - r1 * res, bounds[3] - r0 * res)


def truecolor(year: int, win) -> np.ndarray:
    """True-colour composite, stretched over the window rather than the whole scene."""
    import sys

    sys.path.insert(0, str(C.ROOT / "src"))
    import s2_common as s2c

    st = C.load_stack(year)
    r0, r1, c0, c1 = win
    return np.clip(s2c.rgb({k: v[r0:r1, c0:c1] for k, v in st.items()}), 0, 1)


def outline(ax, geom, **style):
    parts = geom.geoms if hasattr(geom, "geoms") else [geom]
    for p in parts:
        x, y = p.exterior.xy
        ax.plot(x, y, **style)


def draw_rgb(ax, year: int, lake: str, rasters, win):
    ax.imshow(truecolor(year, win), extent=extent(win), interpolation="nearest")
    outline(ax, rasters[lake]["geom"], **SHORE_STYLE)
    _bare_axes(ax)


def draw_classes(ax, classmap: np.ndarray, lake: str, rasters, win,
                 fence_m: int = 30, year: int | None = None,
                 fade_to: str = "#ffffff", line_color: str = "#111111"):
    """
    Land-cover classes inside the lake's labelled region, over a true-colour base
    faded toward the page surface so the surroundings stay legible without
    competing with the classes. The footprint is outlined solid and the
    statutory fence dashed.
    """
    from matplotlib.colors import to_rgb

    r0, r1, c0, c1 = win
    ext = extent(win)
    if year is not None:
        base = truecolor(year, win)
        ax.imshow(base * 0.35 + np.array(to_rgb(fade_to)) * 0.65, extent=ext,
                  interpolation="nearest")
    region = rasters[lake]["region"][r0:r1, c0:c1]
    cls = classmap[r0:r1, c0:c1].astype(float)
    cls[(~region) | (cls == C.IGNORE)] = np.nan
    ax.imshow(np.ma.masked_invalid(cls), cmap=CLASS_CMAP, vmin=-0.5, vmax=3.5,
              extent=ext, interpolation="nearest")
    poly = rasters[lake]["geom"]
    outline(ax, poly, **{**SHORE_STYLE, "color": line_color})
    outline(ax, poly.buffer(fence_m), **{**FENCE_STYLE, "color": line_color})
    _bare_axes(ax)


def composition_ha(classmap: np.ndarray, lake: str, rasters) -> dict:
    """Hectares of each class inside the lake footprint."""
    inside = rasters[lake]["inside"]
    px = C.pixel_area_ha()
    return {name: float((inside & (classmap == k)).sum()) * px
            for k, name in enumerate(C.CLASS_NAMES)}


def comp_line(ha: dict) -> str:
    return (f"water {ha['water']:.0f}  ·  hyacinth {ha['vegetation']:.0f}  ·  "
            f"bare {ha['bare']:.0f}  ·  built {ha['built']:.0f} ha")


def legend_handles():
    return [Patch(facecolor=c, edgecolor="none", label=l)
            for c, l in zip(CLASS_COLORS, CLASS_LABELS)]


def _bare_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")
