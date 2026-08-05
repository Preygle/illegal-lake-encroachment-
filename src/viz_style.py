"""
viz_style.py
============
One visual system for every figure in the project.

Implements the validated reference palette: eight categorical slots in fixed
order (never cycled), a single-hue blue sequential ramp, a blue<->red diverging
ramp with a neutral grey midpoint, and a reserved status palette.

Both light and dark variants are "selected" - the dark steps are chosen for the
dark surface, not produced by inverting the light ones.

Validator results (scripts/validate_palette.py, dataviz skill):
  3-slot all-pairs light : PASS  worst CVD dE 9.2, normal-vision 24.0
  3-slot all-pairs dark  : PASS  worst CVD dE 9.4, normal-vision 20.9
  8-slot adjacent light  : PASS  worst CVD dE 9.1, normal-vision 19.6
  Contrast WARN on aqua / yellow / magenta over the light surface
  -> relief rule honoured: every figure ships direct labels or a table view.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------
# Palette definition
# --------------------------------------------------------------------------
THEMES = {
    "light": {
        "surface":   "#fcfcfb",
        "page":      "#f9f9f7",
        "ink":       "#0b0b0b",
        "ink2":      "#52514e",
        "muted":     "#898781",
        "grid":      "#e1e0d9",
        "baseline":  "#c3c2b7",
        "series":    ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                      "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "div_mid":   "#f0efec",
        "seq":       ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                      "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                      "#184f95", "#104281", "#0d366b"],
    },
    "dark": {
        "surface":   "#1a1a19",
        "page":      "#0d0d0d",
        "ink":       "#ffffff",
        "ink2":      "#c3c2b7",
        "muted":     "#898781",
        "grid":      "#2c2c2a",
        "baseline":  "#383835",
        "series":    ["#3987e5", "#d95926", "#199e70", "#c98500",
                      "#d55181", "#008300", "#9085e9", "#e66767"],
        "div_mid":   "#383835",
        "seq":       ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                      "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                      "#184f95", "#104281", "#0d366b"],
    },
}

# Reserved - never used for a data series
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

# Fixed semantic assignment: modality keeps its hue in every chart of the project
MODALITY_SLOT = {"Satellite": 0, "Non-satellite": 1}


class Theme:
    """Resolved colours + colormaps for one mode."""

    def __init__(self, mode: str = "light"):
        self.mode = mode
        t = THEMES[mode]
        self.__dict__.update(t)
        self.status = STATUS
        self.seq_cmap = LinearSegmentedColormap.from_list(f"seq_{mode}", t["seq"])
        # diverging: blue <- neutral -> red, equal arms
        self.div_cmap = LinearSegmentedColormap.from_list(
            f"div_{mode}",
            ["#0d366b", "#256abf", "#3987e5", "#86b6ef", t["div_mid"],
             "#f0a0a0" if mode == "light" else "#a85a5a",
             "#e34948", "#d03b3b", "#8f2020"])
        # water-emphasis ramp for index maps (one hue, light->dark)
        self.water_cmap = self.seq_cmap

    def color(self, i: int) -> str:
        """Categorical slot i, assigned in fixed order - never cycled."""
        if i >= len(self.series):
            raise IndexError(
                f"slot {i}: past 8 series fold into 'Other' or use small multiples")
        return self.series[i]

    def modality(self, name: str) -> str:
        return self.series[MODALITY_SLOT[name]]


def apply(mode: str = "light") -> Theme:
    """Set global rcParams for `mode` and return the resolved Theme."""
    t = Theme(mode)
    mpl.rcParams.update({
        "figure.facecolor":  t.page,
        "savefig.facecolor": t.page,
        "axes.facecolor":    t.surface,
        "axes.edgecolor":    t.baseline,
        "axes.labelcolor":   t.ink2,
        "axes.titlecolor":   t.ink,
        "axes.linewidth":    0.8,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.color":        t.grid,
        "grid.linewidth":    0.7,
        "text.color":        t.ink,
        "xtick.color":       t.muted,
        "ytick.color":       t.muted,
        "xtick.labelcolor":  t.ink2,
        "ytick.labelcolor":  t.ink2,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size":         10,
        "axes.titlesize":    13,
        "axes.titleweight":  "600",
        "axes.titlelocation": "left",
        "axes.titlepad":     10,
        "axes.labelsize":    10,
        "legend.frameon":    False,
        "legend.fontsize":   9.5,
        "lines.linewidth":   2.0,
        "lines.markersize":  6.5,
        "lines.solid_capstyle": "round",
        "figure.dpi":        110,
        "savefig.dpi":       160,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.30,
    })
    return t


def title(ax, headline: str, sub: str | None = None, t: Theme | None = None):
    """Headline + optional deck, both left-aligned above the plot."""
    ax.set_title(headline, loc="left", pad=18 if sub else 10)
    if sub:
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9.5, color=(t.muted if t else "#898781"))


def source(fig, text: str, t: Theme):
    """Provenance line - every figure names where its data came from."""
    fig.text(0.005, 0.004, text, ha="left", va="bottom",
             fontsize=8, color=t.muted)


def save(fig, name: str, outdir, t: Theme):
    """Write <name>.png into the theme's subfolder."""
    from pathlib import Path
    d = Path(outdir) / t.mode
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def bar_ends(ax, orient: str = "v", radius: float = 3.0):
    """Round the data-end of every bar; the baseline end stays square."""
    from matplotlib.patches import FancyBboxPatch, BoxStyle
    for p in list(ax.patches):
        if not hasattr(p, "get_width"):
            continue
        x, y, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
        if w <= 0 or h <= 0:
            continue
        c = p.get_facecolor()
        p.remove()
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=BoxStyle("Round", pad=0, rounding_size=radius),
            mutation_aspect=1, facecolor=c, edgecolor="none",
            linewidth=0, clip_on=False, zorder=2))
