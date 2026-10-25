"""
anylake/render.py
=================
Pictures for the any-lake page, and the vector -> image step.

Vector -> image: the predicted-encroachment polygons (vectors, in map
coordinates) are rasterised back onto the Sentinel-2 grid with
rasterio.features.rasterize, blended into the true-colour image, and their
outlines are drawn from the polygon coordinates themselves. The lake outline
and the 30 m / 75 m fence rings are drawn the same way, from vectors. The
result is written both as a PNG and as a georeferenced GeoTIFF that lines up
with the satellite image in QGIS.

Colours match the rest of the project (src/dl/viz.py).
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image, ImageDraw

from . import LAKES
from .lake import Lake

sys.path.insert(0, str(LAKES.parents[1] / "src"))
import s2_common as s2c  # noqa: E402

CLASS_COLORS = ["#2166ac", "#1b7837", "#c98a1e", "#c82828"]
CLASS_LABELS = ["Open water", "Vegetation / hyacinth", "Bare ground / dry bed", "Built-up"]
GROWTH = "#ff2d2d"
FUTURE = "#ff00c8"            # predicted encroachment: magenta, distinct from observed red
TARGET_W = 760


def hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], "float32") / 255


def truecolor(lake: Lake, year: int, win) -> np.ndarray:
    r0, r1, c0, c1 = win
    st = lake.stack(year)
    return np.clip(s2c.rgb({k: v[r0:r1, c0:c1] for k, v in st.items()}), 0, 1).astype("float32")


def scale_of(win) -> int:
    return max(1, round(TARGET_W / (win[3] - win[2])))


def _px(lake: Lake, coords, win, k):
    r0, _, c0, _ = win
    out = []
    for x, y, *_ in coords:
        c, r = lake.to_px(x, y)
        out.append(((c - c0) * k, (r - r0) * k))
    return out


def _dashed(draw, pts, color, lw, dash, gap):
    on, left = True, dash
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg, pos = float(np.hypot(x1 - x0, y1 - y0)), 0.0
        while pos < seg - 1e-9:
            step = min(left, seg - pos)
            if on:
                a, b = pos / seg, (pos + step) / seg
                draw.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a),
                           (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)], fill=color, width=lw)
            pos, left = pos + step, left - step
            if left <= 1e-9:
                on, left = not on, (gap if on else dash)


def outline(img: Image.Image, lake: Lake, win, color: str, fences=(30,)) -> Image.Image:
    """Lake outline (solid) and fence rings (dashed), drawn from the vector outline."""
    k = scale_of(win)
    draw = ImageDraw.Draw(img)
    lw = max(2, round(img.width / 320))
    for p in getattr(lake.geom, "geoms", [lake.geom]):
        for ring in [p.exterior, *p.interiors]:
            draw.line(_px(lake, ring.coords, win, k), fill=color, width=lw, joint="curve")
    for i, m in enumerate(fences):
        g = lake.geom.buffer(m)
        for p in getattr(g, "geoms", [g]):
            _dashed(draw, _px(lake, p.exterior.coords, win, k), color, max(1, lw - i),
                    (4 if i == 0 else 2) * lw, (3 if i == 0 else 3) * lw)
    return img


def upscale(arr: np.ndarray, k: int) -> Image.Image:
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype("uint8"))
    return img.resize((img.width * k, img.height * k), Image.NEAREST)


def edge_ring(mask: np.ndarray, k: int) -> Image.Image:
    big = np.kron(mask, np.ones((k, k), bool))
    edge = big & ~(np.roll(big, 1, 0) & np.roll(big, -1, 0) & np.roll(big, 1, 1) & np.roll(big, -1, 1))
    return Image.fromarray(edge.astype("uint8") * 255)


# ==========================================================================
# Images
# ==========================================================================
def plain(arr: np.ndarray, lake: Lake, win, color="#ffffff") -> Image.Image:
    return outline(upscale(arr, scale_of(win)), lake, win, color)


def class_map(lake: Lake, classmap: np.ndarray, year: int, win) -> Image.Image:
    r0, r1, c0, c1 = win
    arr = truecolor(lake, year, win) * 0.35 + 0.65
    reg = lake.masks["region"][r0:r1, c0:c1]
    cls = classmap[r0:r1, c0:c1]
    for i, h in enumerate(CLASS_COLORS):
        arr[reg & (cls == i)] = hex_rgb(h)
    return outline(upscale(arr, scale_of(win)), lake, win, "#0b0b0b")


def with_mask(lake: Lake, year: int, mask: np.ndarray, win, color=GROWTH) -> Image.Image:
    r0, r1, c0, c1 = win
    arr = truecolor(lake, year, win).copy()
    g = mask[r0:r1, c0:c1]
    arr[g] = arr[g] * 0.25 + hex_rgb(color) * 0.75
    k = scale_of(win)
    img = outline(upscale(arr, k), lake, win, "#ffffff")
    img.paste(Image.new("RGB", img.size, "#ffffff"), mask=edge_ring(g, k))
    return img


def vector_in_image(lake: Lake, year: int, polys, win, color=FUTURE, alpha=0.6):
    """
    Burn vector polygons into the satellite image. Returns (PIL image at
    display size, uint8 RGB array at grid resolution for the GeoTIFF).
    """
    from rasterio.features import rasterize

    r0, r1, c0, c1 = win
    base = truecolor(lake, year, win)
    burned = (rasterize(((p, 1) for p in polys), out_shape=lake.shape, transform=lake.transform,
                        all_touched=False).astype(bool) if polys else np.zeros(lake.shape, bool))
    b = burned[r0:r1, c0:c1]
    arr = base.copy()
    arr[b] = arr[b] * (1 - alpha) + hex_rgb(color) * alpha
    k = scale_of(win)
    img = outline(upscale(arr, k), lake, win, "#ffffff", fences=(30, 75))
    draw = ImageDraw.Draw(img)
    lw = max(2, round(img.width / 380))
    for p in polys:
        for g in getattr(p, "geoms", [p]):
            draw.line(_px(lake, g.exterior.coords, win, k), fill=color, width=lw, joint="curve")
    return img, (np.clip(arr, 0, 1) * 255).round().astype("uint8")


def write_geotiff(rgb: np.ndarray, lake: Lake, win, path) -> None:
    import rasterio
    from rasterio.windows import Window, transform as wtransform

    r0, r1, c0, c1 = win
    tf = wtransform(Window(c0, r0, c1 - c0, r1 - r0), lake.transform)
    with rasterio.open(path, "w", driver="GTiff", height=r1 - r0, width=c1 - c0, count=3,
                       dtype="uint8", crs=lake.crs, transform=tf, compress="deflate") as dst:
        for i in range(3):
            dst.write(rgb[:, :, i], i + 1)


def side_by_side(a: Image.Image, b: Image.Image, gap: int = 12) -> Image.Image:
    h = max(a.height, b.height)
    out = Image.new("RGB", (a.width + b.width + gap, h), "#ffffff")
    out.paste(a, (0, 0))
    out.paste(b, (a.width + gap, 0))
    return out
