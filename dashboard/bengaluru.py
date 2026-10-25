"""
dashboard/bengaluru.py
======================
Page 1 of the Lake Encroachment Monitor: runs the trained U-Net and shows

  - how long the model took
  - the Sentinel-2 image, with the growth the model detected marked in red
  - the model's land-cover map and the hectares of each class

    run_dashboard.bat                          (double-click), or
    python -m streamlit run dashboard/app.py

Each lake is mapped by the U-Net from the fold that held it out, so the model
never trained on the lake it is showing. "Growth" is built-up that appears
between the two chosen years and stays built-up in every later year: a one-year
spike is the model mistaking dry lake bed for a hard surface, not a building.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import s2_common as s2c  # noqa: E402
from dl import common as C  # noqa: E402

DL = C.DL
BUILT = C.CLASS_NAMES.index("built")
# Same class colours as every figure in the project (src/dl/viz.py).
CLASS_COLORS = ["#2166ac", "#1b7837", "#c98a1e", "#c82828"]
CLASS_LABELS = ["Open water", "Hyacinth / vegetation", "Bare lake bed", "Built-up"]
GROWTH = "#ff2d2d"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"

st.markdown(f"""
<style>
.block-container {{padding-top: 1.5rem; max-width: 1350px}}
.legend {{display:flex; flex-wrap:wrap; gap:.3rem 1.1rem; font-size:.9rem; color:{INK2}; margin:.2rem 0}}
.legend span {{display:inline-flex; align-items:center; gap:.4rem}}
.legend i {{width:.85rem; height:.85rem; border-radius:3px; display:inline-block}}
.note {{font-size:.82rem; color:{MUTED}}}
</style>
""", unsafe_allow_html=True)


# ==========================================================================
# Data
# ==========================================================================
@st.cache_resource(show_spinner=False)
def rasters() -> dict:
    return C.lake_rasters()


@st.cache_resource(show_spinner=False)
def lakes() -> list[str]:
    return C.lake_order()


@st.cache_resource(show_spinner=False)
def folds() -> list[dict]:
    return C.folds()


@st.cache_resource(show_spinner=False)
def grid_info():
    bounds, shape, _, _ = C.grid()
    return bounds, shape, (bounds[2] - bounds[0]) / shape[1]


@st.cache_resource(show_spinner=False)
def scene(year: int) -> dict:
    return C.load_stack(year)


@st.cache_data(show_spinner=False)
def scores() -> pd.DataFrame:
    return pd.read_csv(DL / "unet" / "per_fold.csv")


def held_out(lake: str) -> dict:
    return next(f for f in folds() if f["test"] == lake)


# ==========================================================================
# Running the U-Net
# ==========================================================================
@st.cache_resource(show_spinner=False)
def imagery():
    """Seven years of Sentinel-2 channels, loaded once. Returns (store, index, seconds)."""
    from dl import dataset as D

    t0 = time.perf_counter()
    store = D.PatchStore(rasters())
    index = D.build_index(store.rasters, verbose=False)
    return store, index, time.perf_counter() - t0


@st.cache_resource(show_spinner=False)
def trainer():
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_unet", ROOT / "src" / "14_train_unet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_model(lake: str) -> dict:
    """Map every year of one lake with the U-Net that never saw it, and time it."""
    import torch

    from dl import dataset as D
    from dl.unet import UNet

    fold = held_out(lake)
    store, index, t_load = imagery()
    t0 = time.perf_counter()
    model = UNet()
    model.load_state_dict(torch.load(DL / "unet" / f"fold{fold['fold']}_best.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    mean, std = D.fit_norm(store, index, fold["train"])   # the fold's own training statistics
    t1 = time.perf_counter()
    maps = {}
    with torch.no_grad():
        for y in C.YEARS:
            maps[y], _ = trainer().predict_region(model, store, lake, y, mean, std, torch.device("cpu"))
    return {"maps": maps, "fold": fold["fold"], "t_infer": time.perf_counter() - t1,
            "t_setup": t1 - t0, "t_load": t_load, "live": True}


def saved_maps(lake: str) -> dict:
    """Fallback when torch is not installed: the maps the training run saved."""
    return {"maps": {y: np.load(DL / "pred" / f"landcover_{y}.npy") for y in C.YEARS},
            "fold": held_out(lake)["fold"], "live": False}


def growth_mask(maps: dict, lake: str, y0: int, y1: int) -> np.ndarray:
    """Built-up in y1 (and every later year) that was not built-up in y0."""
    new = rasters()[lake]["region"] & (maps[y0] != BUILT) & (maps[y1] == BUILT)
    for y in C.YEARS:
        if y > y1:
            new &= maps[y] == BUILT
    return new


def hectares(classmap: np.ndarray, lake: str) -> dict:
    inside = rasters()[lake]["inside"]
    px = C.pixel_area_ha()
    return {n: float((inside & (classmap == k)).sum()) * px for k, n in enumerate(C.CLASS_NAMES)}


# ==========================================================================
# Drawing
# ==========================================================================
def window(lake: str, margin: int = 6):
    _, shape, _ = grid_info()
    reg = rasters()[lake]["region"]
    r = np.where(reg.any(1))[0]
    c = np.where(reg.any(0))[0]
    return (max(0, int(r[0]) - margin), min(shape[0], int(r[-1]) + 1 + margin),
            max(0, int(c[0]) - margin), min(shape[1], int(c[-1]) + 1 + margin))


@st.cache_data(show_spinner=False, max_entries=64)
def truecolor(year: int, win: tuple) -> np.ndarray:
    r0, r1, c0, c1 = win
    return np.clip(s2c.rgb({k: v[r0:r1, c0:c1] for k, v in scene(year).items()}), 0, 1).astype("float32")


def _hex(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype="float32") / 255


def _to_px(coords, win, k):
    bounds, _, res = grid_info()
    r0, _, c0, _ = win
    return [(((x - bounds[0]) / res - c0) * k, ((bounds[3] - y) / res - r0) * k) for x, y, *_ in coords]


def _draw(arr: np.ndarray, lake: str, win, color: str) -> Image.Image:
    """Upscale with crisp pixels, outline the lake (solid) and its 30 m fence (dashed)."""
    k = max(1, round(760 / (win[3] - win[2])))
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype("uint8"))
    img = img.resize((img.width * k, img.height * k), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    lw = max(2, round(img.width / 320))
    geom = rasters()[lake]["geom"]
    for p in getattr(geom, "geoms", [geom]):
        for ring in [p.exterior, *p.interiors]:
            draw.line(_to_px(ring.coords, win, k), fill=color, width=lw, joint="curve")
    for p in getattr(geom.buffer(30), "geoms", [geom.buffer(30)]):
        pts = _to_px(p.exterior.coords, win, k)
        on, left, dash, gap = True, 4 * lw, 4 * lw, 3 * lw
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
    return img


def satellite_with_growth(lake: str, year: int, growth: np.ndarray) -> Image.Image:
    win = window(lake)
    r0, r1, c0, c1 = win
    arr = truecolor(year, win).copy()
    g = growth[r0:r1, c0:c1]
    arr[g] = arr[g] * 0.25 + _hex(GROWTH) * 0.75
    # edge pixels of each growth patch, drawn at the upscaled size as a white ring
    k = max(1, round(760 / (c1 - c0)))
    big = np.kron(g, np.ones((k, k), bool))
    edge = big & ~(np.roll(big, 1, 0) & np.roll(big, -1, 0) & np.roll(big, 1, 1) & np.roll(big, -1, 1))
    img = _draw(arr, lake, win, "#ffffff")
    ring = Image.fromarray(edge.astype("uint8") * 255)
    img.paste(Image.new("RGB", img.size, "#ffffff"), mask=ring)
    return img


def model_map(classmap: np.ndarray, lake: str, year: int) -> Image.Image:
    win = window(lake)
    r0, r1, c0, c1 = win
    arr = truecolor(year, win) * 0.35 + 0.65          # faded image around the mapped area
    region = rasters()[lake]["region"][r0:r1, c0:c1]
    cls = classmap[r0:r1, c0:c1]
    for i, h in enumerate(CLASS_COLORS):
        arr[region & (cls == i)] = _hex(h)
    return _draw(arr, lake, win, INK)


def legend(items) -> None:
    chips = "".join(f"<span><i style='background:{c}'></i>{t}</span>" for c, t in items)
    st.markdown(f"<div class='legend'>{chips}</div>", unsafe_allow_html=True)


# ==========================================================================
# Page
# ==========================================================================
needed = [DL / "unet" / "fold0_best.pt", DL / "pred" / "landcover_2019.npy", DL / "unet" / "per_fold.csv"]
if not all(p.exists() for p in needed):
    st.error("The trained model is missing. Run the pipeline first (README, 'Running it').")
    st.stop()

st.markdown("## Lake Encroachment Monitor")
st.caption("Six Bengaluru lakes · Sentinel-2 satellite images, 2019–2025 · U-Net land-cover model")

names = [l.replace(" Lake", "") for l in lakes()]
a, b = st.columns([1, 2])
lake = lakes()[names.index(a.selectbox("Lake", names))]
y0, y1 = b.select_slider("Detect growth between", options=C.YEARS, value=(2019, 2025))
if y0 == y1:
    st.warning("Pick two different years.")
    st.stop()

runs = st.session_state.setdefault("runs", {})
rerun = st.button("Run the model again", icon=":material/replay:")
if lake not in runs or rerun:
    try:
        first = "imagery" not in st.session_state
        msg = "Loading seven years of Sentinel-2 and running the U-Net ..." if first else "Running the U-Net ..."
        with st.spinner(msg):
            runs[lake] = run_model(lake)
        st.session_state["imagery"] = True
    except ImportError:
        runs[lake] = saved_maps(lake)
r = runs[lake]
maps = r["maps"]
growth = growth_mask(maps, lake, y0, y1)
growth_ha = float(growth.sum()) * C.pixel_area_ha()
h0, h1 = hectares(maps[y0], lake), hectares(maps[y1], lake)
f1 = scores().set_index("test_lake").macro_f1[lake]

m = st.columns(4)
if r["live"]:
    m[0].metric("Time taken by the model", f"{r['t_infer']:.2f} s", border=True,
                help=f"U-Net on all 7 years of this lake, on the CPU. Loading the weights and "
                     f"statistics took {r['t_setup']:.2f} s more; loading the imagery the first time "
                     f"took {r['t_load']:.1f} s (done once).")
else:
    m[0].metric("Time taken by the model", "not run", border=True,
                help="torch is not installed, so these are the maps saved by the training run.")
m[1].metric(f"Growth detected, {y0}–{y1}", f"{growth_ha:.1f} ha", border=True,
            help="New built-up that stays built-up in every later year, inside the lake and the 500 m "
                 "around it that the model maps.")
m[2].metric(f"Open water in {y1}", f"{h1['water']:.1f} ha", f"{h1['water'] - h0['water']:+.1f} ha since {y0}",
            border=True)
m[3].metric("Model accuracy on this lake", f"{f1:.3f}", border=True,
            help="Macro-F1 on this lake when it was held out of training (1.0 = perfect).")

left, right = st.columns(2)
with left:
    st.markdown(f"**Satellite image, {y1}**, growth since {y0} marked in red")
    st.image(satellite_with_growth(lake, y1, growth), use_container_width=True)
    legend([(GROWTH, f"New built-up since {y0}: {growth_ha:.1f} ha")])
with right:
    st.markdown(f"**U-Net result, {y1}**")
    st.image(model_map(maps[y1], lake, y1), use_container_width=True)
    legend(zip(CLASS_COLORS, [f"{t} {h1[n]:.0f} ha" for t, n in zip(CLASS_LABELS, C.CLASS_NAMES)]))
st.markdown("<div class='note'>Solid line: lake outline (OpenStreetMap). Dashed line: 30 m legal buffer. "
            "Hectares are inside the lake outline.</div>", unsafe_allow_html=True)

st.markdown(f"**Inside the lake, {y0} → {y1}**")
tbl = pd.DataFrame({"Class": CLASS_LABELS,
                    f"{y0} (ha)": [h0[n] for n in C.CLASS_NAMES],
                    f"{y1} (ha)": [h1[n] for n in C.CLASS_NAMES],
                    "Change (ha)": [h1[n] - h0[n] for n in C.CLASS_NAMES]})
st.dataframe(tbl, hide_index=True, use_container_width=False,
             column_config={c: st.column_config.NumberColumn(format="%.1f")
                            for c in [f"{y0} (ha)", f"{y1} (ha)"]}
             | {"Change (ha)": st.column_config.NumberColumn(format="%+.1f")})

note = (f"Mapped by the U-Net from fold {r['fold']}, which never trained on {lake}. ")
if y1 == C.YEARS[-1]:
    note += f"{y1} is the last year, so growth marked there cannot yet be checked for persistence."
st.caption(note)
