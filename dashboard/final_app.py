"""
dashboard/final_app.py
======================
Lake Encroachment Intelligence - the final dashboard.

Pick any lake in India (search by name or "lat, lon", or open one already
analysed), press Run. The pipeline (src/anylake/pipeline.py) runs as its own
background process and reports every stage live; closing or refreshing the
tab does not stop it. Results open in tabs:

  Overview · Land cover & growth · Future encroachment · Image enhancement ·
  Image -> vector · Model & accuracy · Method & data

    run_final_dashboard.bat                    (double-click), or
    python -m streamlit run dashboard/final_app.py
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anylake import LAKES, OUT, pipeline, search  # noqa: E402
from anylake import fetch as F  # noqa: E402
from anylake.status import read_status  # noqa: E402

st.set_page_config(page_title="Lake Encroachment Intelligence", page_icon="🛰️", layout="wide")

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
CLASS_COLORS = ["#2166ac", "#1b7837", "#c98a1e", "#c82828"]
CLASS_LABELS = ["Open water", "Vegetation / hyacinth", "Bare ground / dry bed", "Built-up"]
NAMES = ["water", "vegetation", "bare", "built"]
GROWTH, FUTURE = "#ff2d2d", "#ff00c8"
IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
LIGHT = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Base map: Esri, Maxar, Earthstar Geographics, (c) OpenStreetMap contributors"
PLOT_CONFIG = {"displayModeBar": False}
ICON = {"done": ("✓", GOOD), "running": ("◐", BLUE), "pending": ("○", MUTED), "failed": ("✕", CRIT)}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1380px;}
.stage {display:flex; gap:.6rem; align-items:baseline; padding:.28rem 0; border-bottom:1px solid #eeede9;}
.stage .ic {width:1.2rem; font-weight:700; text-align:center;}
.stage .lb {flex:0 0 26rem; color:#0b0b0b;}
.stage .nt {color:#52514e; font-size:.86rem; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;}
.kpi {border:1px solid #e6e5e0; border-radius:8px; padding:.7rem .9rem; background:#fcfcfb; height:100%;}
.kpi .v {font-size:1.5rem; font-weight:650; color:#0b0b0b; line-height:1.2;}
.kpi .l {font-size:.8rem; color:#52514e;}
.kpi .s {font-size:.78rem; color:#8a8984;}
.note {color:#52514e; font-size:.85rem;}
.chip {display:inline-flex; align-items:center; gap:.35rem; margin-right:1rem; font-size:.86rem; color:#52514e;}
.chip i {width:.85rem; height:.85rem; border-radius:3px; display:inline-block; border:1px solid #bbb;}
[data-testid="stHeaderActionElements"] {display:none;}
</style>""", unsafe_allow_html=True)


# ==========================================================================
# Helpers
# ==========================================================================
def mmss(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def kpi(col, value, label, sub=""):
    col.markdown(f"<div class='kpi'><div class='v'>{value}</div><div class='l'>{label}</div>"
                 f"<div class='s'>{sub}</div></div>", unsafe_allow_html=True)


def chips(items):
    st.markdown("".join(f"<span class='chip'><i style='background:{c}'></i>{t}</span>" for c, t in items),
                unsafe_allow_html=True)


def alive(pid) -> bool:
    try:
        return bool(pid) and psutil.pid_exists(int(pid)) and psutil.Process(int(pid)).is_running()
    except psutil.Error:
        return False


def layout(fig, h=300, title=""):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=36 if title else 10, b=10),
                      title=dict(text=title, font=dict(size=14, color=INK)), plot_bgcolor="#fcfcfb",
                      paper_bgcolor="#ffffff", font=dict(color=INK2, size=12),
                      legend=dict(orientation="h", y=-0.22), hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(gridcolor="#eeede9", zeroline=False)
    fig.update_yaxes(gridcolor="#eeede9", zeroline=False)
    return fig


def f3(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def known_lakes() -> dict[str, str]:
    """Every registered lake: slug -> label. Analysed ones first."""
    out = {}
    for meta in sorted(LAKES.glob("*/meta.json")):
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:                                         # noqa: BLE001
            continue
        done = (OUT / m["slug"] / "summary.json").exists()
        place = ", ".join(x for x in (m.get("district"), m.get("state")) if x)
        out[m["slug"]] = (not done, f"{m['name']}" + (f" · {place}" if place else "")
                          + ("" if done else " · not analysed yet"))
    return {k: v[1] for k, v in sorted(out.items(), key=lambda kv: (kv[1][0], kv[1][1]))}


def launch(slug: str) -> None:
    d = OUT / slug
    d.mkdir(parents=True, exist_ok=True)
    log = open(d / "process.log", "w", encoding="utf-8")
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8")
    p = subprocess.Popen([sys.executable, "-u", "-m", "anylake.run", slug], cwd=str(ROOT), stdout=log,
                         stderr=subprocess.STDOUT, env=env, creationflags=flags)
    now = time.time()                    # provisional status until the run writes its own
    (d / "status.json").write_text(json.dumps({
        "slug": slug, "name": slug, "state": "running", "pid": p.pid, "started": now, "updated": now,
        "finished": None, "error": None, "log_tail": ["starting ..."],
        "stages": [{"key": k, "label": v, "state": "pending", "started": None, "ended": None,
                    "seconds": None, "detail": ""} for k, v in pipeline.STEP_NAMES.items()]}), encoding="utf-8")


# ==========================================================================
# Header and lake choice
# ==========================================================================
st.markdown("## Lake encroachment intelligence")
st.markdown("<div class='note'>Choose any lake in India. The system finds its outline, downloads seven years "
            "of Sentinel-2 imagery in all 12 bands, enhances it, maps land cover with a U-Net on raw and on "
            "enhanced bands, turns the image into vectors, checks the 30 m / 75 m geo-fence, and forecasts where "
            "built-up land may appear in the next three years.</div>", unsafe_allow_html=True)
st.write("")

if not pipeline.ready():
    st.warning("The model library has not been built yet. Run this once (about 35 minutes on a CPU), then reload:")
    st.code("python src/19_anylake_library.py", language="bash")
    st.stop()

c1, c2, c3 = st.columns([1.5, 2.2, 1.0])
with c1:
    with st.form("search", border=False):
        q = st.text_input("Search a lake in India", placeholder="Hussain Sagar · Upper Lake Bhopal · 17.42, 78.47")
        go_search = st.form_submit_button("Search", icon=":material/search:", use_container_width=True)
if go_search and q.strip():
    try:
        with st.spinner("Searching OpenStreetMap ..."):
            hits = [search.public(h) for h in search.search(q)]
        st.session_state["hits"] = hits
        if hits:
            st.session_state["slug"] = hits[0]["slug"]
        else:
            st.session_state["flash"] = "No lake found in India for that search. Add the city or state, or give coordinates."
    except Exception as e:                                        # noqa: BLE001
        st.session_state["flash"] = f"Search failed: {e}"

hits = {h["slug"]: h for h in st.session_state.get("hits", [])}
options = {s: f"🔎 {h['name']} · {', '.join(x for x in (h['district'], h['state']) if x)} · "
              f"{h['area_ha']:,.0f} ha" for s, h in hits.items()}
for s, label in known_lakes().items():
    options.setdefault(s, label)
if not options:
    st.info("Search for a lake to begin.")
    st.stop()
keys = list(options)
cur = st.session_state.get("slug")
with c2:
    slug = st.selectbox("Lake", keys, index=keys.index(cur) if cur in keys else 0, format_func=options.get)
st.session_state["slug"] = slug
if msg := st.session_state.pop("flash", None):
    st.info(msg)

meta_path = LAKES / slug / "meta.json"
status = read_status(slug)
running = bool(status and status.get("state") == "running" and alive(status.get("pid")))
with c3:
    st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    if running:
        if st.button("Stop run", use_container_width=True, icon=":material/stop:"):
            try:
                p = psutil.Process(int(status["pid"]))
                for ch in p.children(recursive=True):
                    ch.kill()
                p.kill()
            except psutil.Error:
                pass
            time.sleep(0.5)
            st.rerun()
    else:
        label = "Run again" if (OUT / slug / "summary.json").exists() else "Run analysis"
        if st.button(label, type="primary", use_container_width=True, icon=":material/play_arrow:"):
            if not meta_path.exists():
                F.register(hits[slug])
            launch(slug)
            time.sleep(1.0)
            st.rerun()

if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cached = sum((LAKES / slug / f"s2_{y}.npz").exists() for y in range(2019, 2026))
    g = meta["grid"]
    if meta.get("lat") is None:
        from shapely.geometry import shape as _shape

        _c = _shape(meta["geometry"]).representative_point()
        meta["lat"], meta["lon"] = round(_c.y, 5), round(_c.x, 5)
    st.markdown(f"<div class='note'><b>{meta.get('display') or meta['name']}</b> &middot; centre "
                f"{meta['lat']}, {meta['lon']} &middot; grid {g['shape'][1]} x {g['shape'][0]} "
                f"pixels at 20 m ({g['shape'][1] * 0.02:.1f} x {g['shape'][0] * 0.02:.1f} km), {g['crs']}"
                f"{' (30 km window, the lake is larger)' if g.get('clipped') else ''} &middot; imagery cached "
                f"for {cached}/7 years{' &middot; one of the 6 training lakes' if meta.get('library') else ''}"
                f"</div>", unsafe_allow_html=True)
elif slug in hits:
    h = hits[slug]
    st.markdown(f"<div class='note'><b>{h['display']}</b> &middot; centre {h['lat']}, {h['lon']} &middot; "
                f"{h['area_ha']:,.0f} ha &middot; not downloaded yet</div>", unsafe_allow_html=True)


# ==========================================================================
# Live progress
# ==========================================================================
def progress_panel():
    s = read_status(slug)
    if not s:
        if not (OUT / slug / "summary.json").exists():
            st.info("This lake has not been analysed yet. Press **Run analysis**: the first run downloads "
                    "the imagery (a few minutes, depending on the lake's size), later runs take seconds.")
        return
    live = s["state"] == "running" and alive(s.get("pid"))
    if s["state"] == "done" and st.session_state.get("seen_done") == (slug, s.get("finished")):
        with st.expander(f"Last run: finished in {mmss((s.get('finished') or 0) - s['started'])} "
                         f"({time.strftime('%d %b %H:%M', time.localtime(s['started']))})"):
            render_stages(s, False)
        return
    end = s.get("finished") or (time.time() if live else s.get("updated", time.time()))
    head = {"done": "Finished", "failed": "Stopped with an error",
            "running": "Running" if live else "Interrupted"}[s["state"]]
    stages = s["stages"]
    n_done = sum(x["state"] == "done" for x in stages)
    weight = {"dataset": 6, "forecast": 3}
    tot = sum(weight.get(x["key"], 1) for x in stages)
    got = sum(weight.get(x["key"], 1) for x in stages if x["state"] == "done")
    st.markdown(f"**{head}** &middot; {s['name']} &middot; elapsed {mmss(end - s['started'])} &middot; "
                f"{n_done}/{len(stages)} stages", unsafe_allow_html=True)
    st.progress(min(1.0, got / max(tot, 1)))
    render_stages(s, live)
    if s.get("error"):
        st.error(s["error"])
    with st.expander("Log"):
        st.code("\n".join(s.get("log_tail", [])) or "(empty)", language=None)
    if s["state"] == "done":
        st.session_state["seen_done"] = (slug, s.get("finished"))
        st.rerun()


def render_stages(s, live):
    rows = []
    for x in s["stages"]:
        state = "failed" if (x["state"] == "running" and not live) else x["state"]
        ic, col = ICON[state]
        note = x.get("detail") or ""
        dur = ""
        if x.get("seconds") is not None:
            dur = f"{x['seconds']:.1f} s"
        elif x.get("started") and state == "running":
            dur = mmss(time.time() - x["started"])
        rows.append(f"<div class='stage'><span class='ic' style='color:{col}'>{ic}</span>"
                    f"<span class='lb'>{x['label']}</span><span class='nt'>{note if state == 'running' else ''}</span>"
                    f"<span class='nt' style='margin-left:auto'>{dur}</span></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


st.fragment(progress_panel, run_every=1.0 if running else None)()

S = pipeline.load_summary(slug)
if not S or running:
    st.stop()


# ==========================================================================
# Results: data and map layers
# ==========================================================================
D = OUT / slug
y0, y1, yf = S["first_year"], S["last_year"], S["future_year"]
CH = S["chosen_variant"]
SC = S["unet_scores"]
FI = S.get("forecast") or {}
FZ, GZ = S["future_zones"], S["growth_zones"]
GEO = pd.read_csv(D / "geofence_per_year.csv")
STAMP = (D / "summary.json").stat().st_mtime


@st.cache_resource(show_spinner=False)
def lake_obj(slug_, stamp):
    from anylake.lake import Lake

    return Lake(slug_)


@st.cache_data(show_spinner=False)
def class_maps(slug_, variant, stamp):
    return pipeline.load_maps(slug_, variant)


def png_uri(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def hexa(h, a=255):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)] + [a]


@st.cache_data(show_spinner=False, max_entries=96)
def overlay(slug_, layer, year, ystart, stamp):
    """(data-URI PNG, [[s, w], [n, e]]) for one map layer over the lake window."""
    from rasterio.warp import transform_bounds

    from anylake import geofence as G
    from anylake import render as R

    L = lake_obj(slug_, stamp)
    r0, r1, c0, c1 = L.window()
    region = L.masks["region"][r0:r1, c0:c1]
    h, w = r1 - r0, c1 - c0
    rgba = np.zeros((h, w, 4), "uint8")
    if layer == "satellite":
        rgba[..., :3] = (R.truecolor(L, year, (r0, r1, c0, c1)) * 255).round().astype("uint8")
        rgba[..., 3] = 255
    elif layer == "landcover":
        cls = class_maps(slug_, CH, stamp)[year][r0:r1, c0:c1]
        for k, hcol in enumerate(CLASS_COLORS):
            rgba[region & (cls == k)] = hexa(hcol, 215)
    elif layer == "growth":
        m = class_maps(slug_, CH, stamp)
        g = G.growth_mask(m, L.masks["region"], ystart, year)[r0:r1, c0:c1]
        rgba[g] = hexa(GROWTH, 235)
    elif layer == "risk":
        import matplotlib

        prob = np.load(D / "forecast_prob.npy")[r0:r1, c0:c1]
        cm = matplotlib.colormaps["plasma"]
        col = (cm(np.clip(prob, 0, 1)) * 255).astype("uint8")
        show = (prob >= 0.10) & region
        rgba[show, :3] = col[show, :3]
        rgba[show, 3] = (90 + 165 * np.clip(prob[show], 0, 1)).astype("uint8")
    x0, x1 = L.bounds[0] + c0 * L.res, L.bounds[0] + c1 * L.res
    yt, yb = L.bounds[3] - r0 * L.res, L.bounds[3] - r1 * L.res
    wl, sl, el, nl = transform_bounds(L.crs, "EPSG:4326", x0, yb, x1, yt)
    return png_uri(rgba), [[sl, wl], [nl, el]]


@st.cache_data(show_spinner=False)
def vectors(slug_, stamp):
    """Lake outline, fence lines, growth and predicted polygons as GeoJSON (lon/lat)."""
    from pyproj import Transformer
    from shapely.geometry import mapping
    from shapely.ops import transform

    L = lake_obj(slug_, stamp)
    tf = Transformer.from_crs(L.crs, "EPSG:4326", always_xy=True).transform
    out = {"outline": {"type": "Feature", "properties": {"name": L.name}, "geometry": L.meta["geometry"]}}
    for m in (30, 75):
        out[f"fence_{m}"] = {"type": "Feature", "properties": {"fence": f"{m} m"},
                             "geometry": mapping(transform(tf, L.geom.buffer(m).boundary))}
    for name, f in (("growth", D / "vectors" / f"growth_{y0}_{y1}.geojson"),
                    ("future", D / "vectors" / f"predicted_encroachment_{yf}.geojson")):
        out[name] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    return out


def lake_map(layer_keys: list[str], key: str, year: int, ystart: int = None, height: int = 520):
    titles = {"satellite": f"Sentinel-2 true colour, {year}", "landcover": f"U-Net land cover, {year}",
              "growth": f"Growth {ystart}–{year} (persistent new built-up)",
              "risk": f"Encroachment probability, {yf} forecast"}
    choice = st.radio("Layer", layer_keys, format_func=titles.get, horizontal=True, key=f"r_{key}",
                      label_visibility="collapsed")
    uri, b = overlay(slug, choice, year, ystart if ystart is not None else y0, STAMP)
    (s0, w0), (n0, e0) = b
    zoom = int(min(16, max(9, math.floor(math.log2(360.0 / max(e0 - w0, n0 - s0, 1e-4) * 2.0)))))
    m = folium.Map(location=[(s0 + n0) / 2, (w0 + e0) / 2], zoom_start=zoom, control_scale=True, tiles=None)
    folium.TileLayer(IMAGERY, attr=ESRI_ATTR, name="Satellite base map (Esri)").add_to(m)
    folium.TileLayer(LIGHT, attr=ESRI_ATTR, name="Light base map (Esri)", show=False).add_to(m)
    folium.raster_layers.ImageOverlay(uri, bounds=b, name=titles[choice], opacity=1.0 if choice == "satellite" else 0.9,
                                      pixelated=True, interactive=False).add_to(m)
    V = vectors(slug, STAMP)
    folium.GeoJson(V["outline"], name="Lake outline (OpenStreetMap)",
                   style_function=lambda f: {"color": "#00d2ff", "weight": 2.5, "fill": False}).add_to(m)
    folium.GeoJson(V["fence_30"], name="30 m buffer",
                   style_function=lambda f: {"color": "#ffffff", "weight": 1.6, "dashArray": "6 4"}).add_to(m)
    folium.GeoJson(V["fence_75"], name="75 m buffer", show=False,
                   style_function=lambda f: {"color": "#ffe066", "weight": 1.4, "dashArray": "2 4"}).add_to(m)
    if V["future"] and V["future"]["features"]:
        folium.GeoJson(V["future"], name=f"Predicted encroachment {yf} (vector polygons)", show=choice == "risk",
                       style_function=lambda f: {"color": FUTURE, "weight": 1.5, "fillColor": FUTURE,
                                                 "fillOpacity": 0.25},
                       tooltip=folium.GeoJsonTooltip(fields=["area_ha"], aliases=["Predicted area (ha)"])).add_to(m)
    if V["growth"] and V["growth"]["features"]:
        folium.GeoJson(V["growth"], name=f"Observed growth {y0}–{y1} (vector polygons)", show=False,
                       style_function=lambda f: {"color": GROWTH, "weight": 1.5, "fillOpacity": 0.3},
                       tooltip=folium.GeoJsonTooltip(fields=["area_ha"], aliases=["Area (ha)"])).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, height=height, use_container_width=True, returned_objects=[], key=f"m_{key}_{choice}_{year}")
    if choice == "landcover":
        chips(zip(CLASS_COLORS, CLASS_LABELS))
    elif choice == "growth":
        chips([(GROWTH, "New built-up that stays built-up")])
    elif choice == "risk":
        st.markdown("<div style='display:flex;align-items:center;gap:.5rem'><span class='note'>0.1</span>"
                    "<div style='flex:0 0 220px;height:10px;background:linear-gradient(90deg,#0d0887,#7e03a8,"
                    "#cc4778,#f89540,#f0f921);border:1px solid #ddd'></div><span class='note'>1.0 probability of "
                    f"becoming built-up by {yf}</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='note'>Cyan: lake outline · white dashed: 30 m buffer · layers menu (top right) adds "
                "the 75 m buffer and the vector polygons.</div>", unsafe_allow_html=True)


# ==========================================================================
# Results
# ==========================================================================
st.divider()
L_ = S["lake"]
st.markdown(f"### {L_['name']} <span class='note'>({S['area_ha']:,.0f} ha · analysed {S['created']} in "
            f"{S['total_seconds']:.0f} s)</span>", unsafe_allow_html=True)
for w in S.get("warnings", []):
    st.warning(w, icon=":material/warning:")

tabs = st.tabs(["Overview", "Land cover & growth", "Future encroachment", "Image enhancement",
                "Image → vector", "Model & accuracy", "Method & data"])
h0, h1 = S["hectares_inside_first"], S["hectares_inside_last"]
last = GEO.set_index("year").loc[y1]

with tabs[0]:
    k = st.columns(5)
    kpi(k[0], f3(SC[CH]["macro_f1"]), "U-Net macro-F1 on this lake",
        f"raw {f3(SC['raw']['macro_f1'])} · enhanced {f3(SC['enh']['macro_f1'])}"
        + (" · held-out fold" if S.get("library_lake") else " · lake never seen in training"))
    kpi(k[1], f"{h1['water']:,.0f} ha", f"Open water inside the lake, {y1}",
        f"{h1['water'] - h0['water']:+,.1f} ha since {y0}")
    kpi(k[2], f"{S['growth_ha']:,.1f} ha", f"Growth {y0}–{y1} (lake + 500 m)",
        f"{GZ['inside_lake']} ha inside the lake · {GZ['fence_75m']} ha within 75 m")
    kpi(k[3], "–" if S.get("forecast_error") else f"{FZ['inside_lake'] + FZ['fence_75m']:,.1f} ha",
        f"Predicted new built-up by {yf}, within 75 m",
        (f"forecast test ROC-AUC {FI.get('val_auc'):.2f}" if FI.get("val_auc") else "") + " · a prediction")
    kpi(k[4], f"{S['fence'].get('osm_buildings_30m') if S['fence'].get('osm_buildings_30m') is not None else '–'}",
        "OpenStreetMap buildings within 30 m", f"{S['fence'].get('osm_buildings_75m')} within 75 m")
    st.write("")
    a, b = st.columns([1.25, 1])
    with a:
        lake_map(["satellite", "landcover", "growth", "risk"], "ov", y1, y0)
    with b:
        st.markdown("**What the analysis found**")
        lines = [f"- The U-Net ({'enhanced' if CH == 'enh' else 'raw'} bands) maps **{h1['water']:,.0f} ha of open "
                 f"water**, {h1['vegetation']:,.0f} ha of vegetation or hyacinth and {h1['bare']:,.0f} ha of bare "
                 f"bed inside the outline in {y1}; open water changed by {h1['water'] - h0['water']:+,.1f} ha "
                 f"since {y0}."]
        lines.append(f"- **{S['growth_ha']:,.1f} ha** of new built-up appeared between {y0} and {y1} and stayed: "
                     f"{GZ['inside_lake']} ha inside the outline, {GZ['fence_30m']} ha within 30 m, "
                     f"{GZ['fence_75m']} ha within 75 m.")
        lines.append(f"- Built-up mapped inside the 30 m buffer in {y1}: **{last['built_ha_30m']:.1f} ha**; "
                     f"OpenStreetMap shows {S['fence'].get('osm_buildings_30m')} buildings there and "
                     f"{S['fence'].get('osm_buildings_inside')} inside the outline.")
        if not S.get("forecast_error"):
            lines.append(f"- The forecast expects **{FZ['inside_lake'] + FZ['fence_75m']:,.1f} ha** of new built-up "
                         f"within 75 m by {yf} ({FZ['fence_30m']} ha within 30 m), and "
                         f"{FZ['beyond_75m']:,.1f} ha further out. On its unseen test period "
                         f"({FI.get('val_transition')}) it ranked pixels with ROC-AUC {f3(FI.get('val_auc'))}.")
        e = S.get("experiment") or {}
        if e.get("mean_macro_f1"):
            lines.append(f"- Image enhancement: across the six training lakes the enhanced-band U-Net scored "
                         f"{e['mean_macro_f1'].get('enh')} against {e['mean_macro_f1'].get('raw')} for raw bands, "
                         f"so the {'enhanced' if CH == 'enh' else 'raw'} model is used.")
        st.markdown("\n".join(lines))
        gt = GEO[["year", "water_ha_inside", "built_ha_30m", "built_ha_75m"]].rename(columns={
            "year": "Year", "water_ha_inside": "Open water (ha)", "built_ha_30m": "Built-up ≤30 m (ha)",
            "built_ha_75m": "Built-up ≤75 m (ha)"})
        st.dataframe(gt, hide_index=True, use_container_width=True)

with tabs[1]:
    years = S["years"]
    a, b = st.columns([1.25, 1])
    with a:
        yr = st.select_slider("Year", options=years, value=y1, key="lc_year")
        lake_map(["landcover", "satellite", "growth"], "lc", yr, years[0] if yr != years[0] else years[0])
    with b:
        maps = class_maps(slug, CH, STAMP)
        L = lake_obj(slug, STAMP)
        inside = L.masks["inside"]
        rows = [{"year": y, **{n: float((inside & (maps[y] == k_)).sum()) * L.px_ha for k_, n in enumerate(NAMES)}}
                for y in years]
        df = pd.DataFrame(rows)
        fig = go.Figure()
        for n, col, lab in zip(NAMES, CLASS_COLORS, CLASS_LABELS):
            fig.add_trace(go.Scatter(x=df.year, y=df[n], name=lab, mode="lines+markers", line=dict(color=col, width=2)))
        st.plotly_chart(layout(fig, 290, "Inside the lake outline, hectares per class"), use_container_width=True,
                        config=PLOT_CONFIG)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=GEO.year, y=GEO.built_ha_75m, name="within 75 m", marker_color="#e7a3a3"))
        fig.add_trace(go.Bar(x=GEO.year, y=GEO.built_ha_30m, name="within 30 m", marker_color=CLASS_COLORS[3]))
        fig.update_layout(barmode="overlay")
        st.plotly_chart(layout(fig, 250, "Built-up inside the geo-fence, hectares"), use_container_width=True,
                        config=PLOT_CONFIG)
        st.caption("Year-to-year jumps in built-up are often a dry lake bed read as a hard surface; the growth "
                   "layer only counts built-up that appears and then stays in every later year.")

with tabs[2]:
    a, b = st.columns([1.25, 1])
    with a:
        st.image(str(D / f"future_{yf}.png"), use_container_width=True,
                 caption=f"Vector → image: predicted {yf} built-up as polygons drawn onto the {y1} image "
                         f"(white: lake outline, 30 m and 75 m buffers)")
        with st.expander("Interactive map of the forecast"):
            lake_map(["risk", "satellite"], "fc", y1, y0, height=460)
    with b:
        if S.get("forecast_error"):
            st.info(f"No forecast for this lake: {S['forecast_error']}.")
        else:
            st.markdown(f"**Random Forest on per-pixel feature vectors.** Trained on how pixels changed over "
                        f"{', '.join(FI['train_transitions'])} at this lake and the six library lakes, tested on "
                        f"{FI['val_transition']}, which it had not seen, then refitted and projected {y1} → {yf}.")
            st.dataframe(pd.DataFrame({
                "Test on " + FI["val_transition"]: ["ROC-AUC, all lakes", "ROC-AUC, this lake", "Average precision",
                                                    "Precision at the threshold", "Recall at the threshold",
                                                    "Share that became built-up (base rate)"],
                "Value": [FI.get("val_auc"), FI.get("val_auc_this_lake"), FI.get("val_ap"), FI.get("val_precision"),
                          FI.get("val_recall"), FI.get("val_base_rate")]}), hide_index=True, use_container_width=True,
                column_config={"Value": st.column_config.NumberColumn(format="%.3f")})
            st.dataframe(pd.DataFrame({"Zone": ["Inside the lake outline", "Within 30 m", "Within 75 m", "75–500 m"],
                                       f"Predicted new built-up by {yf} (ha)": [FZ["inside_lake"], FZ["fence_30m"],
                                                                               FZ["fence_75m"], FZ["beyond_75m"]]}),
                         hide_index=True, use_container_width=True)
            imp = FI.get("importance", [])[:10][::-1]
            if imp:
                fig = go.Figure(go.Bar(x=[v for _, v in imp], y=[k_ for k_, _ in imp], orientation="h",
                                       marker_color=BLUE))
                st.plotly_chart(layout(fig, 300, "What drives the forecast (Random Forest importance)"),
                                use_container_width=True, config=PLOT_CONFIG)
            st.caption("A projection of the 2019–2025 pattern: where built-up land is likely to appear, not a record "
                       "of construction and not a legal finding.")

with tabs[3]:
    a, b = st.columns(2)
    a.image(str(D / "enh_before.png"), use_container_width=True, caption=f"Before: raw Sentinel-2 true colour, {y1}")
    b.image(str(D / "enh_after.png"), use_container_width=True,
            caption="After: bilateral denoise → CLAHE contrast → unsharp-mask sharpening")
    q = S["enhancement_quality"]
    a, b = st.columns([1, 1.2])
    with a:
        st.markdown("**Image quality, before and after**")
        st.dataframe(pd.DataFrame({
            "Measure": ["Contrast (grey-level std)", "Entropy (bits)", "Sharpness (Laplacian variance)",
                        "Noise estimate (grey levels)"],
            "Before": [q["before"][k_] for k_ in ("contrast", "entropy", "sharpness", "noise")],
            "After": [q["after"][k_] for k_ in ("contrast", "entropy", "sharpness", "noise")]}),
            hide_index=True, use_container_width=True,
            column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("Before", "After")})
        st.markdown(f"**On this lake** the raw-band U-Net scores {f3(SC['raw']['macro_f1'])} macro-F1 and the "
                    f"enhanced-band U-Net {f3(SC['enh']['macro_f1'])}; the two maps agree on "
                    f"{S['agreement_raw_enh'] * 100:.0f}% of pixels.")
    with b:
        e = S.get("experiment") or {}
        if e.get("per_lake_macro_f1"):
            pl = e["per_lake_macro_f1"]
            lk = list(pl["raw"])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=lk, y=[pl["raw"][x] for x in lk], name="Raw bands", marker_color=BLUE))
            fig.add_trace(go.Bar(x=lk, y=[pl.get("enh", {}).get(x) for x in lk], name="Enhanced bands",
                                 marker_color=ORANGE))
            fig.update_yaxes(range=[0.4, 0.9], title_text="macro-F1, held-out lake")
            st.plotly_chart(layout(fig, 300, "Experiment: two U-Nets, identical except for the input"),
                            use_container_width=True, config=PLOT_CONFIG)
            st.caption(f"Mean macro-F1 raw {e['mean_macro_f1'].get('raw')} vs enhanced {e['mean_macro_f1'].get('enh')}"
                       + (f", Wilcoxon p = {e['wilcoxon_p']}" if 'wilcoxon_p' in e else "")
                       + f". Rule fixed before the run: {e.get('rule')}. Enhancement makes pictures clearer to a "
                         "person, but CLAHE rescales every 1.28 km tile on its own, so the same surface gets different "
                         "values in different places and years, which costs the model accuracy.")

with tabs[4]:
    a, b = st.columns([1.2, 1])
    with a:
        @st.cache_data(show_spinner=False)
        def vector_table(p, stamp):
            return pd.read_csv(p)

        T = vector_table(D / f"pixel_vectors_{y1}.csv.gz", STAMP)
        bands = ["coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "nir09",
                 "swir16", "swir22"]
        nm = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
        sig = T[T.unet_class != "none"].groupby("unet_class")[bands].mean()
        fig = go.Figure()
        for cls, col in zip(NAMES, CLASS_COLORS):
            if cls in sig.index:
                fig.add_trace(go.Scatter(x=nm, y=sig.loc[cls].values, name=cls.capitalize(), mode="lines+markers",
                                         line=dict(color=col, width=2.5)))
        fig.update_yaxes(title_text="surface reflectance")
        st.plotly_chart(layout(fig, 320, f"Spectral signature of each class, all 12 bands, {y1}"),
                        use_container_width=True, config=PLOT_CONFIG)
        st.markdown(f"**Pixel feature vectors** · {len(T):,} rows, one per pixel of lake + 500 m. Each row: "
                    f"position, 12 band reflectances, NDWI, MNDWI, NDVI, NDBI, 3×3 texture of NIR and SWIR, and the "
                    f"U-Net class. These vectors feed the forecast model.")
        st.dataframe(T.head(300), use_container_width=True, height=260)
    with b:
        st.markdown("**Raster → vector polygons (GeoJSON)**")
        vt = pd.DataFrame([{"Layer": k_.replace("_", " "), "Polygons": v["polygons"], "Vector area (ha)": v["vector_ha"],
                            "Raster area (ha)": v["raster_ha"]} for k_, v in S["vectors"].items()])
        st.dataframe(vt, hide_index=True, use_container_width=True)
        st.caption("Each connected patch of pixels is traced into a polygon; vector and raster areas match.")
        files = sorted((D / "vectors").glob("*.geojson"))
        pick = st.selectbox("GeoJSON layer", files, format_func=lambda p: p.name)
        st.download_button("Download GeoJSON", pick.read_bytes(), file_name=f"{slug}_{pick.name}",
                           mime="application/geo+json", icon=":material/download:", use_container_width=True)
        st.download_button("Download pixel vectors (CSV)", (D / f"pixel_vectors_{y1}.csv.gz").read_bytes(),
                           file_name=f"{slug}_pixel_vectors_{y1}.csv.gz", icon=":material/download:",
                           use_container_width=True)
        st.download_button(f"Download {yf} forecast map (GeoTIFF)", (D / f"future_{yf}.tif").read_bytes(),
                           file_name=f"{slug}_future_{yf}.tif", icon=":material/download:", use_container_width=True)

with tabs[5]:
    a, b = st.columns([1.2, 1])
    with a:
        tm = OUT / "training_metrics.csv"
        if tm.exists():
            M = pd.read_csv(tm)
            M = M[M.lake.str.startswith("ALL")].copy()
            M["_o"] = M.variant.map({"raw": 0, "enh": 1}) * 10 + M.split.map({"train": 0, "val": 1, "test": 2})
            M = M.sort_values("_o")
            M["Model"] = M.variant.map({"raw": "Raw", "enh": "Enhanced"})
            M["Split"] = M.split.map({"train": "Train (5 lakes)", "val": "Validation (Varthur)",
                                      "test": "Test (held-out, mean of 6)"})
            st.markdown("**U-Net accuracy: training, validation and test**")
            st.dataframe(M[["Model", "Split", "accuracy", "macro_f1", "kappa", "f1_water", "f1_vegetation", "f1_bare",
                            "f1_built"]].rename(columns={"accuracy": "Accuracy", "macro_f1": "Macro-F1", "kappa": "Kappa",
                                                         "f1_water": "F1 water", "f1_vegetation": "F1 veg",
                                                         "f1_bare": "F1 bare", "f1_built": "F1 built"}),
                         hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format="%.3f")
                                        for c in ("Accuracy", "Macro-F1", "Kappa", "F1 water", "F1 veg", "F1 bare",
                                                  "F1 built")})
            st.caption("Scored against seed labels (ESA Scene Classification Layer + OpenStreetMap buildings), "
                       "every year 2019–2025. Test is the honest number: each lake scored by a model that never saw it.")
        st.markdown("**This lake**")
        rows = []
        for v, lab in (("raw", "Raw bands"), ("enh", "Enhanced bands")):
            r = SC[v]
            rows.append({"Model": lab, "Accuracy": r["accuracy"], "Macro-F1": r["macro_f1"], "Kappa": r["kappa"],
                         "F1 water": r["f1_water"], "F1 veg": r["f1_vegetation"], "F1 bare": r["f1_bare"],
                         "F1 built": r["f1_built"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ("Accuracy", "Macro-F1", "Kappa", "F1 water", "F1 veg", "F1 bare", "F1 built")})
        st.caption(SC[CH].get("source", ""))
    with b:
        t = S["timings"]
        fig = go.Figure(go.Bar(x=list(t.values())[::-1], y=[pipeline.STEP_NAMES[k_] for k_ in t][::-1],
                               orientation="h", marker_color=AQUA,
                               text=[f"{v:.1f} s" for v in list(t.values())[::-1]], textposition="outside"))
        fig.update_xaxes(title_text="seconds", range=[0, max(t.values()) * 1.25])
        st.plotly_chart(layout(fig, 360, f"Time taken by each step (total {S['total_seconds']:.0f} s)"),
                        use_container_width=True, config=PLOT_CONFIG)
        st.caption("The first run of a lake is dominated by the download; the imagery is cached, so a re-run "
                   "takes well under a minute on a CPU.")

with tabs[6]:
    st.markdown(f"""
**Study area.** The lake outline from OpenStreetMap plus a 1 km margin, on a 20 m grid in {S['grid']['crs']}.
Everything that is labelled, scored or forecast lies inside the lake and a 500 m collar around it.

**Imagery.** **Earth Search `sentinel-2-l2a`** (Element 84, AWS Open Data): ESA Copernicus Sentinel-2
Level-2A surface reflectance. One clear January–April scene per year, {y0}–{y1}, chosen by cloud over the lake
itself; all 12 bands (B1–B12 with B8A) plus the Scene Classification Layer, resampled to 20 m.

**Image enhancement.** Bilateral filter (denoise) → CLAHE (contrast) → unsharp mask (sharpen), per band.

**Land-cover model.** A U-Net (a convolutional neural network that labels every pixel) with 16 input
channels: 12 bands plus NDWI, MNDWI, NDVI and NDBI. Trained twice on six Bengaluru lakes, on raw and on
enhanced bands, with nested leave-one-lake-out testing. Four classes: open water, vegetation/hyacinth,
bare ground/dry bed, built-up.

**Image → vector.** Per-pixel feature vectors (bands, indices, texture) and GeoJSON polygons traced from
the class maps. **Vector → image.** The predicted polygons rasterised back onto the satellite image.

**Geo-fence.** 30 m (Bengaluru Master Plan / KTCDA) and 75 m (National Green Tribunal) buffers around the
outline, shown as reference distances; other states set their own.

**Forecast.** Random Forest on the pixel vectors plus distances to built-up, shoreline and roads; trained on
2020→23 and 2021→24, tested on 2022→25, projected {y1}→{yf}.

**Limits.** Labels are seeds (ESA SCL + OpenStreetMap building points), not ground truth. The models learned
from Bengaluru, so very different landscapes are a change of region; the warnings at the top flag lakes where
this breaks the built-up class. The forecast is an estimate, not observed construction.
""")
    rows = [{"Year": y, "Date": v["date"], "Scene ID(s)": ", ".join(v["ids"]),
             "Tile cloud %": ", ".join(str(c) for c in v["tile_cloud_pct"]),
             "Cloud over lake": f"{v['lake_cloud_frac']:.1%}", "Download (s)": v.get("seconds")}
            for y, v in sorted(S["scenes"].items())]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Map data © OpenStreetMap contributors (ODbL), via Nominatim and Overpass. Base maps © Esri.")
