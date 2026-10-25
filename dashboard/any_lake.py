"""
dashboard/any_lake.py
=====================
Page 2 of the Lake Encroachment Monitor: any lake in India.

Search a lake by name (or "lat, lon"), then one button runs the whole
pipeline (src/anylake/pipeline.py):
  Sentinel-2, all 12 bands, 2019-2025 -> image enhancement -> U-Net on raw and
  on enhanced bands -> image -> vector -> geo-fence -> 3-year forecast ->
  vector -> image.
Every step is timed. Lakes already analysed open instantly from
outputs/anylake/<slug>/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anylake import OUT, pipeline, search  # noqa: E402
from anylake.render import CLASS_COLORS, CLASS_LABELS, FUTURE, GROWTH  # noqa: E402

INK2, MUTED = "#52514e", "#898781"
NAMES = ["water", "vegetation", "bare", "built"]
st.markdown(f"""
<style>
.block-container {{padding-top: 1.5rem; max-width: 1350px}}
.legend {{display:flex; flex-wrap:wrap; gap:.3rem 1.1rem; font-size:.9rem; color:{INK2}; margin:.2rem 0}}
.legend span {{display:inline-flex; align-items:center; gap:.4rem}}
.legend i {{width:.85rem; height:.85rem; border-radius:3px; display:inline-block}}
.note {{font-size:.82rem; color:{MUTED}}}
</style>
""", unsafe_allow_html=True)


def legend(items) -> None:
    chips = "".join(f"<span><i style='background:{c}'></i>{t}</span>" for c, t in items)
    st.markdown(f"<div class='legend'>{chips}</div>", unsafe_allow_html=True)


def analysed() -> dict[str, str]:
    out = {}
    for p in sorted(OUT.glob("*/summary.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
            out[p.parent.name] = f"{s['lake']['name']} ({s['lake'].get('state') or ''})".replace(" ()", "")
        except Exception:                                         # noqa: BLE001
            pass
    return out


def f1(x):
    return "–" if x is None else f"{x:.3f}"


# ==========================================================================
# Header, search
# ==========================================================================
st.markdown("## Any lake in India")
st.caption("Search a lake · Sentinel-2, all 12 bands, 2019–2025 · image enhancement · "
           "U-Net on raw and enhanced bands · geo-fence · 3-year encroachment forecast")

if not pipeline.ready():
    st.warning("The model library has not been built yet. Run this once (about an hour on a CPU), "
               "then reload the page:")
    st.code("python src/19_anylake_library.py", language="bash")
    st.stop()

a, b = st.columns([3, 2])
with a:
    with st.form("search", border=False):
        q = st.text_input("Lake name, or coordinates as lat, lon",
                          placeholder="e.g. Hussain Sagar · Upper Lake Bhopal · 17.4239, 78.4738")
        go = st.form_submit_button("Search", icon=":material/search:")
    if go and q.strip():
        try:
            with st.spinner("Searching OpenStreetMap ..."):
                st.session_state["al_hits"] = [search.public(h) for h in search.search(q)]
        except Exception as e:                                    # noqa: BLE001
            st.session_state["al_hits"] = []
            st.error(f"Search failed: {e}")
    hits = st.session_state.get("al_hits")
    if hits == []:
        st.info("No lake found in India for that search. Add the city or state, or give coordinates.")
    if hits:
        i = st.selectbox("Matches", range(len(hits)),
                         format_func=lambda k: f"{hits[k]['name']} · {hits[k]['district']}, "
                                               f"{hits[k]['state']} · {hits[k]['area_ha']:,.0f} ha · "
                                               f"({hits[k]['lat']:.4f}, {hits[k]['lon']:.4f})".replace(" · , ", " · "))
        if st.button("Analyse this lake", type="primary", icon=":material/play_arrow:"):
            rec = hits[i]
            with st.status(f"Analysing {rec['name']} ...", expanded=True) as status:
                lines = st.empty()
                done = []

                def on_step(key, sec):
                    done.append(f"✓ {pipeline.STEP_NAMES[key]} — {sec:.1f} s")
                    lines.markdown("\n\n".join(done))

                try:
                    pipeline.run(rec, log=lambda *x: None, on_step=on_step)
                    status.update(label=f"{rec['name']}: done", state="complete", expanded=False)
                    st.session_state["al_slug"] = rec["slug"]
                except Exception as e:                            # noqa: BLE001
                    status.update(label=f"Failed: {e}", state="error")
with b:
    prev = analysed()
    if prev:
        keys = list(prev)
        cur = st.session_state.get("al_slug")
        pick = st.selectbox("Or open a lake already analysed", keys,
                            index=keys.index(cur) if cur in keys else None,
                            format_func=prev.get, placeholder="Choose a lake")
        if pick:
            st.session_state["al_slug"] = pick

slug = st.session_state.get("al_slug")
s = pipeline.load_summary(slug) if slug else None
if not s:
    st.stop()

# ==========================================================================
# Results
# ==========================================================================
d = OUT / slug
L = s["lake"]
y0, y1, yf = s["first_year"], s["last_year"], s["future_year"]
sc, ch = s["unet_scores"], s["chosen_variant"]
st.divider()
st.markdown(f"### {L['name']}")
st.caption(f"{L.get('display') or ''} · {s['area_ha']:,.0f} ha · centre {L['lat']}, {L['lon']}"
           + (" · analysed in a 30 km window (lake is larger)" if s["grid"].get("clipped") else "")
           + (" · one of the 6 training lakes: maps come from the fold that held it out"
              if s.get("library_lake") else " · never seen by the model during training"))

for w in s.get("warnings", []):
    st.warning(w, icon=":material/warning:")

fz, gz = s["future_zones"], s["growth_zones"]
last = pd.read_csv(d / "geofence_per_year.csv").set_index("year")
m = st.columns(4)
m[0].metric("Total time", f"{s['total_seconds']:.0f} s", border=True,
            help="All steps for this lake on the CPU. The download is cached, so a re-run is faster.")
m[1].metric("U-Net accuracy, raw → enhanced", f"{f1(sc['raw']['macro_f1'])} → {f1(sc['enh']['macro_f1'])}",
            border=True, help="Macro-F1 against this lake's seed labels (ESA SCL + OpenStreetMap buildings). "
                              "The lake was not in the model's training data. 1.0 = perfect.")
m[2].metric(f"Built-up within 30 m of the lake, {y1}", f"{last.loc[y1, 'built_ha_30m']:.1f} ha", border=True,
            help=f"From the U-Net map. OpenStreetMap buildings in the 30 m ring: "
                 f"{s['fence'].get('osm_buildings_30m')}; in the 75 m ring: {s['fence'].get('osm_buildings_75m')}.")
m[3].metric(f"Predicted new built-up by {yf}, within 75 m",
            "–" if s.get("forecast_error") else f"{fz['fence_75m'] + fz['inside_lake']:.1f} ha", border=True,
            help=f"Forecast model. Inside the lake outline {fz['inside_lake']} ha, within 30 m "
                 f"{fz['fence_30m']} ha, within 75 m {fz['fence_75m']} ha. A prediction, not observed construction.")

with st.expander("Time taken by each step", expanded=False):
    st.dataframe(pd.DataFrame({"Step": [pipeline.STEP_NAMES[k] for k in s["timings"]],
                               "Seconds": list(s["timings"].values())}),
                 hide_index=True, use_container_width=False,
                 column_config={"Seconds": st.column_config.NumberColumn(format="%.1f")})

# --- 1 enhancement -----------------------------------------------------------
st.markdown("#### 1 · Image enhancement: before and after")
c1, c2 = st.columns(2)
c1.image(str(d / "enh_before.png"), caption=f"Before: raw Sentinel-2 true colour, {y1}", use_container_width=True)
c2.image(str(d / "enh_after.png"), caption="After: denoise → CLAHE contrast → sharpen", use_container_width=True)
q = s["enhancement_quality"]
qt = pd.DataFrame({"Measure": ["Contrast (grey-level std)", "Entropy (bits)", "Sharpness (Laplacian variance)",
                               "Noise estimate (grey levels)"],
                   "Before": [q["before"][k] for k in ("contrast", "entropy", "sharpness", "noise")],
                   "After": [q["after"][k] for k in ("contrast", "entropy", "sharpness", "noise")]})
e1, e2 = st.columns([1, 1])
with e1:
    st.dataframe(qt, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("Before", "After")})
with e2:
    exp = s.get("experiment") or {}
    if exp.get("per_lake_macro_f1"):
        pl = exp["per_lake_macro_f1"]
        et = pd.DataFrame({"Held-out lake": list(pl["raw"]), "Raw": list(pl["raw"].values()),
                           "Enhanced": [pl.get("enh", {}).get(k) for k in pl["raw"]]})
        st.dataframe(et, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("Raw", "Enhanced")})
        st.caption(f"Two U-Nets, identical except for the input, each lake scored by the model that never saw it. "
                   f"Mean macro-F1: raw {exp['mean_macro_f1'].get('raw')}, enhanced {exp['mean_macro_f1'].get('enh')}"
                   + (f" (Wilcoxon p = {exp['wilcoxon_p']})" if "wilcoxon_p" in exp else "")
                   + f". Used below: **{'enhanced' if ch == 'enh' else 'raw'}** ({exp.get('rule', '')}).")

# --- 2 U-Net + growth ---------------------------------------------------------
st.markdown(f"#### 2 · U-Net result and detected growth, {y0}–{y1}")
h1, h0 = s["hectares_inside_last"], s["hectares_inside_first"]
c1, c2 = st.columns(2)
with c1:
    st.image(str(d / f"growth_{y0}_{y1}.png"), use_container_width=True,
             caption=f"Satellite image {y1}, growth since {y0} marked in red")
    legend([(GROWTH, f"New built-up since {y0}: {s['growth_ha']:.1f} ha "
                     f"({gz['inside_lake']} ha inside the lake, {gz['fence_75m']} ha within 75 m)")])
with c2:
    st.image(str(d / f"landcover_{y1}.png"), use_container_width=True,
             caption=f"U-Net land-cover map, {y1} ({'enhanced' if ch == 'enh' else 'raw'} bands)")
    legend(zip(CLASS_COLORS, [f"{t} {h1[n]:.0f} ha" for t, n in zip(CLASS_LABELS, NAMES)]))
st.markdown("<div class='note'>Solid line: lake outline (OpenStreetMap). Dashed: 30 m buffer. Hectares are "
            "inside the outline. Growth = built-up that appears and stays built-up in every later year.</div>",
            unsafe_allow_html=True)

# --- 3 forecast / vector -> image --------------------------------------------
st.markdown(f"#### 3 · Possible encroachment by {yf} (vector → image)")
c1, c2 = st.columns([3, 2])
with c1:
    st.image(str(d / f"future_{yf}.png"), use_container_width=True,
             caption=f"Predicted built-up by {yf}, as vector polygons drawn onto the {y1} satellite image")
    legend([(FUTURE, f"Predicted new built-up ({s['future_polygons']} polygons)"), ("#ffffff", "Lake outline, "
            "30 m (dashed) and 75 m (fine dashed) buffers")])
with c2:
    fi = s.get("forecast") or {}
    if s.get("forecast_error"):
        st.info(f"No forecast for this lake: {s['forecast_error']}.")
    else:
        st.markdown(f"**Random Forest on per-pixel feature vectors.** Trained on how pixels changed over "
                    f"{', '.join(fi['train_transitions'])} around this lake and the 6 library lakes, then "
                    f"tested on {fi['val_transition']}, which it had not seen:")
        st.dataframe(pd.DataFrame({
            "Test on " + fi["val_transition"]: ["ROC-AUC (all lakes)", "ROC-AUC (this lake)", "Precision", "Recall",
                                                 "Share that became built-up"],
            "Value": [fi.get("val_auc"), fi.get("val_auc_this_lake"), fi.get("val_precision"),
                      fi.get("val_recall"), fi.get("val_base_rate")]}),
            hide_index=True, use_container_width=True,
            column_config={"Value": st.column_config.NumberColumn(format="%.3f")})
        st.dataframe(pd.DataFrame({"Zone": ["Inside the lake outline", "Within 30 m", "Within 75 m",
                                            "75 m – 500 m"],
                                   f"Predicted new built-up by {yf} (ha)": [fz["inside_lake"], fz["fence_30m"],
                                                                           fz["fence_75m"], fz["beyond_75m"]]}),
                     hide_index=True, use_container_width=True)
        st.caption("Strongest inputs: " + ", ".join(k for k, _ in fi.get("importance", [])[:5])
                   + ". A projection of the 2019–2025 pattern, not observed construction or a legal finding.")

# --- 4 image -> vector ---------------------------------------------------------
st.markdown("#### 4 · Image → vector")
c1, c2 = st.columns([3, 2])
with c1:
    st.image(str(d / "spectral_signatures.png"), use_container_width=True,
             caption=f"Spectral signature of each class: mean reflectance in all 12 bands, {y1}")
with c2:
    vt = pd.DataFrame([{"Layer": k.replace("_", " "), "Polygons": v["polygons"], "Vector ha": v["vector_ha"],
                        "Raster ha": v["raster_ha"]} for k, v in s["vectors"].items()])
    st.dataframe(vt, hide_index=True, use_container_width=True)
    st.caption(f"Each region pixel became a vector of {16 + 4} numbers (12 bands, 4 indices, 4 texture) — "
               f"{s['pixels_in_vector_table']:,} rows. Class maps were traced into GeoJSON polygons.")
    files = sorted((d / "vectors").glob("*.geojson"))
    pick = st.selectbox("Download", files, format_func=lambda p: p.name)
    st.download_button("Download GeoJSON", pick.read_bytes(), file_name=f"{slug}_{pick.name}",
                       mime="application/geo+json", icon=":material/download:")
    st.download_button("Download pixel vectors (CSV)", (d / f"pixel_vectors_{y1}.csv.gz").read_bytes(),
                       file_name=f"{slug}_pixel_vectors_{y1}.csv.gz", icon=":material/download:")
    st.download_button(f"Download {yf} map (GeoTIFF)", (d / f"future_{yf}.tif").read_bytes(),
                       file_name=f"{slug}_future_{yf}.tif", icon=":material/download:")

# --- 5 geo-fence + data --------------------------------------------------------
st.markdown("#### 5 · Geo-fence by year")
gt = last.reset_index()[["year", "water_ha_inside", "built_ha_inside", "built_ha_30m", "built_ha_75m"]]
gt.columns = ["Year", "Open water inside (ha)", "Built-up inside (ha)", "Built-up within 30 m (ha)",
              "Built-up within 75 m (ha)"]
st.dataframe(gt, hide_index=True, use_container_width=False)
st.caption("30 m is the Bengaluru Master Plan / KTCDA buffer and 75 m the National Green Tribunal figure; other "
           "states set their own, so both are shown as reference distances.")

with st.expander("Datasets used"):
    rows = [{"Year": y, "Date": v["date"], "Scene ID": ", ".join(v["ids"]),
             "Tile cloud %": ", ".join(str(c) for c in v["tile_cloud_pct"]),
             "Cloud over lake": f"{v['lake_cloud_frac']:.1%}"} for y, v in sorted(s["scenes"].items())]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown("- **Earth Search `sentinel-2-l2a`** (Element 84, AWS Open Data): ESA Copernicus Sentinel-2 "
                "Level-2A surface reflectance. All 12 bands, resampled to 20 m.\n"
                "- **OpenStreetMap** via Nominatim (lake outline) and Overpass (building centroids, roads).\n"
                "- Seed labels: ESA's Scene Classification Layer (SCL) + OpenStreetMap buildings. They are seeds, "
                "not ground truth, and OpenStreetMap building coverage is thin in many parts of India.")
