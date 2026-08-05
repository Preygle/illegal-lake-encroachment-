"""
07_build_report.py
==================
Assembles every figure and table into a single HTML EDA report.

    python 07_build_report.py             -> reports/eda_report.html   (links to PNGs)
    python 07_build_report.py --inline    -> reports/eda_report_standalone.html
                                             (figures base64-embedded, self-contained)

The report is theme-aware: each figure ships a light and a dark rendering and the
correct one is shown for the reader's theme.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
REP = ROOT / "reports"
REP.mkdir(parents=True, exist_ok=True)

INLINE = False
MAX_W = 1500          # downscale width when inlining


# --------------------------------------------------------------------------
def img_src(name: str, mode: str) -> str:
    p = FIG / mode / f"{name}.png"
    if not p.exists():
        return ""
    if not INLINE:
        return f"../outputs/figures/{mode}/{name}.png"
    from PIL import Image
    im = Image.open(p)
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)),
                       Image.LANCZOS)
    if im.mode == "RGBA":
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def figure(name: str, caption: str, num: str) -> str:
    l, d = img_src(name, "light"), img_src(name, "dark")
    if not l and not d:
        return f'<p class="missing">[missing figure: {name}]</p>'
    return f"""
<figure class="fig">
  <img class="light" src="{l}" alt="{html.escape(caption)}" loading="lazy">
  <img class="dark"  src="{d or l}" alt="{html.escape(caption)}" loading="lazy">
  <figcaption><span class="fignum">{num}</span> {caption}</figcaption>
</figure>"""


def table(df: pd.DataFrame, cls: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{'' if pd.isna(v) else html.escape(str(v))}</td>" for v in row)
        + "</tr>"
        for row in df.itertuples(index=False))
    return (f'<div class="tablewrap"><table class="{cls}">'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")


def tiles(items) -> str:
    return ('<div class="tiles">' + "".join(
        f'<div class="tile"><div class="tv">{v}</div>'
        f'<div class="tl">{l}</div><div class="ts">{s}</div></div>'
        for v, l, s in items) + "</div>")


# --------------------------------------------------------------------------
CSS = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
.viz-root{
  --surface-1:#fcfcfb; --page:#f9f9f7; --text-primary:#0b0b0b;
  --text-secondary:#52514e; --muted:#898781; --grid:#e1e0d9;
  --border:rgba(11,11,11,.10); --series-1:#2a78d6; --series-2:#eb6834;
  --series-3:#1baf7a; --good:#0ca30c; --critical:#d03b3b;
  background:var(--page); color:var(--text-primary);
}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])) .viz-root{
    --surface-1:#1a1a19; --page:#0d0d0d; --text-primary:#fff;
    --text-secondary:#c3c2b7; --muted:#898781; --grid:#2c2c2a;
    --border:rgba(255,255,255,.10); --series-1:#3987e5; --series-2:#d95926;
    --series-3:#199e70;
  }
}
:root[data-theme="dark"] .viz-root{
  --surface-1:#1a1a19; --page:#0d0d0d; --text-primary:#fff;
  --text-secondary:#c3c2b7; --muted:#898781; --grid:#2c2c2a;
  --border:rgba(255,255,255,.10); --series-1:#3987e5; --series-2:#d95926;
  --series-3:#199e70;
}
body{margin:0}
.viz-root{
  font:15px/1.62 system-ui,-apple-system,"Segoe UI",sans-serif;
  padding:44px 26px 90px; max-width:1220px; margin:0 auto;
}
h1{font-size:2.05rem;line-height:1.18;margin:0 0 10px;letter-spacing:-.019em;font-weight:650}
h2{font-size:1.3rem;margin:62px 0 6px;letter-spacing:-.01em;font-weight:640;
   padding-top:22px;border-top:1px solid var(--border)}
h3{font-size:1.02rem;margin:34px 0 6px;font-weight:640;color:var(--text-primary)}
p{margin:0 0 14px;color:var(--text-secondary);max-width:78ch}
.lede{font-size:1.08rem;color:var(--text-secondary);max-width:80ch}
.meta{font-size:.83rem;color:var(--muted);margin:16px 0 0}
.sectnote{font-size:.9rem;color:var(--muted);margin:0 0 22px;max-width:80ch}
a{color:var(--series-1)}
code{font:.85em ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--surface-1);border:1px solid var(--border);
  border-radius:4px;padding:1px 5px}
/* figures */
.fig{margin:26px 0 34px;padding:0}
.fig img{width:100%;height:auto;display:block;border:1px solid var(--border);
  border-radius:10px;background:var(--surface-1)}
.fig .dark{display:none}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])) .fig .light{display:none}
  :root:where(:not([data-theme="light"])) .fig .dark{display:block}
}
:root[data-theme="dark"] .fig .light{display:none}
:root[data-theme="dark"] .fig .dark{display:block}
figcaption{font-size:.87rem;color:var(--muted);margin-top:11px;max-width:88ch}
.fignum{font-weight:700;color:var(--text-secondary);margin-right:6px}
.missing{color:var(--critical);font-size:.85rem}
/* stat tiles */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));
  gap:12px;margin:30px 0 8px}
.tile{background:var(--surface-1);border:1px solid var(--border);
  border-radius:11px;padding:15px 16px}
.tv{font-size:1.72rem;font-weight:660;letter-spacing:-.02em;line-height:1.1}
.tl{font-size:.83rem;color:var(--text-secondary);margin-top:3px;font-weight:560}
.ts{font-size:.75rem;color:var(--muted);margin-top:2px}
/* tables */
.tablewrap{overflow-x:auto;margin:16px 0 28px;border:1px solid var(--border);
  border-radius:10px;background:var(--surface-1)}
table{border-collapse:collapse;width:100%;font-size:.82rem;
  font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 11px;border-bottom:1px solid var(--border);
  white-space:nowrap}
th{font-weight:640;color:var(--text-primary);position:sticky;top:0;
  background:var(--surface-1)}
td{color:var(--text-secondary)}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:color-mix(in srgb,var(--series-1) 7%,transparent)}
/* callouts */
.note{border-left:3px solid var(--series-1);background:var(--surface-1);
  padding:13px 17px;border-radius:0 9px 9px 0;margin:20px 0;font-size:.92rem;
  color:var(--text-secondary)}
.note.warn{border-left-color:var(--series-2)}
.note b{color:var(--text-primary)}
ul{color:var(--text-secondary);max-width:80ch;padding-left:20px}
li{margin:5px 0}
@media(max-width:640px){.viz-root{padding:26px 15px 60px}h1{font-size:1.55rem}}
"""


def build() -> str:
    ds = pd.read_csv(PROC / "dataset_inventory.csv")
    proj = pd.read_csv(PROC / "projects.csv")
    edges = pd.read_csv(PROC / "project_dataset_edges.csv")
    health = pd.read_csv(PROC / "endpoint_health.csv")
    n_sat = int((ds.modality == "Satellite").sum())
    n_non = int((ds.modality == "Non-satellite").sum())

    feat = pd.read_csv(PROC / "features_lake_year.csv") if (
        PROC / "features_lake_year.csv").exists() else pd.DataFrame()
    area = pd.read_csv(PROC / "lake_water_area.csv") if (
        PROC / "lake_water_area.csv").exists() else pd.DataFrame()
    scenes = pd.read_csv(PROC / "s2_scenes.csv") if (
        PROC / "s2_scenes.csv").exists() else pd.DataFrame()
    osm_inv = pd.read_csv(TAB / "osm_layer_inventory.csv") if (
        TAB / "osm_layer_inventory.csv").exists() else pd.DataFrame()

    n_bldg = int(osm_inv.loc[osm_inv.layer == "Building footprints", "n"].iloc[0]) \
        if len(osm_inv) else 0
    wx = pd.read_csv(PROC / "weather_daily.csv")

    T = tiles([
        (f"{len(ds)}", "Data sources catalogued", "from the research report"),
        (f"{n_sat} / {n_non}", "Satellite / non-satellite", "17 EO, 27 ground"),
        (f"{len(edges)}", "Project ↔ source links", f"across {len(proj)} projects"),
        (f"{int(health.reachable.sum())}/{len(health)}", "Endpoints reachable",
         "live-probed"),
        (f"{len(scenes)}", "Sentinel-2 scenes", "cloud-free, 2019–2025"),
        (f"{n_bldg:,}", "Building footprints", "OpenStreetMap"),
        (f"{len(wx):,}", "City-days of weather", "NASA POWER, 8 cities"),
        (f"{len(feat)}", "Lake-years assembled", "analysis-ready rows"),
    ])

    # --- inventory table (relief for the contrast WARN + the "what" of the ask)
    inv = ds[["dataset", "provider", "modality", "theme", "access_tier",
              "res_m", "year_start", "year_end", "cost"]].copy()
    inv.columns = ["Dataset", "Provider", "Modality", "Theme", "Access",
                   "Res (m)", "From", "To", "Cost"]
    inv["Res (m)"] = inv["Res (m)"].apply(lambda v: "—" if pd.isna(v) else f"{v:,.0f}")

    usage = (edges.groupby(["project", "modality"]).size().unstack(fill_value=0)
             .reset_index())
    for c in ("Satellite", "Non-satellite"):
        if c not in usage:
            usage[c] = 0
    usage["Total"] = usage["Satellite"] + usage["Non-satellite"]
    usage = (usage.merge(proj[["project", "uniqueness", "label_availability",
                               "difficulty", "feasible_1sem"]], on="project")
                  .sort_values("Total", ascending=False))
    usage.columns = ["Project", "Non-satellite", "Satellite", "Total sources",
                     "Uniqueness", "Label availability", "Difficulty", "Feasible?"]
    usage = usage[["Project", "Satellite", "Non-satellite", "Total sources",
                   "Uniqueness", "Label availability", "Difficulty", "Feasible?"]]

    lake_tbl = comp_tbl = ""
    if len(area):
        pv = area.pivot_table(index="lake", columns="year", values="water_ha")
        pv.insert(0, "Footprint (ha)",
                  area.groupby("lake")["ref_ha"].first().round(0))
        lake_tbl = table(pv.round(1).reset_index().rename(columns={"lake": "Lake"}))

        yr0, yr1 = int(area.year.min()), int(area.year.max())
        nice = {"water_ha": "Water", "veg_ha": "Vegetation", "other_ha": "Bare/built"}
        c = area[area.year.isin([yr0, yr1])].pivot_table(
            index="lake", columns="year",
            values=list(nice)).round(1)
        c.columns = [f"{nice[metric]} {yr}" for metric, yr in c.columns]
        c = c[[f"{nice[m]} {y}" for y in (yr0, yr1) for m in nice]]
        comp_tbl = table(c.reset_index().rename(columns={"lake": "Lake"}))

    feat_tbl = ""
    if len(feat):
        show = feat[["lake", "year", "ref_ha", "water_ha", "veg_ha", "water_frac",
                     "mean_mndwi", "rain_365d", "bldg_per_ha_250m",
                     "road_km_500m"]].round(3)
        show.columns = ["Lake", "Year", "Footprint ha", "Water ha", "Veg ha",
                        "Water frac", "Mean MNDWI", "Rain 365d mm",
                        "Bldg/ha 250m", "Road km 500m"]
        feat_tbl = table(show)

    return f"""<div class="viz-root">
<h1>Exploratory data analysis: where the data comes from, and what it shows</h1>
<p class="lede">A data-provenance and exploratory analysis for the predictive-analytics
project shortlist. It answers two questions: <b>which datasets does each candidate
project actually need and where do they come from</b>, and <b>what do those datasets
look like once downloaded</b> — split into satellite and non-satellite evidence.
Every number and figure below is computed from data pulled live, not illustrative.</p>
{T}
<p class="meta">Built from <code>deep-research-report.md</code> · Sentinel-2 imagery via
Earth Search STAC / AWS Open Data · NASA POWER · NASA GIBS · OpenStreetMap (Overpass, ODbL)
· all figures generated by <code>src/03</code>–<code>src/06</code></p>

<h2>1 · What datasets, and where they come from</h2>
<p class="sectnote">The research report names 44 distinct data sources across 12 candidate
projects. Cataloguing them makes the structure visible: no project is a single-file
download, satellite data is frictionless while Indian civic records are not, and the
sources cluster into a small number of reusable building blocks.</p>

{figure("F01_project_dataset_matrix",
        "The full usage matrix. Rows are data sources ordered by how many projects need "
        "them, columns are projects ordered by one-semester feasibility. NASA POWER and "
        "OpenStreetMap are the two universal dependencies — 11 and 9 projects respectively.",
        "Figure 1")}

{figure("F02_sources_per_project",
        "Data-source load per project. Illegal Lake Encroachment needs the most (20 sources); "
        "even the leanest needs four. This is the integration cost the report's feasibility "
        "scores are really measuring.", "Figure 2")}

{figure("F22_source_timeline",
        "Temporal coverage of every source against the 2018–2025 study window. Landsat reaches "
        "back to 1984 and IMD to 1901, but the civic complaint portals — the only real source of "
        "ground-truth labels — mostly start after 2015.", "Figure 3")}

{figure("F03_access_tier",
        "How obtainable each source is. Every satellite source but one is an open API or open "
        "download; the friction is concentrated in scraping, RTI requests and restricted "
        "government systems, all of which are non-satellite.", "Figure 4")}

{figure("F04_resolution_vs_archive",
        "The gridded-data design space: spatial resolution against archive length. There is a "
        "hard trade-off — Sentinel-2 gives 10 m but only since 2015, while Landsat gives 40 years "
        "at 30 m. Weather grids sit four orders of magnitude coarser.", "Figure 5")}

{figure("F07_provider_origin",
        "Who publishes the data. Indian agencies supply most ground truth and demographics; "
        "essentially all Earth observation comes from NASA, ESA and other foreign providers.",
        "Figure 6")}

{figure("F05_study_areas_map",
        "Where the work happens. Eight study cities named across the 12 projects, on a NASA MODIS "
        "true-colour basemap, with the Bengaluru lake-belt window used for the satellite "
        "deep-dive marked in green.", "Figure 7")}

{figure("F06_endpoint_health",
        "Live reachability probe of every catalogued endpoint. 14 of 16 responded; "
        "censusindia.gov.in and the SAC/ISRO wetland portal failed TLS negotiation at the time "
        "of extraction — a real availability risk for anything depending on them.", "Figure 8")}

<h3>Source catalogue</h3>
<p>The complete inventory behind the figures above.</p>
{table(inv)}

<h3>Project requirements</h3>
{table(usage)}

<h2>2 · Non-satellite evidence</h2>
<p class="sectnote">Three non-satellite families were pulled live: NASA POWER daily
meteorology for all eight study cities, the full OpenStreetMap extract over the Bengaluru
lake belt, and the endpoint probe above. Together they supply the context and the
built-environment features that satellite imagery cannot see.</p>

{figure("F08_rainfall_seasonality",
        "Bengaluru rainfall by month and year. The two-peak monsoon regime leaves a reliable "
        "January–April dry window — which is what makes a like-for-like annual comparison of "
        "lake area possible at all.", "Figure 9")}

{figure("F09_monsoon_climatology",
        "Monsoon climatology across all eight study cities on a shared scale. Mumbai receives "
        "roughly six times Ahmedabad's rainfall and peaks a month earlier — any rainfall-driven "
        "model trained in one city will not transfer to another unmodified.", "Figure 10")}

{figure("F10_rainfall_with_scene_dates",
        "Daily rainfall with the seven Sentinel-2 acquisition dates overlaid. Every scene falls "
        "in a dry gap, which is the precondition for treating the year-to-year water-area series "
        "as comparable.", "Figure 11")}

{figure("F11_osm_inventory",
        "The OpenStreetMap extract: five Overpass queries over one 22 × 17 km box returning "
        "477k elements — waterbodies, roads, buildings, drains and civic point assets — with no "
        "login, licence fee or approval.", "Figure 12")}

{figure("F12_building_density",
        "The building footprints falling inside the study box, with the six study-lake outlines. "
        "Dense fabric pressing directly against a shoreline is the visual signature the "
        "encroachment project is trying to formalise.", "Figure 13")}

{figure("F13_encroachment_pressure",
        "Built-up pressure in concentric bands from each shoreline. A tall innermost bar means "
        "development has reached the water's edge; the profile shape distinguishes lakes with a "
        "protective buffer from those without.", "Figure 14")}

<h2>3 · Satellite evidence</h2>
<p class="sectnote">Seven cloud-free dry-season Sentinel-2 L2A scenes (2019–2025), all from
MGRS tile 43PGQ, resampled to a common 20 m grid so that every year is pixel-comparable.
Blue, green, red, NIR and SWIR bands were pulled as windowed reads directly from the AWS
cloud-optimised GeoTIFFs — no bulk download, no account.</p>

<div class="note"><b>A scaling detail that changes the answer.</b> The AWS archive is
reprocessed to Sentinel-2 baseline&nbsp;05.x, whose <code>raster:bands</code> metadata declares
<code>scale&nbsp;=&nbsp;0.0001, offset&nbsp;=&nbsp;−0.1</code>. Ignoring that offset adds +0.1
reflectance to every band, which collapses MNDWI over open water from roughly +0.78 to +0.14 and
pushes most genuine water pixels below the detection threshold. The first run of this pipeline
reported near-zero water at Bellandur and Varthur for exactly that reason. Raw digital numbers
are now cached and scaling is applied at analysis time.</div>

{figure("F14_truecolor_by_year",
        "The Bellandur–Varthur corridor in true colour, one dry-season scene per year. "
        "Yellow outlines are the fixed OpenStreetMap footprints used as reference boundaries.",
        "Figure 15")}

{figure("F15_mndwi_by_year",
        "MNDWI for the same scenes, on a diverging scale centred on the zero water threshold. "
        "Blue is open water, red is dry land and built-up surface.", "Figure 16")}

{figure("F19_spectral_signatures",
        "Why the index works: mean reflectance of water, vegetation and built-up pixels across "
        "the five bands. Water collapses toward zero in NIR and SWIR while built-up surfaces stay "
        "bright — the 1610 nm gap is precisely what MNDWI exploits.", "Figure 17")}

{figure("F18_index_separability",
        "MNDWI against NDWI, both scored on the same scene using Sentinel-2's own Scene "
        "Classification Layer as an independent water reference. The comparison justifies the "
        "choice of index rather than assuming it.", "Figure 18")}

{figure("F16_footprint_composition",
        "What is actually inside each lake footprint each year, split into open water, vegetation "
        "and bare/built surface. This split matters: a hyacinth-covered lake reads as vegetation, "
        "not water, and reporting water area alone would misattribute that to lake loss.",
        "Figure 19")}

{figure("F17_water_area_timeseries",
        "Open-water area per lake, 2019–2025, each panel on its own scale. This series is the "
        "candidate label for the encroachment model — and the next callout is why it cannot be "
        "used naively.", "Figure 20")}

<div class="note warn"><b>The clearest finding, and it is a warning.</b> Bellandur's open water
runs 15.7&nbsp;ha (2019) → 0.4&nbsp;ha (2021) → 107.9&nbsp;ha (2025); Varthur follows the same
shape. A label defined as <i>"water area fell sharply, therefore encroachment"</i> would fire hard
on 2021 — but three independent lines of evidence say that is not what happened. The true-colour
scenes (Figure&nbsp;15) show a pale, dry lake bed in 2021. Sentinel-2's own Scene Classification
Layer independently classes 161&nbsp;ha of Bellandur and 150&nbsp;ha of Varthur as
<b>bare soil</b> that year. And by 2025 the water has returned, which encroachment does not do.
The footprint was drained, not built on. Separately, in every other year most of both beds is
<b>vegetation</b> — 260&nbsp;ha at Bellandur in 2019 — which is hyacinth mat, not land.
Water-area change alone conflates at least four distinct processes: drought, drawdown or
desilting works, hyacinth cover, and genuine land filling. Only the last is encroachment.</div>

<p>This is the central methodological result of the EDA. The research report proposed labelling
encroachment as <i>"lake area decreased &gt;10% over five years"</i>. Applied to these six lakes
that rule would produce mostly false positives. Any usable label needs at minimum: the
water/vegetation/bare split rather than water alone, per-lake testing of the rainfall
relationship rather than a blanket correction (Figure&nbsp;23 shows it holds for one lake in six),
consistent scene timing, and spot validation against high-resolution imagery or municipal records.</p>

<h3>Footprint composition, first and last year (hectares)</h3>
<p>The same numbers behind Figure&nbsp;19, for readers who want the values rather than the shapes.</p>
{comp_tbl}

{figure("F20_water_change_map",
        "Water change between the first and last scene, restricted to the reference footprints: "
        "120 ha gained, 1 ha lost. Run city-wide the same comparison reports thousands of hectares "
        "of spurious gain — the February 2025 scene has a lower sun elevation and building shadows "
        "read as water. Confining the comparison to the lakes removes that artefact.", "Figure 21")}

{figure("F21_gibs_regional_context",
        "Regional context from NASA GIBS — MODIS true colour, VIIRS night lights and MODIS "
        "land-surface temperature. Three sensors and three resolutions from one keyless WMS "
        "endpoint, framing the fine-resolution work.", "Figure 22")}

<h3>Open-water area by lake and year (hectares)</h3>
{lake_tbl}

<h2>4 · Joining the two</h2>
<p class="sectnote">The point of the split is to put the two back together. The final table
carries satellite-derived state and non-satellite drivers side by side for every lake-year.</p>

{figure("F23_rainfall_vs_water",
        "Open water against rainfall in the preceding 365 days, per lake. Only Sankey Tank — small, "
        "clean, rain-fed — actually tracks rainfall (r = +0.74). The large sewage-fed lakes show no "
        "relationship at all (|r| ≤ 0.29), because their level is set by wastewater inflow and "
        "management works rather than the monsoon. That is the opposite of what one would assume, "
        "and it is why the test is worth running per lake.", "Figure 23")}

{figure("F24_pressure_vs_change",
        "Built-up pressure against observed water change. Six lakes is far too small a sample to "
        "model on; the figure is included to show the intended design, and to make the case for "
        "scaling to hundreds of lakes.", "Figure 24")}

{figure("F25_feature_correlation",
        "Correlation structure of the assembled table, with satellite [S] and non-satellite [N] "
        "features marked. Strong within-block correlation shows these features are far from "
        "independent — relevant to both model choice and feature selection.", "Figure 25")}

<h3>Analysis-ready feature table</h3>
<p>One row per lake-year, satellite state joined to non-satellite drivers.
Full file: <code>data/processed/features_lake_year.csv</code>.</p>
{feat_tbl}

<h2>5 · What this establishes, and what it does not</h2>
<div class="note"><b>Established.</b> The data pipeline works end to end on free, keyless
sources: 44 catalogued sources, 14 of 16 endpoints live, seven cloud-free Sentinel-2 scenes,
420k building footprints and 23k city-days of weather, joined into a 42-row analysis-ready
table. The satellite/non-satellite split is real and complementary — imagery supplies lake
state, OSM and NASA POWER supply the drivers.</div>

<div class="note warn"><b>Not established.</b> This is exploratory analysis, not a model.
Specifically:
<ul>
<li><b>Six lakes over seven years is 42 rows.</b> Nothing here supports fitting or validating a
predictive model. The lake belt holds hundreds of waterbodies; scaling the same pipeline to all
of them is the obvious next step.</li>
<li><b>OpenStreetMap is a single 2026 snapshot.</b> Building counts do not vary by year, so the
built-up features cannot yet express change over time. OSM history extracts or the GHSL
multi-epoch layers would fix this.</li>
<li><b>There is still no encroachment ground truth.</b> Water-area change is a proxy. It confounds
drought, hyacinth cover and genuine land filling — Figure 19 and Figure 23 show all three at work.
Validating even a handful of cases against high-resolution imagery or municipal records is
required before any label is trustworthy.</li>
<li><b>One scene per year</b> characterises the dry-season minimum, not the annual cycle. A
denser time series would separate seasonal drawdown from permanent loss.</li>
</ul></div>

<p class="meta">Reproduce with <code>python src/00_dataset_inventory.py</code> →
<code>01_fetch_nonsatellite.py</code> → <code>02_fetch_satellite.py</code> →
<code>03</code>–<code>07</code>. Palette validated with the dataviz
<code>validate_palette</code> six-checks script in both light and dark modes.</p>
</div>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inline", action="store_true",
                    help="base64-embed figures into a standalone file")
    a = ap.parse_args()
    INLINE = a.inline

    body = build()
    out = REP / ("eda_report_standalone.html" if INLINE else "eda_report.html")
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Predictive DA — Exploratory Data Analysis</title>"
           f"<style>{CSS}</style></head><body>{body}</body></html>")
    out.write_text(doc, encoding="utf-8")
    print(f"  {out.relative_to(ROOT)}   {out.stat().st_size/1e6:.2f} MB")
