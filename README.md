# Predictive DA — Exploratory Data Analysis

Data-provenance and exploratory analysis for the predictive-analytics project
shortlist in `deep-research-report.md`.

It answers two questions:

1. **Where and what** — which datasets does each of the 12 candidate projects
   actually need, who publishes them, how obtainable are they, and what do they
   cover in space and time?
2. **What do they look like** — pulled live and plotted, split into
   **satellite** and **non-satellite** evidence, then joined back together.

Every figure is computed from data fetched at run time. Nothing is mocked.

---

## Results

| Output | What it is |
|---|---|
| `reports/eda_simple.html` | **The main EDA — start here.** Five sections: data sources, exact datasets selected, raw data, my implementation plan, and a look at the final dataset. Written in plain language for submission. |
| `reports/dashboard.html` | Interactive version — filterable source explorer, lake selector, year slider over the imagery |
| `reports/eda_report.html` | Detailed version — all 25 figures with full analytical write-up |
| `outputs/figures_simple/` | The 7 simple figures |
| `outputs/figures/{light,dark}/` | All 25 detailed figures as PNG, in both themes |
| `outputs/tables/` | CSV versions of every table |
| `data/processed/features_lake_year.csv` | Analysis-ready table: 6 lakes × 7 years, satellite state joined to non-satellite drivers |
| `data/processed/dataset_inventory.csv` | The 44-source catalogue |

**Headline numbers:** 44 sources catalogued (17 satellite / 27 non-satellite),
109 project↔source links, 14/16 endpoints live, 7 cloud-free Sentinel-2 scenes,
420,080 building footprints, 23,376 city-days of weather.

---

## Data actually pulled

All free, all keyless, all fetched live.

| Source | Modality | What was pulled |
|---|---|---|
| **Sentinel-2 L2A** (Earth Search STAC → AWS COGs) | Satellite | 7 cloud-free dry-season scenes 2019–2025, MGRS 43PGQ, bands blue/green/red/NIR/SWIR16/SCL, windowed to a 22 × 17 km AOI at 20 m |
| **NASA GIBS** (WMS) | Satellite | MODIS Terra true colour, VIIRS Day–Night Band, MODIS land-surface temperature |
| **NASA POWER** | Satellite-derived | Daily rainfall, temperature, humidity, wind — 8 cities × 2,922 days |
| **OpenStreetMap** (Overpass) | Non-satellite | 672 waterbodies, 51,123 roads, 420,080 buildings, 1,232 drains, 4,588 civic assets |
| **Endpoint probe** | Non-satellite | Live HTTP status of 16 catalogued sources |

The six study lakes — Bellandur, Varthur, Hebbal, Madiwala, Ulsoor (mapped in OSM
under its Kannada name *Halasuru*) and Sankey Tank — use OSM multipolygon
relations as fixed reference footprints.

---

## Running it

```bash
python src/00_dataset_inventory.py     # catalogue -> data/processed/*.csv
python src/01_fetch_nonsatellite.py    # OSM + NASA POWER + endpoint probe   (~4 min)
python src/02_fetch_satellite.py       # Sentinel-2 + GIBS                   (~8 min)
python src/03_eda_provenance.py        # F01-F07   where & what
python src/04_eda_nonsatellite.py      # F08-F13   non-satellite
python src/05_eda_satellite.py         # F14-F21   satellite
python src/06_integrate_features.py    # F22-F25   joined + feature table
python src/07_build_report.py          # reports/eda_report.html
python src/08_build_dashboard.py       # reports/dashboard.html  (interactive)
python src/09_simple_eda.py            # 7 simple figures
python src/10_build_simple_report.py   # reports/eda_simple.html  (main deliverable)
```

Everything is cached under `data/raw/`, so re-runs are cheap. Add `--inline` to
step 07 for a single self-contained HTML file.

Requires `pandas numpy matplotlib rasterio shapely pyproj requests pillow`.

---

## What the analysis found

**The proposed encroachment label would mostly produce false positives.** The research
report suggested labelling encroachment as *"lake area decreased >10% over five years"*.
Bellandur's open water runs 15.7 ha (2019) → 0.4 ha (2021) → 107.9 ha (2025), and Varthur
follows the same shape. That rule fires hard on 2021 — but three independent lines of
evidence say encroachment is not what happened: the true-colour scenes show a pale, dry
lake bed; Sentinel-2's own Scene Classification Layer independently classes 161 ha of
Bellandur and 150 ha of Varthur as **bare soil** that year; and the water returns by 2025,
which encroachment does not do. The bed was drained, not built on.

**Most of these lakes are vegetation, not water.** Bellandur holds 260 ha of vegetation
inside a 315 ha footprint in 2019 — hyacinth mat, not land. Reporting open-water area alone
misattributes that to lake loss, so the pipeline reports a water / vegetation / bare split.

**Rainfall explains one lake in six.** Sankey Tank — small, clean, rain-fed — tracks
antecedent rainfall at r = +0.74. The large sewage-fed lakes show no relationship at all
(|r| ≤ 0.29): their level is set by wastewater inflow and management works, not the monsoon.
A blanket rainfall correction would be wrong; the test has to be run per lake.

**Water-area change conflates at least four processes** — drought, drawdown or desilting
works, hyacinth cover, and genuine land filling. Only the last is encroachment.

---

## Two things worth knowing

**The Sentinel-2 reflectance offset.** The AWS archive is reprocessed to baseline
05.x, whose `raster:bands` metadata declares `scale = 0.0001, offset = -0.1`.
Ignoring the offset adds +0.1 reflectance to every band, which collapses MNDWI
over open water from about +0.78 to +0.14 and pushes most genuine water pixels
below the detection threshold — the first run of this pipeline reported near-zero
water at Bellandur for exactly that reason. Raw digital numbers are cached and
scaling is applied at analysis time (`src/s2_common.py`), so a scaling fix never
costs another download.

**Lake relations are multipolygons.** Bellandur is four disjoint basins totalling
317 ha; keeping only the largest ring would discard two-thirds of the lake.
Madiwala has three inner rings (islands) that must be cut out. `_osm_candidates()`
stitches outer members with `linemerge`/`polygonize`, unions them, and subtracts
the inner rings.

---

## Limitations

- **42 rows.** Six lakes over seven years supports exploration, not modelling.
- **OSM is one 2026 snapshot**, so built-up features do not vary by year. OSM
  history extracts or multi-epoch GHSL would fix this.
- **No encroachment ground truth.** Water-area change is a proxy that confounds
  drought, hyacinth cover and genuine land filling — all three are visible in the
  figures.
- **One scene per year** characterises the dry-season minimum, not the annual cycle.

---

## Layout

```
src/
  viz_style.py            validated palette + matplotlib theme (light & dark)
  s2_common.py            Sentinel-2 scaling and spectral indices
  00_dataset_inventory.py the 44-source catalogue
  01_fetch_nonsatellite.py
  02_fetch_satellite.py
  03..06                  the four EDA stages
  07_build_report.py
data/raw/                 cached downloads (OSM JSON, Sentinel-2 NPZ, GIBS PNG)
data/processed/           tidy CSV/GeoJSON
outputs/figures/          light/ and dark/ PNGs
reports/                  the HTML report
```

Figure colour is the validated reference palette from the `dataviz` skill —
categorical hues assigned in fixed order, one-hue sequential ramp for magnitude,
blue↔red diverging with a neutral midpoint for signed indices, status colours
reserved for state. Checked with `validate_palette` in both modes.
