# Illegal Lake Encroachment Prediction — Exploratory Data Analysis

Finding out which lakes in Bengaluru are losing their water area, using free
satellite images and open map data.

This repository covers **one project only**. The original research report
(`deep-research-report.md`) shortlisted 12 candidate projects; everything here
belongs to the lake encroachment one and nothing else.

Every number and figure is computed from data downloaded at run time. Nothing is
mocked or copied from a Kaggle file.

---

## Results

| Output | What it is |
|---|---|
| `reports/eda_simple.html` | **Start here.** Five sections — data sources, exact datasets selected, raw data, my implementation plan, and a look at the final dataset. Plain language, ready for submission. |
| `reports/dashboard.html` | Interactive version — filterable source list, lake selector, year slider over the satellite images, rainfall |
| `outputs/figures_simple/` | The 7 simple figures used in the report |
| `outputs/figures/{light,dark}/` | 16 detailed figures, light and dark |
| `data/processed/features_lake_year.csv` | The final dataset: 6 lakes × 7 years, 35 columns |
| `data/processed/dataset_inventory.csv` | The 20 sources for this project |

**Numbers:** 20 catalogued sources (4 actually used), 12 of 13 links live,
7 cloud-free Sentinel-2 scenes, 420,080 building footprints, 42 rows in the
final table.

---

## Data actually downloaded

All free, no login, no payment.

| Source | Type | What was pulled |
|---|---|---|
| **Sentinel-2 L2A** (Earth Search STAC → AWS) | Satellite | 7 cloud-free dry-season scenes 2019–2025, tile 43PGQ, bands B02/B03/B04/B08/B11/SCL over a 22 × 17 km box at 20 m |
| **NASA POWER** | Satellite-derived | Daily rainfall and temperature for Bengaluru, 2,922 days |
| **OpenStreetMap** (Overpass) | Non-satellite | 672 waterbodies, 51,123 roads, 420,080 buildings, 1,232 drains |
| **NASA GIBS** (WMS) | Satellite | MODIS Terra true colour, for regional context |

The other 16 catalogued sources (Landsat, SRTM, CartoDEM, GPM IMERG, GHSL, Global
Urban Footprint, WorldPop, ASCAT, Bhuvan, Census 2011, CPCB water quality, IMD,
data.gov.in, National Wetland Atlas, news scraping, Open-Meteo) are options named
in the research report that can be added later.

The six lakes — Bellandur, Varthur, Hebbal, Madiwala, Ulsoor (mapped in OSM under
its Kannada name *Halasuru*) and Sankey Tank — use OSM multipolygon relations as
fixed reference boundaries.

---

## Running it

```bash
python src/00_dataset_inventory.py     # the 20-source catalogue
python src/01_fetch_nonsatellite.py    # OSM + NASA POWER + link check   (~4 min)
python src/02_fetch_satellite.py       # Sentinel-2 + GIBS               (~8 min)
python src/04_eda_nonsatellite.py      # F08, F10-F13
python src/05_eda_satellite.py         # F14-F21
python src/06_integrate_features.py    # F23-F25 + the final table
python src/08_build_dashboard.py       # reports/dashboard.html
python src/09_simple_eda.py            # the 7 simple figures
python src/10_build_simple_report.py   # reports/eda_simple.html
```

Everything is cached under `data/raw/`, so re-runs are cheap.

Requires `pandas numpy matplotlib rasterio shapely pyproj requests pillow scipy`.

---

## What the analysis found

**The encroachment label suggested in the report would mostly give false
positives.** The report proposed *"lake area decreased >10% over five years"*.
Bellandur's open water runs 15.7 ha (2019) → 0.4 ha (2021) → 107.9 ha (2025), and
Varthur is the same shape. That rule fires hard on 2021 — but the true-colour
image shows a dry lake bed, Sentinel-2's own Scene Classification Layer
independently calls 161 ha of Bellandur and 150 ha of Varthur **bare soil** that
year, and the water is back by 2025. Encroachment does not reverse like that. The
bed was drained, not built on.

**Most of these lakes are plants, not water.** Bellandur holds 260 ha of
vegetation inside a 315 ha boundary in 2019 — water hyacinth, not land. So the
pipeline reports a water / vegetation / bare split instead of water alone.

**Rainfall explains one lake in six.** Sankey Tank — small, clean, rain-fed —
tracks rainfall at r = +0.74. The large sewage-fed lakes show no relationship at
all (|r| ≤ 0.29): their level is set by wastewater inflow and cleaning works, not
the monsoon. A blanket rainfall correction would be wrong.

---

## Two bugs worth knowing about

**The Sentinel-2 reflectance offset.** The AWS archive is reprocessed to baseline
05.x, whose `raster:bands` metadata declares `scale = 0.0001, offset = -0.1`.
Ignoring the offset adds +0.1 to every band, which collapses MNDWI over open
water from about +0.78 to +0.14 and pushes most real water pixels below the
threshold — the first run reported near-zero water at Bellandur for exactly this
reason. Raw digital numbers are cached and scaling is applied at analysis time
(`src/s2_common.py`), so a scaling fix never costs another download.

**Lake relations are multipolygons.** Bellandur is four separate basins totalling
317 ha; keeping only the largest ring would throw away two-thirds of the lake.
Madiwala has three inner rings (islands) that must be cut out. `_osm_candidates()`
stitches the outer members with `linemerge`/`polygonize`, unions them, and
subtracts the inner rings.

---

## Limitations

- **42 rows.** Six lakes over seven years is enough to explore, not to train a
  model. Next step is to run the same code on every lake in Bengaluru.
- **OSM is one 2026 snapshot**, so the building counts do not change by year and
  cannot show growth. OSM history extracts or multi-epoch GHSL would fix this.
- **No real ground truth.** Nobody has supplied a list of encroached lakes, so
  the label is derived from the data and needs manual checking on a few cases.
- **One scene per year** shows the dry-season minimum, not the full year.

---

## Layout

```
src/
  viz_style.py             palette + matplotlib theme (light & dark)
  s2_common.py             Sentinel-2 scaling and spectral indices
  00_dataset_inventory.py  the 20-source catalogue
  01_fetch_nonsatellite.py OSM + NASA POWER + link check
  02_fetch_satellite.py    Sentinel-2 + GIBS
  04, 05, 06               the EDA stages
  08_build_dashboard.py    + dashboard_template.html
  09, 10                   the simple figures and the simple report
data/raw/                  cached downloads (OSM JSON, Sentinel-2 NPZ, GIBS PNG)
data/processed/            tidy CSV / GeoJSON
outputs/figures/           light and dark PNGs
outputs/figures_simple/    the 7 simple figures
reports/                   the two HTML outputs
```

`deep-research-report.md` is kept as the original source material — it is the
input this project came from, and it still describes all 12 candidate projects.
