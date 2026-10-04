# Illegal Lake Encroachment Detection and Predictive Risk Analysis Using Multi-Temporal Satellite Imagery

Six lakes in Bengaluru, seven years of Sentinel-2 imagery (2019–2025), and a
pipeline that separates **three different reasons a lake can lose water** —
desilting, drought, and actual encroachment.

Every number in this repository is computed from data downloaded at run time.
Nothing is mocked and nothing comes from a Kaggle file.

---

## The problem this project is built around

The obvious approach is to measure open water area each year and treat a drop as
a warning. That does not work, because a drop has three explanations and only one
of them is illegal:

| What the satellite sees | Innocent explanation | Guilty explanation |
|---|---|---|
| Water area falls | Lake drained for desilting (silt removal) | Lake bed built on |
| Water area falls | Weak monsoon | Lake bed built on |
| Water low, vegetation high | Water hyacinth mat floating on the lake | — |

Bellandur Lake is the proof. Its open water ran **39.0 ha (2019) → 0.0 ha (2021)
→ 67.8 ha (2025)**. A "water decreased" rule fires hard on 2021 — but the
true-colour image shows a dry bed, the satellite's own Scene Classification
Layer calls it bare soil, `JRC/GSW1_4/YearlyHistory` independently reports 0%
permanent water that year, and the water is back by 2025. **Encroachment does
not reverse.** The bed was drained, not built on.

So the pipeline is organised in three layers, each judged on its own terms:

1. **Land cover** — a deep learning model classifies every pixel into four
   classes, including a *bare lake bed* class. Dry bed = desilting. Built
   surface where the lake should be = encroachment. Two measurements instead of
   one ambiguous number.
2. **Geo-fence** — count buildings inside the statutory buffer zone around each
   lake. This is the encroachment evidence that drought and desilting cannot
   fake.
3. **Risk** — a statistical model on the 42 lake-years, reporting honest
   uncertainty rather than false precision.

---

## The four land-cover classes

Every 20 m pixel inside a lake and its 500 m surrounding band gets one of:

| Class | Meaning |
|---|---|
| **Open water** | The lake surface. |
| **Vegetation** | Inside a lake boundary this is water hyacinth — a floating weed mat, not land. |
| **Bare lake bed** | Dry, exposed bed. The desilting signature. **This class is the point of the redesign.** |
| **Built-up** | Buildings and hard surfaces. The encroachment signature. |

The old pipeline had three classes (water / vegetation / other), which is
exactly why it could never tell a drained lake from a built-on one.

---

## Pipeline

Nine scripts, run in order. Each needs the previous one's output. Everything is
cached, so re-runs are cheap.

```
01 ─ OpenStreetMap + NASA POWER ─┐
02 ─ Sentinel-2 L2A → 20 m grid ─┼─→ 06 join rainfall ─→ 12 seed labels
                                 │                            │
                                 │          ┌─────────────────┴──────────────┐
                                 │          ▼                                ▼
                                 │   13 MNDWI rule + Random Forest    14 U-Net (deep learning)
                                 │          └──────────── same 6 folds ──────┘
                                 │                         │
         17 Earth Engine layers ─┴─────────────────────────┼─→ 15 geo-fence ─→ 16 risk + desilting rule
```

| # | Script | What it does |
|---|---|---|
| 01 | `01_fetch_nonsatellite.py` | Lake outlines, buildings, roads, drains from **OpenStreetMap** (Overpass API); daily rainfall and temperature from **NASA POWER**; checks every source is online. |
| 02 | `02_fetch_satellite.py` | Searches **Sentinel-2 L2A** (collection `sentinel-2-l2a`, Earth Search STAC on AWS Open Data), keeps one near-cloud-free dry-season scene per year from tile `MGRS-43PGQ`, reads 6 bands onto a shared 20 m grid. |
| 06 | `06_integrate_features.py` | Joins rainfall over the 30/90/365 days before each scene date to each lake-year. |
| 12 | `12_make_labels.py` | Builds four-class seed labels from the **Scene Classification Layer** plus OpenStreetMap buildings; exports GeoTIFFs for hand correction in QGIS. |
| 13 | `13_train_rf.py` | The two non-deep baselines: the MNDWI/NDVI threshold rule, and a **Random Forest** pixel classifier. |
| 14 | `14_train_unet.py` | Trains the **U-Net** under nested leave-one-lake-out and writes out-of-fold land-cover maps. |
| 17 | `17_fetch_gee.py` | Google Earth Engine layers: per-year building counts, plus two independent cross-checks. |
| 15 | `15_geofence.py` | Draws the 30 m and 75 m buffers, counts building intrusion, reports land-cover composition per lake-year. |
| 16 | `16_risk_model.py` | Mixed-effects water model, building-growth model, jackknife+ intervals, and the desilting rule. |

### Why 13 runs before 14

If the labels or the folds are broken, the Random Forest exposes it in two
minutes instead of after an hour of training.

---

## Data sources

All free. Only Google Earth Engine needs a (free) sign-in.

| Source | Exact dataset | What is used |
|---|---|---|
| ESA / Copernicus | Sentinel-2 L2A, collection `sentinel-2-l2a` via Earth Search STAC → AWS Open Data | 7 dry-season scenes 2019–2025, tile `MGRS-43PGQ`, bands B02, B03, B04, B08, B11, SCL at 20 m |
| OpenStreetMap | Overpass API | 6 lake boundaries, 446,924 building points, 56,171 roads, 1,366 drains |
| NASA | NASA POWER | 2,922 days of daily rainfall and temperature |
| Google | `GOOGLE/Research/open-buildings-temporal/v1` | Yearly building presence and count inside each buffer, 2016–2023 |
| Google | `GOOGLE/DYNAMICWORLD/V1` | Independent 9-class land cover, matched to the same Sentinel-2 scene |
| EC JRC / Google | `JRC/GSW1_4/YearlyHistory`, `JRC/GSW1_4/GlobalSurfaceWater` | Landsat-based water history 1984–2021, for cross-checking open water |
| NASA | NASA GIBS (WMS) | MODIS true colour, regional context only |

### The six lakes

Boundaries are OpenStreetMap multipolygon relations, used as fixed reference
footprints so the denominator never moves between years.

| Lake | Footprint | Acres | Statutory buffer tier |
|---|---|---|---|
| Bellandur | 315.1 ha | 779 | 30 m (over 100 acres) |
| Varthur | 154.0 ha | 380 | 30 m |
| Madiwala | 82.9 ha | 205 | 30 m |
| Hebbal | 45.4 ha | 112 | 30 m |
| Ulsoor (mapped as *Halasuru*) | 38.1 ha | 94 | under 100 acres — tier must be looked up |
| Sankey Tank | 12.4 ha | 31 | under 100 acres — tier must be looked up |

---

## The models

### U-Net — the deep learning model

A **U-Net** is a convolutional neural network (CNN) that outputs a class for
*every pixel* instead of one label per image. It shrinks the image while learning
patterns, grows it back, and copies fine detail across at each level so edges
stay sharp — which is what matters when the thing being measured is a shoreline.

- **Input:** 8 channels — 5 reflectance bands (B02, B03, B04, B08, B11) plus
  NDWI, MNDWI and NDVI — in 64 × 64 pixel patches (1.28 km across).
- **Size:** 537,156 parameters. Deliberately small; six lakes cannot support a
  large network.
- **Loss:** weighted cross-entropy plus Dice. Class balance is severe and varies
  by lake, and weighting the loss is preferable to resampling spatial pixels.
- **Trained from scratch**, not fine-tuned from ImageNet: the input is 20 m
  surface reflectance across 8 channels, not a 3-channel photograph, so
  pretrained filters do not match the physics. Every pretrained change-detection
  checkpoint surveyed in the research was trained on imagery 25–100× sharper.

**SCL is not an input channel.** It seeds the labels and masks cloud. Feeding it
in as well would let the network copy ESA's classifier, making any later
agreement with SCL circular.

### Random Forest — the baseline the U-Net must beat

A **Random Forest** combines many decision trees. Same 8 channels, same labels,
same folds — but it sees one pixel at a time with no surrounding context, which
is exactly the capability the U-Net adds.

### MNDWI / NDVI rule — the null baseline

The project's original method: MNDWI > 0 is water, else NDVI > 0.30 is
vegetation, else bare. It structurally cannot output "built-up", which is the
finding, not a bug.

### Mixed-effects risk model — and why not XGBoost

There are 42 rows, but they are **six lakes measured seven times each**. Every
OpenStreetMap building and road column is identical across all seven years of a
given lake, so those columns are effectively a name tag for the lake. A tree
model splits on them, memorises which lake it is looking at, and returns a
confident number that means nothing.

`16_risk_model.py --show-leakage` demonstrates this on purpose: XGBoost scores
**R² +1.000 in-sample** and collapses out-of-fold. A linear mixed-effects model
absorbs the same columns into a per-lake intercept instead, and reports an
honestly wide interval.

Predictions are wrapped in **jackknife+ conformal intervals**, left out by lake
and evaluated on lakes the intervals never saw. Two caveats travel with every
interval: coverage is over the project's *own derived target*, not true
encroachment; and the jackknife+ worst-case guarantee is 1 − 2α, so a nominal
90% target promises 80%.

---

## How the models are tested

### Nested leave-one-lake-out

Patches from one lake share the same illumination, the same atmosphere, and
across years literally the same pixels. A random train/test split would put
near-copies on both sides and report a flattering, fake score.

```
fold   Bellandur  Varthur  Madiwala  Hebbal  Ulsoor  Sankey
  0      TEST       val     train    train   train   train
  1      train      TEST     val     train   train   train
  2      train     train     TEST     val    train   train
  3      train     train    train     TEST    val    train
  4      train     train    train    train    TEST    val
  5       val      train    train    train   train    TEST
```

The validation lake is a *lake*, not a year, for the same reason. Normalisation
statistics and class weights are computed on the training lakes only — computing
them across all six first would leak the test lake into its own training.

**The effective sample size for generalising across lakes is 6**, not 42 and not
the pixel count. Say it before the panel does.

### Two checks that can fail

- **Shuffle-label control.** Train once on scrambled labels; the score must
  collapse to near chance. If it does not, the pipeline leaks and no other
  number can be trusted. On the last complete run it fell from **0.766 to
  0.373**.
- **A pre-registered decision rule.** Fixed in `MODEL_DESIGN.md` *before* any
  numbers existed: if the U-Net does not beat the Random Forest by more than
  **2 points of macro-F1**, the report recommends the Random Forest and presents
  the U-Net as the deep model that was built, tested, and found not to earn its
  complexity. That is a legitimate result, and it stops the conclusion being
  bent to fit the effort.

### Metrics

Accuracy alone is never reported — bare soil is about half the labelled pixels,
so a model predicting "bare" everywhere would look respectable and be useless.
Reported instead: per-class **IoU** (overlap between predicted and actual area),
**macro-F1** (averaged equally across the four classes, so the rare water class
counts as much as the common bare class), **Cohen's kappa**, the water↔vegetation
confusion cell specifically, and **Expected Calibration Error** after temperature
scaling.

---

## Results

### Land-cover classification, held-out lakes

| Model | macro-F1 | kappa | IoU water | hyacinth | bare | built |
|---|---|---|---|---|---|---|
| MNDWI / NDVI rule | 0.719 | 0.411 | 0.736 | 0.557 | 0.439 | **0.000** |
| **Random Forest** | **0.771** | **0.684** | 0.752 | **0.894** | **0.640** | 0.350 |
| U-Net *(5 of 6 folds)* | 0.800\* | — | **0.806** | 0.886 | 0.639 | **0.441** |

\* The U-Net currently leads by about 0.019 — just under the 0.02 margin — so
**Sankey Tank decides the recommendation**, and that is the fold which has not
finished. See *Run status* below.

### Building growth inside the 75 m fence

From `GOOGLE/Research/open-buildings-temporal/v1`. This is the per-year
encroachment-pressure signal the project previously had no way to measure.

| Lake | 2016 | 2019 | 2023 | Change |
|---|---|---|---|---|
| Madiwala | 700 | 783 | 1,053 | +50% |
| Ulsoor | 223 | 240 | 319 | +43% |
| Hebbal | 19 | 20 | 37 | +89% |
| Sankey Tank | 142 | 164 | 209 | +47% |
| Varthur | 63 | 99 | 104 | +65% |
| Bellandur | 246 | 228 | 289 | +17% |

A pre-flight check runs first: a secondary source claimed the dataset suppresses
detections near water, which would make every fence read zero while the pipeline
looked healthy. The check found 645–6,371 buildings in the 30–250 m ring around
each lake, so the layer is safe to use.

### The desilting rule

| Pattern between two years | Reading |
|---|---|
| Water down, **bare up**, water returns later | Drained for desilting — check works records |
| Water down, **built up**, and it persists | Encroachment signal — check the building count in the fence |
| Water down, **vegetation up** | Hyacinth spread, not lake loss |
| Water down across all lakes, rainfall low | Drought |

The rule's first version called Bellandur 2021 an encroachment signal: the model
had labelled 28.7 ha of dry, cracked bed as "built-up" (spectrally similar, and
built-up is the weakest class). Bellandur's built fraction across the seven years
runs `0, 0, .09, .02, .07, .00, 0` — buildings do not appear and vanish. The rule
now requires a rise in built-up to **persist** before reading it as
encroachment, and flags a non-persistent spike as probable bed misclassification.

---

## Run status

The pipeline was interrupted by a low-memory condition partway through its final
re-run, so the committed outputs are **not all from the same vintage**. This is
stated explicitly rather than averaged away.

| Output | Vintage | Status |
|---|---|---|
| `data/processed/lake_water_area.csv`, `features_lake_year.csv` | current | ✅ |
| `outputs/dl/labels/` | current | ✅ |
| `outputs/dl/rf/` | current | ✅ |
| `outputs/dl/gee/` | current | ✅ |
| `outputs/dl/unet/`, `outputs/dl/pred/` | before the reflectance fix | ⚠️ needs re-run |
| `outputs/dl/geofence/`, `outputs/dl/risk/`, `features_lake_year_dl.csv` | before the area and reflectance fixes | ⚠️ needs re-run |

To bring everything to one vintage (~20 minutes):

```bash
python src/14_train_unet.py --epochs 100
python src/14_train_unet.py --shuffle-labels --epochs 30
python src/17_fetch_gee.py --only dynamic_world
python src/17_fetch_gee.py --only gsw
python src/15_geofence.py
python src/16_risk_model.py --show-leakage
```

---

## Four bugs this build uncovered

Properties of the data, not model results. Each was found by something refusing
to reconcile.

### 1. The reflectance offset was applied twice

The pipeline computed `reflectance = DN × 0.0001 − 0.1`, following the band
metadata. But every scene also carries `earthsearch:boa_offset_applied: true` —
the provider had **already** subtracted that offset before serving the file.
Subtracting it again pushed every water pixel negative, and a normalised
difference of two negative numbers flips sign, so water read as NDVI ≈ +0.9
("dense vegetation").

**Effect:** MNDWI > 0 found only **4–16%** of the pixels ESA's own classifier
calls water. After the fix, **82–96%**. Sankey Tank, a full clean tank, had been
reported as ~1 ha of water out of 12.4. The index rule's score rose from **0.304
to 0.719** — it had been judged on broken inputs.

Verified against the STAC items for 2019, 2021, 2023 and 2025 (all processing
baseline 05.xx, all `boa_offset_applied: true`) and against raw values over
SCL-water pixels in every year.

### 2. Madiwala Lake was one-third off the map

The study area's southern edge cut through the lake. One of its two basins
(9.58 ha) was outside the grid entirely, so every per-pixel figure described
**54.88 ha of an 82.84 ha lake**. The area of interest was widened
(`lat_min` 12.905 → 12.890) and the imagery re-fetched. Madiwala now measures
82.92 ha; the other five lakes came through unchanged to the pixel.
`15_geofence.py` now prints an explicit warning if any lake is clipped again.

### 3. The index columns were numerically corrupt

`mean_mndwi = −61.63` for Bellandur 2025; `mean_ndwi = −111.35` for Sankey 2023.
A normalised difference is bounded by ±1. The divisions now guard against a
near-zero denominator and clip to the valid range.

### 4. The planned patch size did not fit the lakes

The design called for 128-pixel training patches, but four of six lakes are
smaller than that — Sankey Tank is 76 × 74 — so each yielded a single patch per
year. At 64 pixels with 50% overlap the training set went from 98 patches to
roughly 500, and 1.28 km of context still exceeds the network's receptive field,
so nothing was lost.

---

## The legal buffer

Contested, and changed twice. The pipeline tests **30 m and 75 m as bracketing
cases** rather than asserting one correct number.

- **30 m** — Bengaluru Master Plan, and the Karnataka Tank Conservation and
  Development Authority Act, 2014.
- **75 m** — National Green Tribunal, O.A. 222/2014 (order dated 4 May 2016).
- **Supreme Court**, *Mantri Techzone Pvt. Ltd. v. Forward Foundation*
  (5 March 2019) — **set aside the NGT's general 75 m direction**, but sustained
  it specifically against the two encroaching parties at the Bellandur–Agara
  site. The common report that the Court "reinstated 30 m city-wide,
  prospectively" is a simplification the primary text does not support.
- **Karnataka Act 19 of 2026** (gazetted 18 February 2026) replaced the flat
  30 m with a table tiered by tank size, measured **from the revenue boundary**.
  Over 100 acres stays at 30 m.

At 20 m pixels, only the 24 m and 30 m tiers are resolvable at all; a 1–12 m
statutory buffer is sub-pixel and therefore untestable with Sentinel-2. That is a
finding, not a failure.

**The boundary caveat:** the 2026 Act measures from the *revenue* boundary. This
project uses an OpenStreetMap polygon, which has no statutory standing. No
login-free machine-readable authoritative boundary was found at KTCDA, Survey of
India, Bhuvan, National Wetland Atlas 2024 or the BBMP Lakes Monitoring System.

---

## Limitations

- **No encroachment ground truth.** No official list of encroached lakes was
  obtained, so no model here is validated against real encroachment. Every score
  measures agreement with labels the project made itself.
- **The labels are still uncorrected seeds**, from the Scene Classification Layer
  and OpenStreetMap. The hand-correction step
  (`12_make_labels.py --export-qgis`) is built and ready but has not been done;
  the design budgets 6–10 student-days for it.
- **The built-up class is not a time series.** Its training labels come from one
  2026 OpenStreetMap snapshot applied to all seven years, so it says *where*
  buildings are, not *when* they appeared. The yearly signal comes from Open
  Buildings Temporal instead, which ends in 2023 — so 2024 and 2025 have no
  per-year building data, and the gap is never interpolated.
- **Six lakes, one image per year.** Everything describes the dry-season minimum,
  not the annual cycle.
- **Between-lake variance is several times the residual variance**, and it is
  estimated from six clusters — below the 15–50 usually cited as a minimum. The
  conformal intervals are wide by construction, and that is the finding.

### The honest next step

The project's own limitation is sample size. Government survey figures
(Times of India, 11 Jan 2026) report **730 of 837** water bodies in Bengaluru
Urban District under illegal occupation, surveyed by the Minor Irrigation
Department and the State Survey Settlement and Land Records Department, with
per-lake GPS boundaries uploaded to an internal Web GIS platform. If that
platform is obtainable — an RTI request is the cheapest route — it would supply
**both** a legally meaningful boundary and real lake-level ground truth, and take
the project from 42 rows to roughly 1,400. Note that ~1,400 rows would still be
~200 independent lakes, so the clustering problem reappears at larger scale, just
with a real label attached.

---

## Running it

```bash
pip install -r requirements.txt          # base pipeline
pip install -r requirements-dl.txt       # modelling stack
earthengine authenticate                 # one-time, opens a browser

# data
python src/00_dataset_inventory.py       # the 20-source catalogue
python src/01_fetch_nonsatellite.py      # OpenStreetMap + NASA POWER   (~8 min)
python src/02_fetch_satellite.py         # Sentinel-2                   (~20 min)
python src/06_integrate_features.py      # join rainfall

# labels, then models
python src/12_make_labels.py             # seed labels
python src/12_make_labels.py --export-qgis       # optional: hand correction
python src/12_make_labels.py --import-corrected  # fold corrections back in
python src/13_train_rf.py                # baselines                    (~3 min)
python src/14_train_unet.py --epochs 100  # 6 folds                     (~12 min CPU)
python src/14_train_unet.py --shuffle-labels --epochs 30   # the control

# evidence and risk
python src/17_fetch_gee.py               # Earth Engine layers
python src/15_geofence.py                # buffers and composition
python src/16_risk_model.py --show-leakage

# EDA reports (optional, from the original exploratory stage)
python src/04_eda_nonsatellite.py
python src/05_eda_satellite.py
python src/08_build_dashboard.py
python src/09_simple_eda.py
python src/10_build_simple_report.py
```

Add `--device cuda` to stage 14 for a GPU. Everything else is processor-bound.
Earth Engine needs a free account and a Google Cloud project; set `EE_PROJECT` if
yours differs from the default in `src/dl/gee.py`.

---

## Repository layout

```
src/
  s2_common.py              Sentinel-2 scaling and spectral indices
  viz_style.py              palette and matplotlib theme (light & dark)
  00_dataset_inventory.py   the 20-source catalogue
  01_fetch_nonsatellite.py  OpenStreetMap + NASA POWER + link check
  02_fetch_satellite.py     Sentinel-2 + NASA GIBS
  04, 05, 06                EDA stages and the feature join
  08, 09, 10                dashboard, simple figures, simple report
  12_make_labels.py         four-class seed labels (+ QGIS round-trip)
  13_train_rf.py            MNDWI rule + Random Forest baselines
  14_train_unet.py          U-Net training, nested leave-one-lake-out
  15_geofence.py            buffer zones and intrusion counting
  16_risk_model.py          mixed-effects models, conformal, desilting rule
  17_fetch_gee.py           Open Buildings Temporal, Dynamic World, JRC GSW
  dl/
    common.py               grid, lake geometry, folds, channels
    labels.py               seed-label construction
    dataset.py              patch extraction, fold-local normalisation
    unet.py                 the model + temperature scaling
    losses.py               weighted cross-entropy + Dice
    metrics.py              IoU, macro-F1, kappa, ECE, confusion
    gee.py                  Earth Engine access
data/raw/                   cached downloads (OSM JSON, Sentinel-2 NPZ, GIBS PNG)
data/processed/             tidy CSV / GeoJSON, the analysis grid
outputs/dl/                 labels, model results, predictions, geo-fence, risk
outputs/figures/            EDA figures, light and dark
reports/                    the two HTML reports from the EDA stage
research-dl-lake-encroachment/
  RESEARCH_DEEP_LEARNING.md 30 methods researched and ranked
  MODEL_DESIGN.md           the design this pipeline implements
  RESULTS.md                measured results and findings
  results/                  30 JSON files, 35 fields each
deep-research-report.md     the original source material (12 candidate projects)
```

### Documents worth reading in order

1. **`research-dl-lake-encroachment/RESEARCH_DEEP_LEARNING.md`** — thirty
   candidate methods researched and ranked, with what was rejected and why.
2. **`research-dl-lake-encroachment/MODEL_DESIGN.md`** — the design, including
   the pre-registered decision rule.
3. **`research-dl-lake-encroachment/RESULTS.md`** — what was measured.

> Note: `reports/eda_simple.html` and `reports/dashboard.html` date from the
> exploratory stage and still contain the pre-fix water areas and the old
> (incorrect) account of the reflectance offset. Re-run stages 09 and 10 to
> regenerate them.

---

## Credits

Sentinel-2 L2A from ESA Copernicus via Earth Search on AWS Open Data.
Map data © OpenStreetMap contributors (ODbL). Weather from NASA POWER.
Building history from Google Open Buildings Temporal. Land cover from Google
Dynamic World. Water history from the EC Joint Research Centre Global Surface
Water dataset. Regional context from NASA GIBS.
