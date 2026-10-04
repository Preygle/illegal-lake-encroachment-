# Results: four-class segmentation, geo-fence, and risk model

Run on 17 September 2026, CPU (16 cores), ~8 minutes for all six U-Net folds.
Implements `MODEL_DESIGN.md`. Every number below is reproducible from the
commands in §6.

---

## 1. The headline

**The Random Forest wins, and the pre-registered rule says report it as the
recommendation.**

| Model | macro-F1 | spread over 6 lakes | kappa | IoU water | hyacinth | bare | built |
|---|---|---|---|---|---|---|---|
| MNDWI/NDVI rule (current baseline) | 0.299 | 0.247 – 0.402 | 0.079 | 0.041 | 0.333 | 0.198 | **0.000** |
| **Random Forest** | **0.771** | 0.720 – 0.799 | 0.688 | 0.749 | 0.891 | **0.629** | 0.365 |
| U-Net | 0.759 | 0.680 – 0.818 | 0.680 | 0.727 | 0.837 | 0.543 | **0.461** |

Delta U-Net − Random Forest = **−0.013**, against a required margin of +0.02.
Wilcoxon signed-rank over the six paired lakes p = 1.000 — which with six pairs
has almost no power and is reported as descriptive only.

The decision rule was fixed in the design *before* the numbers existed, so this
is a result rather than a rationalisation: **build the U-Net, report it, and
recommend the Random Forest.** That satisfies the guide's deep-learning
requirement and is a stronger scientific finding than a deep model with no
baseline to compare against.

Two things the mean hides:

* **The U-Net is clearly better at the built class** (IoU 0.461 vs 0.365).
  Spatial context is what distinguishes a building from a bright bare pixel,
  and only the U-Net has it.
* **The Random Forest is clearly better at bare** (0.629 vs 0.543), which is
  what drags the U-Net's mean down. Bare bed is spectrally distinctive on its
  own, so per-pixel evidence is enough and context adds nothing.

Both beat the index rule by a wide margin, and the rule's 0.000 on built is not
a bug — an index threshold structurally cannot detect buildings, which is
exactly why the comparison was worth running.

### Negative control

| | macro-F1 |
|---|---|
| Real labels | 0.759 |
| Permuted labels | 0.340 |
| **Drop** | **0.418** |

The pipeline does not leak. The control sits above pure chance because labels
are permuted *within* each patch, which preserves per-patch class proportions
and lets the model still learn patch-level priors; the size of the gap is the
point.

### Calibration

Mean Expected Calibration Error 0.047 after temperature scaling, fitted on the
inner-validation lake. Per-fold temperatures ranged 1.00 – 1.43.

---

## 2. What the four-class output bought

This is the part that matters more than the leaderboard. Bellandur, from the
segmentation map against the old threshold table:

| Year | old `water_ha` | water | hyacinth | **bare** | built |
|---|---|---|---|---|---|
| 2019 | 15.68 | 41.16 | 268.16 | 6.12 | 0.00 |
| 2020 | 25.16 | 27.16 | 266.52 | 21.76 | 0.00 |
| **2021** | 0.40 | **0.00** | 129.72 | **157.04** | 28.68 |
| 2022 | 7.32 | 21.36 | 180.44 | 109.04 | 4.60 |
| 2023 | 6.40 | 31.36 | 105.68 | 158.04 | 20.36 |
| 2024 | 3.08 | 40.84 | 81.84 | 191.76 | 1.00 |
| 2025 | 107.88 | 68.20 | 235.88 | 11.36 | 0.00 |

**Bellandur 2021: zero open water and 157 ha of bare bed.** The drained-lake
signature is now a measured quantity instead of an inference from a true-colour
image. Varthur 2021 is the same shape: 0 water, 149 ha bare. The old
three-class split could only ever report "water went down", which is precisely
the ambiguity the whole project turns on.

---

## 3. The desilting rule (Layer D)

Applied to consecutive years. The readings that fired:

| Lake | Transition | Reading |
|---|---|---|
| Bellandur | 2020 → 2021 | drained (bed bare, water returns) — check desilting records |
| Varthur | 2020 → 2021 | drained (bed bare, water returns) — check desilting records |
| Ulsoor | 2023 → 2024 | drained (bed bare, water returns) — check desilting records |
| Madiwala | 2019→20, 2022→23, 2024→25 | hyacinth spread, not lake loss |

Everything else: no material water loss. **No lake-year produced an
encroachment signal.**

### A failure found and fixed during implementation

The rule's first version called Bellandur 2020 → 2021 an **encroachment
signal**. The cause: the segmentation labelled 28.68 ha of the 2021 dry bed as
*built*, because cracked lakebed is spectrally close to an urban surface — and
built is the weakest class (IoU 0.461).

Bellandur's built fraction across 2019–2025 runs `0, 0, .091, .015, .065, .003,
0`. Buildings do not appear and vanish. The rule now requires a rise in the
built class to **persist** before it can be read as encroachment, and labels a
non-persistent spike as probable bed misclassification. With that check,
Bellandur and Varthur 2021 both read correctly as drained.

This is worth putting in the report: the four-class model made a mistake that a
three-class model could not even express, and the mistake was catchable because
the classes are physically interpretable.

---

## 4. Geo-fence (Layer B)

Building intrusion from OpenStreetMap centroids, both fences:

| Lake | Acres | KTCDA tier | 30 m ring | 75 m ring | on grid |
|---|---|---|---|---|---|
| Bellandur | 780.4 | 30 m | 2 | 66 | 100% |
| Varthur | 381.3 | 30 m | 1 | 17 | 100% |
| Madiwala | 204.7 | 30 m | **11** | **229** | **66%** |
| Hebbal | 112.0 | 30 m | 0 | 7 | 100% |
| Ulsoor | 93.6 | sub-100-acre — check gazette | 8 | 75 | 100% |
| Sankey Tank | 30.8 | sub-100-acre — check gazette | 5 | 63 | 100% |

Madiwala carries by far the heaviest intrusion, consistent with it having the
highest building density in the existing table (14.7 buildings/ha within 250 m).

Ulsoor at 93.6 acres misses the 100-acre threshold by 6.4 acres, so its
statutory buffer under the February 2026 tiered table is **not** 30 m and has to
be looked up. Sankey Tank at 30.8 acres likewise. Note the acreage here is
computed from the OpenStreetMap polygon, which is **not** the revenue boundary
the 2026 Act measures from.

---

## 5. Risk model (Layer C)

Mixed-effects model, random intercept per lake, on `water_frac`:

| | |
|---|---|
| Between-lake variance | 0.1494 (sd 0.386) |
| Residual variance | 0.0273 (sd 0.165) |
| Jackknife+ interval half-width | **0.659** on a fraction in [0, 1] |
| Empirical marginal coverage | 90.5% (nominal 90%, worst-case guarantee 80%) |

**Between-lake variance is 5.5× the residual variance.** Almost all the
variation is *between* lakes rather than within them, and with six clusters that
component cannot be estimated precisely. The consequence is the interval: ±0.659
on a quantity bounded in [0, 1] is close to useless for predicting an unseen
lake — which is the honest answer, and the number to put in front of the panel
when asked whether six lakes can support a predictive model.

Every interval covers the project's **own derived target**, not true
encroachment. There is no ground truth to calibrate against, and with six
clusters only marginal coverage is claimable, never per-lake.

### The XGBoost leakage exhibit

| | R² |
|---|---|
| In-sample | **+1.000** |
| Leave-one-lake-out | +0.314 |

**17 of 33 numeric features are constant within a lake** — building counts, ring
areas, footprint area, coordinates. Those columns are a lake-identity label, and
a tree model splits on them. A perfect in-sample fit collapsing to +0.314
out-of-fold is the memorisation made visible. This figure belongs in the report
in place of a SHAP plot.

---

## 6. Reproducing this

```bash
pip install -r requirements-dl.txt

python src/12_make_labels.py              # seed labels from SCL + OpenStreetMap
python src/12_make_labels.py --export-qgis        # optional: hand correction
python src/12_make_labels.py --import-corrected   # optional: read it back

python src/13_train_rf.py                 # baseline + index rule, ~2 min
python src/14_train_unet.py --epochs 100  # 6 folds, ~8 min CPU / ~1 min GPU
python src/14_train_unet.py --shuffle-labels --epochs 30   # negative control
python src/15_geofence.py                 # Layer B
python src/16_risk_model.py --show-leakage   # Layers C and D
```

`--device cuda` on step 4 for a GPU. Everything else is CPU-bound and fast.

Outputs land in `outputs/dl/`: `labels/`, `rf/`, `unet/`, `pred/`, `geofence/`,
`risk/`. The joined table is `data/processed/features_lake_year_dl.csv`
(42 rows × 36 columns), which replaces the threshold-derived
`features_lake_year.csv` for downstream work.

---

## 7. Findings that came out of building it

Four things that were not visible from the design and are not model results —
they are properties of the data that the implementation exposed.

**Madiwala Lake is 34% outside the analysis grid.** Its southern lobe falls
below the AOI's bottom edge (grid `miny` 1428000; the lake reaches 1427309). One
of its two polygon parts, 9.58 ha, is missing entirely. So `ref_ha = 54.88` in
the existing `lake_water_area.csv` describes 54.88 ha of an 82.84 ha lake, and
every per-pixel figure ever published for Madiwala — including in the EDA report
— covers two-thirds of it. Fix: widen `AOI` in `src/02_fetch_satellite.py:46`
and re-fetch. `15_geofence.py` now prints an explicit warning rather than
letting the numbers reconcile silently.

**The built class cannot be read as a time series.** Its training labels come
from a single 2026 OpenStreetMap snapshot applied to all seven years, so a pixel
built in 2023 is labelled built in 2019 too. The model receives contradictory
supervision — same label, different spectra — and the resulting `built_frac_*`
says *where* buildings are, not *when* they appeared. A genuine per-year
building series needs `GOOGLE/Research/open-buildings-temporal/v1` (annual
2016–2023). Recorded in `outputs/dl/geofence/provenance.json`.

**128 px patches were unusable; the design's estimate was wrong.** Four of the
six lakes have a footprint-plus-collar bounding box smaller than 128 px — Sankey
Tank is 76 × 74. At 128/64 the index held 98 patches, with 7 per small lake
(one position × seven years). At 64/32 it holds ~500, and 1.28 km still exceeds
the network's receptive field, so no context was lost.

**Two indices in the existing feature table were numerically corrupt**, as the
design predicted: `mean_mndwi = −61.63` for Bellandur 2025, `mean_ndwi =
−111.35` for Sankey 2023. After the −0.1 offset the denominator passes through
zero. `s2_common._ratio()` now guards the division and clips to [−1, 1]. Masks
were never affected; the scene-mean statistics were.

---

## 8. What this does not do

* **No encroachment ground truth**, so no model here is validated against real
  encroachment. Every score measures agreement with hand-correctable seed
  labels.
* **Seed labels are uncorrected in this run.** They come from SCL plus
  OpenStreetMap. `12_make_labels.py --export-qgis` produces the GeoTIFFs for
  hand correction; the design budgets 6–10 student-days for it, and the scores
  above should be re-run afterwards.
* **Six lakes.** Effective sample size for cross-lake generalisation is 6, not
  42 and not the pixel count.
* **One dry-season scene per year**, so all of this is the annual minimum, not
  the annual cycle.
