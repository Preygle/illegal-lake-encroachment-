# Results

Final run, 4 October 2026. Every output in `outputs/dl/` is on the corrected data
(reflectance offset applied once, Madiwala fully on the grid): stages 02–13 last
ran on 22 September and stages 14–17 on 4 October, and nothing stages 02–13
produce depends on anything changed in between. Implements `MODEL_DESIGN.md`. Every number is reproducible from the
commands in the README.

---

## 1. Land-cover classification

Six held-out lakes, nested leave-one-lake-out. Each lake is scored by a model
that never trained on it.

| Model | macro-F1 | spread over 6 lakes | kappa | IoU water | hyacinth | bare | built |
|---|---|---|---|---|---|---|---|
| MNDWI / NDVI rule | 0.719 | 0.608 – 0.797 | 0.411 | 0.736 | 0.557 | 0.439 | **0.000** |
| **Random Forest** | **0.771** | 0.721 – 0.799 | 0.684 | 0.752 | **0.894** | **0.640** | 0.350 |
| U-Net | 0.786 | 0.713 – 0.840 | **0.710** | **0.755** | 0.879 | 0.609 | **0.458** |

**Verdict: the pre-registered rule recommends the Random Forest.** The U-Net
leads by **+0.015** macro-F1, below the +0.02 margin fixed in the design before
any numbers existed. Wilcoxon signed-rank over the six paired lakes p = 0.312,
which with six pairs has little power and is reported as descriptive only.

What the mean hides:

- **The U-Net wins clearly on built-up** (0.458 vs 0.350). Spatial context is
  what separates a building from a bright bare pixel, and only the U-Net has it.
- **The Random Forest wins on bare bed and vegetation** (0.640 vs 0.609; 0.894 vs
  0.879). Those classes are spectrally distinctive per pixel, so context adds
  little.
- **The U-Net wins on 4 of 6 lakes**, most clearly Hebbal (+0.060).

### Per lake

| Held-out lake | U-Net | Random Forest | Index rule |
|---|---|---|---|
| Bellandur | **0.814** | 0.799 | 0.797 |
| Varthur | **0.779** | 0.763 | 0.754 |
| Madiwala | 0.757 | **0.777** | 0.718 |
| Hebbal | **0.840** | 0.780 | 0.739 |
| Ulsoor | **0.810** | 0.786 | 0.698 |
| Sankey Tank | 0.713 | **0.721** | 0.608 |

### Checks

| Check | Result |
|---|---|
| Shuffle-label control | 0.786 → **0.389** with scrambled labels. The pipeline does not leak. |
| Run-to-run stability | Retraining moved individual folds by up to ±0.009, but the five-fold mean moved by 0.0004 (0.7996 → 0.800). The verdict is not an artefact of noise. |
| Calibration | Mean Expected Calibration Error 0.054 after temperature scaling. |

The control sits above pure chance because labels are permuted *within* each
patch, which preserves per-patch class proportions and lets the model still
learn patch-level priors. The size of the gap is the point.

---

## 2. What the four-class output shows

Footprint composition from the out-of-fold U-Net maps.

| Lake | Year | Water | Hyacinth | Bare bed | Built |
|---|---|---|---|---|---|
| Bellandur | 2019 | 41.3 | 267.4 | 6.4 | 0.0 |
| Bellandur | **2021** | **0.0** | 134.3 | **145.8** | 35.0 |
| Bellandur | 2025 | 67.6 | 237.8 | 9.6 | 0.0 |
| Varthur | **2021** | **0.0** | 4.6 | **149.4** | 0.0 |
| Madiwala | 2019 | 65.3 | 4.5 | 12.6 | 0.4 |
| Madiwala | **2020** | 10.8 | **71.2** | 1.0 | 0.0 |
| Hebbal | 2019–2025 | 27.4 – 39.4 | 3.7 – 17.6 | 0.3 – 5.9 | 0.0 |
| Sankey Tank | 2019–2025 | 11.1 – 11.6 | 0.1 – 0.6 | 0.4 – 1.2 | 0.0 |

All figures in hectares.

- **Bellandur and Varthur 2021: zero open water, ~146–149 ha of bare bed.** The
  drained-lake signature as a measured quantity.
- **Madiwala 2020: water falls from 65 to 11 ha while hyacinth rises from 4.5 to
  71 ha.** A weed mat, not a lost lake.
- **Hebbal, Sankey Tank and Ulsoor are stable full lakes.** Under the old
  double-offset bug, the exploratory analysis had all three nearly dry.

### Bellandur 2021's "35 ha built"

That is dry, cracked lake bed misclassified as built-up — spectrally close, and
built-up is the weakest class. Bellandur's built fraction across the seven years
runs `0, 0, .11, .02, .03, .00, 0`; buildings do not appear and vanish. The
desilting rule handles this by requiring a rise in built-up to *persist* before
reading it as encroachment.

---

## 3. The desilting rule

Every reading other than "no material water loss":

| Lake | Change | Reading |
|---|---|---|
| Bellandur | 2020 → 2021 | **Drained** (bed bare, water returns) — check desilting records |
| Varthur | 2020 → 2021 | **Drained** (bed bare, water returns) — check desilting records |
| Hebbal | 2019 → 2020 | Hyacinth spread, not lake loss |
| Madiwala | 2019→20, 2021→22, 2022→23, 2024→25 | Hyacinth spread, not lake loss |

**No lake-year produced an encroachment signal.** Water losses in this study
period are explained by desilting and hyacinth, not construction inside the
footprint.

---

## 4. Independent cross-checks

None of these is ground truth; they are reported as agreement, never accuracy.

### `GOOGLE/DYNAMICWORLD/V1`

Matched to the exact same Sentinel-2 scene for all seven years.

| Lake | Agreement | Dynamic World calls flooded vegetation | …of our hyacinth pixels |
|---|---|---|---|
| Bellandur | 72.4% | 5.2% | 6.7% |
| Varthur | 66.0% | 4.3% | 4.9% |
| Madiwala | 87.9% | 6.2% | 13.0% |
| Hebbal | 93.8% | 1.3% | 8.1% |
| Ulsoor | 84.6% | 0.2% | 2.0% |
| Sankey Tank | 91.6% | 0.0% | 0.0% |

Agreement is lowest on the two hyacinth lakes, and for a specific reason:
**Dynamic World rarely uses its `flooded_vegetation` class for the hyacinth
mats.** The research picked that class as the best ready-made hyacinth proxy; in
practice Dynamic World labels the mats as trees, grass or crops. A legend
mismatch, not a disagreement about what is there.

### `JRC/GSW1_4/YearlyHistory` (2019–2021)

GSW counts water seen at *any* time of year; ours is one dry-season date, so GSW
should read higher. On matched pixels it does in **14 of 18** lake-years. The
four exceptions are marginal — three within 1.3 points, Hebbal 2021 by 4.5 —
consistent with GSW's coarser 30 m Landsat pixels calling shoreline edges
non-water.

GSW independently confirms the drain: **Bellandur 2021 has 0% permanent water
and 10.6% water at any point in the year.**

---

## 5. Geo-fence

### Building growth inside the fence — `GOOGLE/Research/open-buildings-temporal/v1`

| Lake | 75 m, 2016 | 75 m, 2023 | Growth per year |
|---|---|---|---|
| Madiwala | 700 | 1,053 | +5.6% |
| Ulsoor | 223 | 319 | +5.4% |
| Hebbal | 19 | 37 | +9.1% |
| Sankey Tank | 142 | 209 | +4.9% |
| Varthur | 63 | 104 | +5.3% |
| Bellandur | 246 | 289 | +2.3% |

The pre-flight check passed for every lake: 645–6,371 buildings detected in the
30–250 m ring, so the dataset is not suppressing detections near water.

The 30 m ring is noisier. Bellandur and Varthur show falling counts there
(−5.1% and −10.8% per year), driven by unusually high 2016 values — consistent
with the documented instability of Open Buildings Temporal's earliest years,
when fewer Sentinel-2 frames were available.

### OpenStreetMap snapshot (single 2026 epoch)

| Lake | Acres | Tier | 30 m ring | 75 m ring |
|---|---|---|---|---|
| Bellandur | 779.3 | 30 m | 2 | 66 |
| Varthur | 381.3 | 30 m | 1 | 17 |
| Madiwala | 204.7 | 30 m | **11** | **290** |
| Hebbal | 112.0 | 30 m | 0 | 7 |
| Ulsoor | 93.6 | sub-100-acre — look up | 8 | 75 |
| Sankey Tank | 30.8 | sub-100-acre — look up | 5 | 63 |

Madiwala's 75 m count rose from 229 to 290 once the grid was widened to include
its southern side.

---

## 6. Risk models

### Water model — `water_frac ~ year + rainfall + (1 | lake)`

| | |
|---|---|
| Year | +0.017 per year, 95% CI −0.002 to +0.036 (not significant) |
| Rainfall (365 days, standardised) | −0.011, 95% CI −0.049 to +0.027 (not significant) |
| Between-lake variance | 0.143 |
| Residual variance | 0.014 |
| **Ratio** | **10.4×** |

**Rainfall explains nothing at the pooled level.** Lake water in this sample is
set by inflow, desilting and hyacinth, not by the monsoon.

**The conformal intervals are honest and useless, and that is the finding.**
Jackknife+ coverage is 100% out-of-sample, but the median interval is **1.30
wide on a quantity that only spans 0 to 1**. With between-lake variance ten times
the residual, a model that has not seen a lake cannot place its water fraction.
Six lakes cannot support a predictive model of an unseen lake's water — say so
before the panel does.

### Building model — `log(1 + buildings in 75 m fence) ~ year + (1 | lake)`

| | |
|---|---|
| **Average growth** | **+5.4% per year**, 95% CI +4.1% to +6.7% (z = 8.4) |
| Jackknife+ coverage | 83.3% (nominal 90%, worst-case guarantee 80%) |

**This is the project's strongest quantitative result.** Building counts inside
the 75 m fence grew at about 5% a year across all six lakes from 2016 to 2023,
and the estimate is tight. It is a growth *rate*, which the model estimates
well; predicting the *level* for an unseen lake is another matter — Hebbal, whose
counts are an order of magnitude below the others, falls outside every interval.

### XGBoost leakage exhibit

| | R² |
|---|---|
| In-sample | **+1.000** |
| Leave-one-lake-out | **+0.302** |

**15 of 30 pressure features are constant within a lake** — building counts, ring
areas, footprint area, coordinates. Those columns are a lake-identity label; a
tree model splits on them and memorises which lake it is looking at. The gap
from 1.000 to 0.302 is the memorisation made visible. This belongs in the report
in place of a SHAP plot.

The exhibit deliberately excludes the land-cover columns derived from the same
map as the target. `water_ha` is the target multiplied by footprint area, and the
footprint fractions sum to ~1 with it; a first version that included them scored
0.612 out-of-fold, which measured arithmetic rather than anything about lakes.

---

## 7. Findings the build exposed

Not model results — properties of the data, each found by a number that would
not reconcile.

1. **The reflectance offset was applied twice.** Every scene carries
   `earthsearch:boa_offset_applied: true`, yet the pipeline subtracted the
   offset again. MNDWI > 0 found 4–16% of ESA's water pixels; after the fix,
   82–96%. The index rule rose from macro-F1 0.304 to 0.719.
2. **Madiwala Lake was a third off the grid.** 54.88 of its 82.84 ha were
   measured. Now 82.92 ha; the other five lakes unchanged to the pixel.
3. **The index columns were numerically corrupt** (`mean_mndwi = −61.63`).
   Divisions now guard near-zero denominators and clip to ±1.
4. **The planned 128 px patches did not fit four of the six lakes.** 64 px with
   50% overlap took the training set from 98 patches to ~500.
5. **Two bugs in my own evaluation code**, both caught before reporting: the GSW
   comparison divided by footprint pixels GSW never observed, and the first
   leakage exhibit fed the target's own area to XGBoost.

---

## 8. What this does not establish

- **No encroachment ground truth.** Every score measures agreement with labels
  the project made itself.
- **The labels are uncorrected seeds** from the Scene Classification Layer and
  OpenStreetMap. Hand correction (`12_make_labels.py --export-qgis`) is built
  but not done; re-run the scores afterwards.
- **The built-up class is not a time series.** Its labels come from one 2026
  OpenStreetMap snapshot; the yearly building signal is Open Buildings Temporal,
  which ends in 2023.
- **Six lakes.** Effective sample size for cross-lake generalisation is 6.
- **One dry-season scene per year** — the annual minimum, not the annual cycle.
