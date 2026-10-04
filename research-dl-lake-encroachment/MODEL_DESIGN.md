# Model design: lake encroachment detection and risk analysis

Design note, 17 September 2026. Written by following the `ml-review` skill's
"suggest an approach" procedure (simplest-thing-first, one wiki page per step,
upgrade triggers, named tradeoffs) against the thirty-method research in
`RESEARCH_DEEP_LEARNING.md`. Wiki citations are to the `ml-skills` plugin reference
library installed at user scope; `(consensus)` and `(heuristic)` mark claims that
come from working knowledge rather than a verified source.

Environment this is designed for, checked on the laptop: Python 3.13.3, PyTorch
2.10.0 **CPU only**, scikit-learn 1.6.1, 16 cores, free Colab/Kaggle GPU available.
Not installed yet: `statsmodels`, `mapie`, `pymannkendall`.

---

## 0. The problem, pinned down

| | |
|---|---|
| **Input** | 7 Sentinel-2 L2A scenes (2019–2025, one dry-season scene each), tile MGRS-43PGQ, bands B02/B03/B04/B08/B11/SCL, on a common 20 m grid of **843 × 1095 pixels**, cached as `data/raw/s2/s2_<year>.npz`. Six lake polygons from OpenStreetMap, rasterised by `lake_masks()` in `src/02_fetch_satellite.py:274` into an `inside` mask and a 500 m `collar` mask per lake. |
| **Output wanted** | (1) A per-pixel land-cover map good enough to tell open water from water hyacinth from dry lake bed from buildings. (2) A per-lake-year count of built structures inside the statutory buffer. (3) A per-lake risk estimate with an honest interval. |
| **Labels** | **None for encroachment.** Labels for land cover must be made. |
| **Scale** | ~920,000 pixels per scene × 7 scenes. **42 lake-years. 6 lakes.** |
| **Deploy target** | A report and a viva. Batch inference over 7 scenes. No latency budget. |
| **Existing stack** | `s2_common.indices()` already produces reflectance, NDWI, MNDWI, NDVI, a cloud mask and the SCL water flag from a cached stack. |

The shape of the problem decides everything: **there is abundant data for a
per-pixel model and almost none for a per-lake model.** So the deep model classifies
pixels, the geo-fence turns pixels into evidence, and a small statistical model — not
a deep one — handles the six lakes.

---

## 1. The design in one picture

```
  s2_<year>.npz ──► s2_common.indices() ──► 8-channel stack (843 × 1095)
                                                    │
                       ┌────────────────────────────┼─────────────────────────┐
                       ▼                            ▼                         ▼
              [A] U-Net, 4 classes          [A'] Random Forest         MNDWI>0 / NDVI>0.3
              per-pixel, from scratch        per-pixel, same folds      (current null baseline)
                       │                            │                         │
                       └─────────── same leave-one-lake-out folds ────────────┘
                                                    │
                                   land-cover map: water / vegetation / bare / built
                                                    │
                    ┌───────────────────────────────┼──────────────────────────┐
                    ▼                               ▼                          ▼
        [B] inside lake polygon           [B] 30 m and 75 m buffer      [B] Open Buildings
        water_frac, veg_frac (hyacinth),   built_frac from U-Net +      Temporal v1 footprints
        bare_frac per lake-year            OSM rasterised built-up      inside buffer, 2016–2023
                    │                               │                          │
                    └───────────────────────────────┼──────────────────────────┘
                                                    ▼
                              42-row lake-year table (+ rainfall from NASA POWER)
                                                    │
                          [C] linear mixed-effects model, random intercept per lake
                              wrapped in jackknife+ conformal intervals
                                                    │
                                   [D] desilting-vs-encroachment RULE (not learned)
```

Three layers, each judged on its own terms:

- **[A] the deep model** — the guide's requirement, and the thing that makes [B]
  trustworthy. Judged by per-class IoU on a held-out lake.
- **[B] the geo-fence** — the actual encroachment signal. Judged by whether a building
  inside the buffer is there or not; no model needed to interpret it.
- **[C] the risk model** — the honest treatment of 42 rows. Judged by interval width
  and coverage, not by accuracy.

---

## 2. Layer A — the deep model

### 2.1 Why a small U-Net from scratch, and not a pretrained backbone

The wiki's architecture table says exactly two things about this situation:
segmentation → U-Net, and *"from scratch (small data) → simple custom CNN — avoid
overfitting"* (wiki: ml-architectures/cnn §"Architecture Selection Guide"). Its
transfer-learning table puts a small dataset with low similarity to ImageNet at
"frozen features + small head" (wiki: ml-architectures/cnn §"When to Use What"),
which does not give per-pixel output. ImageNet weights are also learned on 3-channel
RGB photographs; our input is 8 channels of surface reflectance and indices at 20 m,
so the first convolution would have to be rebuilt anyway and the learned filters
would not match the physics `(consensus)`. The research reached the same conclusion
from the other direction — every pretrained checkpoint it examined was trained on
imagery 25–100× sharper than ours (`RESEARCH_DEEP_LEARNING.md` §7).

A U-Net is a CNN that outputs a class for **every pixel** rather than one label per
image. It has an encoder that shrinks the image while learning features, a decoder
that grows it back, and skip connections that copy fine detail from encoder to
decoder so that boundaries stay sharp. That last property is why it is the standard
for shoreline work.

### 2.2 Input: 8 channels, and one deliberate exclusion

| Ch | Source | Why |
|---|---|---|
| 0–4 | B02, B03, B04, B08, B11 reflectance | `s2_common.reflectance()`, offset −0.1 applied |
| 5 | NDWI | water vs vegetation |
| 6 | MNDWI | water vs built-up — the current baseline's own signal |
| 7 | NDVI | vegetation vigour |

**SCL is not an input channel.** SCL is ESA's own classifier output. Feeding it in
would let the network learn to copy it, and any agreement we then report with SCL
would be circular. SCL is used for two other things only: masking cloud (`ignore`
pixels) and as one of three independent label seeds (§2.4).

Per-channel normalisation (mean/std) is computed **on the training fold only** and
applied to the held-out lake. Fitting it on all six lakes first is the pre-split
preprocessing leak in the wiki's playbook (wiki: ml-training/evaluation §9
"Leakage families", item 4).

Indices 5–7 must be clipped to [−1, 1] before use. The current
`features_lake_year.csv` has `mean_mndwi = −61.63` for Bellandur 2025 because
`(g − s)/(g + s + ε)` explodes when the offset drives `g + s` near zero; the masks are
unaffected but the raw index values are not safe as features.

### 2.3 Output: 4 classes plus an ignore label

| Id | Class | Label definition used for training |
|---|---|---|
| 0 | open water | SCL = 6, corrected by hand |
| 1 | vegetation | SCL = 4, corrected by hand. **Inside the lake polygon this is water hyacinth by construction**; the model does not need to know that — the polygon does |
| 2 | bare / dry bed | SCL = 5 (not vegetated), corrected by hand |
| 3 | built-up | OpenStreetMap building footprints rasterised at 20 m, corrected by hand |
| 255 | ignore | cloud/shadow (`SCL_BAD`), unlabelled, outside the lake + collar |

Four classes rather than five: separating floating from terrestrial vegetation is
an *accounting* decision made by polygon membership after inference, not a spectral
distinction the model should be asked to learn. That keeps the labelling burden down
and the classes spectrally separable. Class 2 is the whole point of building this —
it is what lets §5 tell a drained bed from a built one.

`CrossEntropyLoss(ignore_index=255)` handles the ignore label directly (wiki:
ml-libraries/pytorch §"Segmentation").

### 2.4 Labels: three independent seeds, then hand correction

This is where the 10–16 student-days go, and it is the step that decides whether the
model can be defended.

**The rule: no label may be derived from MNDWI or NDVI alone.** A model trained on
the baseline's own thresholds is being trained to imitate the baseline, and any score
against those labels measures agreement with the threshold, not with the ground. The
seeds are:

1. **SCL** — ESA's classifier. Same sensor, different algorithm.
2. **OpenStreetMap building footprints** — different source entirely, gives class 3.
3. **Hand correction in QGIS** on the true-colour composite, with the published
   Bellandur/Varthur macrophyte mapping (doi 10.3390/rs12223843) open alongside for
   the hyacinth class.

Label the lake polygon **plus its 500 m collar** — the collar is where buildings are,
and `lake_masks()` already produces it. Everything outside is `255`.

Record who labelled what and when. The report will say these are hand-drawn labels,
not ground truth, and it needs to be able to show the audit trail.

### 2.5 Architecture

Three-level U-Net, deliberately small. Roughly 0.5 M parameters `(heuristic)`.

```
input  (B, 8, 128, 128)
  enc1: [conv3-bn-relu] ×2 → 16 ch   (128×128)  ──skip──┐
  pool                                              (64×64)   │
  enc2: [conv3-bn-relu] ×2 → 32 ch    (64×64)   ──skip──┤
  pool                                              (32×32)   │
  enc3: [conv3-bn-relu] ×2 → 64 ch    (32×32)   ──skip──┤
  pool                                              (16×16)   │
  bottleneck: [conv3-bn-relu] ×2 → 128 ch  (16×16), dropout 0.3
  up + concat skip → dec3: 64 ch     (32×32)
  up + concat skip → dec2: 32 ch     (64×64)
  up + concat skip → dec1: 16 ch    (128×128)
  1×1 conv → 4 logits                (B, 4, 128, 128)
```

Details that matter:

- `bias=False` on every conv that is followed by BatchNorm — BN has its own bias
  (wiki: ml-architectures/cnn §"Conv2d Parameter Reference").
- Upsampling by bilinear interpolation + 3×3 conv rather than transposed convolution;
  avoids checkerboard artefacts on smooth water `(consensus)`.
- Receptive field at the bottleneck is roughly 100 px ≈ 2 km at 20 m `(heuristic,
  from the RF recurrence in wiki: ml-architectures/cnn §"Receptive Field
  Calculation")` — enough to see that a vegetation patch is surrounded by water.
- If batches are tiny (≤4) swap BatchNorm for GroupNorm(8) `(heuristic)`; BN
  statistics get noisy with very few samples.

### 2.6 Loss: weighted cross-entropy plus Dice

Class balance is severe and varies by lake: Bellandur 2019 is 5 % water and 83 %
vegetation inside the polygon; Madiwala's collar is mostly built-up. The wiki's order
of preference for imbalance is *"weighted loss > sampling > SMOTE"* (wiki:
ml-architectures/regression-classification §"Decision Guide", step 4), and for
segmentation it adds Dice loss for boundary quality (wiki: ml-libraries/pytorch
§"Segmentation").

```
loss = CE(weight = w, ignore_index = 255) + Dice(over classes 0–3, ignoring 255)
w_c  = 1 / count_c, normalised so mean(w) = 1        # counts from the TRAINING fold only
```

Class weights, like normalisation, come from the training fold. If a class is still
being swallowed after this, switch the CE term to focal loss with γ = 2 (wiki:
ml-architectures/regression-classification §"Focal Loss") — but try weights first.

### 2.7 Patches, augmentation, and what not to augment

- **Patches:** 128 × 128, sampled with stride 64 over the bounding box of each
  lake's `inside ∪ collar`, keeping any patch with ≥ 10 % labelled pixels. Sankey
  Tank (≈ 18 × 18 px) gets one or two patches; Bellandur gets a few dozen. Expect
  roughly 200–400 labelled patches across all lake-years `(heuristic)`.
- **Augmentation:** the 8 rotations and reflections of a square only. Nadir satellite
  imagery has no natural "up", so all eight are valid. Apply the *same* transform to
  image and mask — torchvision's v2 transforms do joint image + mask transforms
  (wiki: ml-libraries/pytorch §"v2 advantages").
- **Do not** use `ColorJitter`, `RandomResizedCrop` with rescaling, MixUp or CutMix.
  Those are photograph augmentations (wiki: ml-architectures/cnn §"Data
  Augmentation"); they would corrupt the reflectance values that the indices are
  computed from `(consensus)`.

### 2.8 Splitting: leave-one-lake-out, nested

This is the decision that keeps the result honest, and it is non-negotiable.

Patches from the same lake share illumination, atmosphere, phenology and, across
years, the very same pixels. A random patch split puts near-copies in train and
test and reports a fake score. The wiki's leakage table names the symptom — *"train
≈ test and both very high → group leak → use `GroupKFold` on the entity ID"* (wiki:
ml-training/evaluation §9) — and the fix (wiki: ml-training/training-workflow §1
"GroupKFold").

```
outer:  GroupKFold(n_splits=6), groups = lake        → 6 folds, one lake held out each
inner:  from the 5 training lakes, hold out 1 lake   → early-stopping validation
```

The inner hold-out is a lake, not a year, for the same reason. This is nested CV
(wiki: ml-training/training-workflow §2 "When to reach for nested CV") — the inner
lake tunes stopping, the outer lake is touched once.

**Effective sample size for cross-lake generalisation is 6.** Say it in the report
before the panel says it. Report all six fold scores and their spread, not one mean.

### 2.9 Training configuration

| | Value | Source |
|---|---|---|
| Optimiser | AdamW, lr 1e-3, weight_decay 1e-4 | wiki: ml-architectures/regression-classification §"Complete PyTorch Training Loop" |
| Scheduler | `ReduceLROnPlateau(factor 0.5, patience 5)` on inner-val loss | wiki: ml-training/training-workflow §"PyTorch Training Loop" |
| Early stopping | patience 10 on inner-val loss, **restore best weights** | wiki: ml-training/training-workflow §"Common Gotchas" 9 |
| Batch | 8 patches | CPU-friendly; see GroupNorm note if smaller |
| Max epochs | 100 | early stopping decides the real number |
| Seeds | `set_seed(42)` plus `worker_init_fn` **and** `generator` on the DataLoader | wiki: ml-training/training-workflow §7 and gotcha 7 |
| Compute | ~1 h per fold on the 16-core CPU `(heuristic)`; minutes per fold on a Colab T4 | |

Six folds overnight on the laptop is realistic; a T4 makes it a coffee break.

### 2.10 Evaluation, and the pre-registered decision rule

Judge on the **held-out lake only**, and report:

| Metric | Why |
|---|---|
| Per-class IoU, per fold | the standard segmentation metric; shows which class fails where |
| Macro-F1 | unweighted across the 4 classes, so the rare water class counts (wiki: ml-training/evaluation §3) |
| Cohen's kappa | the remote-sensing convention; the wiki lists it among imbalance-safe metrics (wiki: ml-training/evaluation §2) |
| Normalised confusion matrix | specifically the **water ↔ vegetation** cell, which is the hyacinth question (wiki: ml-training/evaluation §3) |
| Expected Calibration Error after temperature scaling | probabilities feed §5; scikit-learn 1.6.1 predates native temperature scaling, so use the manual `TemperatureScaling` module fitted on inner-val logits (wiki: ml-architectures/regression-classification §"Temperature Scaling"; ECE from wiki: ml-training/evaluation §4) |

Never accuracy alone (wiki: ml-architectures/regression-classification §"Decision
Guide", step 8).

**Two sanity checks before believing any number:**

1. **Shuffle-label negative control** — train once on permuted labels; macro-F1 must
   fall to chance. If it does not, the pipeline leaks (wiki: ml-training/evaluation
   §9 "Negative-control / shuffle test").
2. **Baseline comparison on the same folds** — the U-Net, the Random Forest (§3) and
   the MNDWI/NDVI rule scored on identical held-out lakes. Compare paired fold scores
   with the Wilcoxon signed-rank test (wiki: ml-training/evaluation §6), and say
   plainly that six pairs gives it little power.

**The decision rule, fixed now so it cannot be bent later:** the wiki's rule is *"if
the simple baseline works within 2 % of deep learning, keep it"* (wiki:
ml-architectures/regression-classification §"Decision rule"). So: **if the U-Net's
mean held-out macro-F1 is not more than 2 points above the Random Forest's, the
report recommends the Random Forest and presents the U-Net as the deep comparison
that was built, tested and found not to earn its complexity.** That is a legitimate
capstone result. It satisfies the guide's requirement either way.

---

## 3. Layer A′ — the Random Forest that the U-Net must beat

Same 8 features per pixel, same seed labels, same six leave-one-lake-out folds.
`RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1)`
(wiki: ml-training/evaluation §1 "class_weight='balanced'"). Trees need no feature
scaling (wiki: ml-architectures/regression-classification §"Feature Scaling").
Subsample training pixels to ~200k per fold to keep it fast; it is a pixel model, so
sample size is not the constraint here.

It gets *no* spatial context — one pixel's 8 numbers — which is exactly the
capability the U-Net adds. If the U-Net wins, the confusion matrices will show it
winning on the water ↔ vegetation cell, where context matters. If it does not win,
context did not matter at 20 m for these lakes, and that is worth knowing.

---

## 4. Layer B — the geo-fence

The land-cover map is not the result. The result is what sits inside the fence.

**Fence.** Both 30 m and 75 m rings around each lake polygon, as bracketing cases,
with the February 2026 tiered law disclosed (`RESEARCH_DEEP_LEARNING.md` §4). Four
of the six lakes fall in the 30 m tier; Ulsoor and Sankey Tank need a lookup.

**Two intrusion measures per lake-year, kept separate:**

| Measure | From | Years | Resolution |
|---|---|---|---|
| `built_frac_30m`, `built_frac_75m` | U-Net class 3 pixels inside the ring | 2019–2025 | 20 m — a 30 m ring is 1.5 px wide, so treat this as coarse |
| `bldg_count_30m`, `bldg_overlap_ha_30m` (and 75 m) | `GOOGLE/Research/open-buildings-temporal/v1` footprints, overlap-fraction against the ring, STRtree-indexed | 2016–2023 | 4 m |

The vector measure is the strong one; the raster measure is the one that reaches
2024–2025. Report both, say which is which, and never interpolate the 2024–2025 gap
in the vector series.

**Pre-flight check before trusting Open Buildings:** run one lake with known
construction and confirm non-zero detections in the 30–250 m ring. The research
found no evidence of a water-proximity filter, but this test costs an hour and a
silent zero would invalidate the whole layer.

**Footprint composition per lake-year**, from U-Net output restricted to the lake
polygon: `water_frac`, `veg_frac` (= hyacinth, by polygon membership), `bare_frac`,
`built_frac`. These replace the current three columns from the fixed thresholds.

---

## 5. Layer C — the risk model, and why it is not XGBoost

### 5.1 The model

A **linear mixed-effects model** with a random intercept per lake:

```
y[lake, year] = β0 + β1·year + β2·rain_365d + β3·bldg_count_75m_prev + u[lake] + ε
u[lake] ~ N(0, σ²_lake)
```

`statsmodels.MixedLM` (to install). Target `y` = `built_frac_30m` the following year,
or its year-on-year change — a continuous, measurable quantity, **not** an
encroachment label that does not exist.

Why this and not a tree ensemble: every OpenStreetMap building and road column in the
current table is constant within a lake (Bellandur is `128 / 733 / 2079` in all seven
years). Those columns *are* a lake-identity label. A tree model splits on them,
memorises which lake it is looking at, and returns a confident number that means
nothing — the group-leak pattern again (wiki: ml-training/evaluation §9). The mixed
model absorbs the same columns into the random intercept, which is the honest
degradation. Note that `VarianceThreshold` would *not* catch this — the columns vary
across lakes, only not within them (wiki: ml-training/feature-selection §"Variance
Threshold" removes globally constant features, which these are not).

Say in the report that estimating between-lake variance from six clusters is below
the usual 15–50 minimum, so `σ²_lake` will be wide. That is the point.

### 5.2 The interval

Wrap predictions in **jackknife+** conformal intervals — leave-one-lake-out refits,
which cost nothing at this size and avoid sacrificing a calibration set (`mapie` to
install, or ~40 lines by hand). State every time: the interval covers **our own
derived target**, not true encroachment; only marginal coverage is claimable because
six clusters plausibly break exchangeability; and jackknife+ guarantees ≥ 1 − 2α,
so a 90 % target promises 80 %.

### 5.3 What is deliberately left out, and why

- **XGBoost + SHAP** — only as a one-figure exhibit *showing* the lake-identity
  leakage: train it, show near-perfect in-sample fit, show it collapse under
  leave-one-lake-out. That is its honest use here.
- **Survival / Cox** — six subjects and zero-to-one events; a paragraph naming it and
  the events-per-variable rule, no fitted model.
- **LSTM / ConvLSTM** — six sequences of length seven. Not trainable.

---

## 6. Layer D — the desilting rule

This is a rule, not a model, and it is where the four-class output earns its keep.
For each lake-year, applied to the footprint fractions from §4:

| Pattern across consecutive years | Reading |
|---|---|
| `water_frac` ↓ and `bare_frac` ↑ and `built_frac` in the polygon flat, then `water_frac` recovers | **drained for desilting** — corroborate with tender records or news |
| `water_frac` ↓ and `built_frac` in the polygon or 30 m ring ↑ and no recovery | **encroachment signal** — go to the vector intrusion count |
| `water_frac` ↓ and `veg_frac` ↑ | **hyacinth spread**, not lake loss |
| `water_frac` ↓ and `rain_365d` low across all lakes | **drought** — check Sankey Tank, the one rain-fed lake |

Bellandur 2021 hits row 1. The current pipeline could only ever hit "water went
down." Documentary evidence is part of the rule because the research established that
no statistical test at n = 7 and no elevation dataset can make this distinction
alone (`RESEARCH_DEEP_LEARNING.md` §12).

---

## 7. Risk review of this design

Scored with the `ml-review` severity scale. These are the things a panel will find
if the build skips them.

| Sev | Risk | Mitigation built in |
|---|---|---|
| **CRITICAL** | Labels seeded from MNDWI/NDVI → the model imitates the baseline and every score is circular | §2.4: SCL + OSM + hand correction only; MNDWI/NDVI appear as *inputs*, never as *labels* |
| **CRITICAL** | Random patch split → near-duplicate pixels in train and test → fake score | §2.8: nested leave-one-lake-out; shuffle-label control |
| **HIGH** | Normalisation stats or class weights computed on all lakes before splitting | §2.2, §2.6: computed per training fold |
| **HIGH** | SCL fed as an input channel → model copies ESA's classifier | §2.2: SCL is mask and seed only |
| **HIGH** | Reporting the U-Net alone, with no baseline on the same folds | §3 and the pre-registered 2-point rule |
| **MEDIUM** | 2021 (drained) and 2025 (refilled) are out-of-distribution years; a fold that holds out Bellandur will see both extremes only at test time | Expected and reportable; the per-fold table will show it |
| **MEDIUM** | Open Buildings ends 2023; the vector intrusion series is blind to 2024–2025 | §4: two measures, gap stated, never interpolated |
| **MEDIUM** | Index columns in the existing feature table are numerically corrupted | §2.2: clip to [−1, 1]; fix `s2_common.indices()` before reuse |
| **LOW** | CPU-only training time | ~6 h for all folds; Colab if impatient |

Dimensions reviewed: architecture, data pipeline and splits, loss, evaluation and
baseline, risk model. Not reviewed: deployment and serving — there is none.

---

## 8. Build order, with upgrade triggers

1. **Fix the index bug and re-fetch B12.** `s2_common.indices()` clip; add `swir22`
   to `BANDS` in `src/02_fetch_satellite.py:49`. Half a day.
2. **Make the labels.** §2.4. This is the long pole: 6–10 days. Nothing downstream is
   trustworthy without it.
3. **Random Forest on the folds.** §3. One day. This is the number to beat, and it
   also validates that the folds and labels are sane.
4. **U-Net on the same folds.** §2. Three to five days including debugging.
   → *Upgrade trigger:* if water ↔ vegetation confusion stays high on the
   Bellandur/Varthur folds, the fix is **Sentinel-1 VV/VH as channels 8–9**, not a
   bigger network — radar sees hyacinth roughness that optical does not
   (`RESEARCH_DEEP_LEARNING.md` §2, item 6).
5. **Geo-fence.** §4. Three to four days, including the Open Buildings pre-flight.
6. **Mixed model + conformal.** §5. Two to three days.
7. **Cross-checks.** Dynamic World `flooded_vegetation` agreement on the exact seven
   scene dates; JRC Global Surface Water `recurrence` for 2019–2021. Two days.
   → *Upgrade trigger:* if the RTI request for the SSLR/Minor Irrigation Web GIS
   boundaries and encroachment status succeeds (`RESEARCH_DEEP_LEARNING.md` §11),
   move to 20–40 lakes with `StratifiedGroupKFold` and a real lake-level label.
   Everything above survives that change unaltered; only the folds get more of them.

Roughly 20–28 student-days end to end. Step 2 is the one that cannot be compressed.

---

## 9. Dependencies

Keep the deep-learning stack out of the main pipeline. `requirements-dl.txt`:

```
torch            # 2.10.0+cpu is installed; GPU build on Colab
torchvision      # installed; v2 transforms for joint image+mask augmentation
scikit-learn     # 1.6.1 installed; note: no native temperature scaling before 1.8
statsmodels      # MixedLM — to install
mapie            # jackknife+ — to install, or hand-roll
pymannkendall    # optional, for the indicative trend tests
geopandas        # installed; geo-fence
```

Not needed: `imbalanced-learn` (class weights suffice — wiki:
ml-architectures/regression-classification §"Decision Guide" step 4), `optuna`
(there is no hyperparameter budget to spend on six folds; the configuration in §2.9
is fixed and reported as such), `segmentation_models_pytorch` (the U-Net is ~80
lines and owning it makes the viva easier).

---

## Sources

- wiki: ml-architectures/cnn — §"Architecture Selection Guide", §"Transfer Learning
  · When to Use What", §"Conv2d Parameter Reference", §"Receptive Field
  Calculation", §"Data Augmentation"
- wiki: ml-architectures/regression-classification — §"Decision Guide", §"Class
  Imbalance Handling", §"Focal Loss", §"Calibration · Temperature Scaling",
  §"Feature Scaling", §"Complete PyTorch Training Loop", §"When to Use · Decision
  rule"
- wiki: ml-training/evaluation — §1 class_weight, §2 metrics, §3 multi-class and
  confusion matrix, §4 calibration and ECE, §6 statistical comparison, §9
  leakage playbook and shuffle test
- wiki: ml-training/training-workflow — §1 GroupKFold, §2 nested CV, §7
  reproducibility, §"PyTorch Training Loop", §"Common Gotchas" 1, 7, 9
- wiki: ml-libraries/pytorch — §"Segmentation" (ignore_index, Dice), §"v2
  advantages" (joint transforms)
- wiki: ml-training/feature-selection — §"Variance Threshold"
- `RESEARCH_DEEP_LEARNING.md` (this folder) — §2 ranking, §4 legal buffer, §7
  rejections, §11 ground truth, §12 elevation
- Bareuther et al. 2020, macrophyte cover in Bellandur and Varthur,
  doi 10.3390/rs12223843
- `src/02_fetch_satellite.py:274` `lake_masks()`, `src/s2_common.py` `indices()`
