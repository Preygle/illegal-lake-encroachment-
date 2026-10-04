# Deep learning and geo-fencing for the lake encroachment project: what to build, and why

Research note, 17 September 2026. Thirty candidate methods were researched
independently by agents answering the same thirty-five questions each, with web search.

Numbers taken from published papers are marked **[verify]** wherever the agent that
found them could not confirm the figure against a primary source. That applies to
**26 of the 30 items** for `published_performance`. Do not quote a performance figure
from this note to the panel without checking it yourself.

The baseline any new method has to beat: the current pipeline's fixed **MNDWI > 0**
water mask (Xu 2006) with **NDVI > 0.30** vegetation, counted inside a fixed
OpenStreetMap polygon. 6 lakes, 7 years, 42 rows.

---

## 1. The constraint that decides everything

**The project measures water. Encroachment is a building standing where the lake
legally is. These are different questions, and the gap between them is where the
project currently loses.**

Water area has three explanations and only one of them is encroachment:

| Observation | Innocent explanation | Guilty explanation |
|---|---|---|
| Water area falls | Lake drained for desilting | Lake bed built on |
| Water area falls | Drought / weak monsoon | Lake bed built on |
| Water area stays low but vegetation is high | Water hyacinth mat | — |

Bellandur ran 15.7 ha (2019) → 0.4 ha (2021) → 107.9 ha (2025). The project already
proved that 2021 was a drained bed, not an encroached one. A model trained to predict
water area will learn desilting schedules and monsoon strength, and will call them
encroachment.

Three consequences follow, and they shape every recommendation below:

1. **The deep learning model should classify land cover, not predict encroachment.**
   There are ~900,000 pixels per scene and zero encroachment labels. A per-pixel
   segmentation model is trainable; a 42-row encroachment classifier is not.
2. **The encroachment signal lives in the geo-fence, not in the water mask.** A
   building footprint inside the statutory buffer is legible, desilting-immune
   evidence. Water loss is not.
3. **The tabular model must treat 42 rows as 6 clusters × 7 repeated measures.**
   Every OpenStreetMap building and road feature is constant within a lake — Bellandur
   is `128 / 733 / 2079` in all seven years. Those columns *are* a lake-identity label,
   and a tree model will split on them and memorise which lake it is looking at.

This is the answer to give when the panel asks "why didn't you just train a CNN to
detect encroachment?".

---

## 2. Ranked recommendation

| # | Method | What it gives the project | Effort | Honest expectation |
|---|---|---|---|---|
| 1 | **U-Net, 4-class segmentation** | The guide's deep learning requirement, and the only method that outputs a bare-lake-bed class — the raw material for a desilting-vs-encroachment rule | 10–16 days | Visibly cleaner hyacinth/water separation than the fixed threshold. Cannot be validated against ground truth, only against itself and external products |
| 2 | **Geo-fence intrusion test** | The actual encroachment signal, legally legible and immune to desilting | 4–5 days | A per-lake building-intrusion series for 2016–2023. The strongest single deliverable in this list |
| 3 | **Mixed-effects panel model + conformal intervals** | The statistically correct treatment of the 42 rows, with honest uncertainty | 2–5 days | Wide intervals and a modest trend estimate. That *is* the result — it is honest where XGBoost would fabricate confidence |
| 4 | **Random Forest pixel classifier** | The classical baseline the U-Net must beat | 4–7 days | A fair comparison. May well match the U-Net, which is a reportable result |
| 5 | **Independent cross-checks** (Dynamic World, JRC Global Surface Water) | Breaks the circularity of scoring your own threshold against itself | 1.5–3 days | Agreement/disagreement percentages, not accuracy. Still the cheapest credibility win available |
| 6 | **Sentinel-1 SAR upgrade** | ~25–30 observations/year instead of 1, which makes the trend tests valid for the first time | 5–8 days | The structural fix for limitation 4. Introduces its own thresholding problem |
| 7 | **Mann-Kendall + Sen's slope + Pettitt** | Named trend and change-point tests on the water series | 2–3 days | Indicative only at n=7. Label them so |
| 8 | **Otsu / sub-pixel water fraction** | Sensitivity check on the fixed threshold | 1–2 days | Few percent change on big lakes; unstable on Sankey Tank and Bellandur 2021, which is itself the finding |
| 9 | **Frozen foundation encoders** (TerraMind v1.0 or Clay v1.5 only) | A modern deep representation, if there are spare days | 4–6 days | Small, unverifiable gain. Only these two accept the cached band subset under a permissive licence |

A solo student with roughly a month can deliver **1, 2 and 3**, plus **4** as the
comparison and **5** as the credibility layer. Everything below that belongs in the
report as "considered, with reasons" — which is itself a deliverable, because the
guide asked about deep learning and geo-fencing and the report must show both were
taken seriously.

**Rejected, with reasons, in §7.**

---

## 3. Plan for #1: the U-Net that answers the guide

A four-class per-pixel segmentation model on the project's own 20 m, 6-band data.
Chosen over every other deep option because it inherits no incompatible pretrained
weights and no foreign resolution assumption, and because it is the only candidate
whose output makes the desilting confound addressable.

**Classes.** Open water · water hyacinth / macrophyte · bare lake bed · built-up.
The third class is the point. The current pipeline reports water / vegetation /
"other"; splitting "other" into bare bed and built-up is what lets a later rule say
*"the water went away AND the bed is bare"* (desilting) versus *"the water went away
AND the bed is built"* (encroachment).

**Inputs.** The six cached bands (B02, B03, B04, B08, B11, SCL) plus MNDWI and NDVI as
derived channels, in patches of 128 × 128 pixels at 20 m.

**Labels.** This is the real cost, and it must not be circular. Seed from spectral
rules, then **hand-correct** — a model trained purely on MNDWI/NDVI output learns to
imitate the baseline it is meant to beat, and any agreement score then measures
agreement with your own threshold, not with reality. Cross-check the hyacinth class
against the published Bellandur/Varthur macrophyte mapping (doi 10.3390/rs12223843).
Budget 6–10 of the 10–16 days here.

**Architecture.** Small: 3 down blocks (16/32/64 filters), 3 up blocks, batch
normalisation, dropout 0.3. Roughly 100k–500k parameters. Anything larger will
memorise six lakes.

**Splitting.** **Leave-one-lake-out, six folds.** Never random patch splits —
neighbouring patches overlap and patches within one lake-year share illumination,
atmosphere and phenology, so a random split leaks badly and will report a fake 95%.

**Evaluation.** Per-class IoU and F1, reported per held-out lake, plus the confusion
between water and hyacinth specifically, which is the class pair that matters. State
plainly that these are measured against hand-drawn labels, not ground truth.

**Effective sample size.** Say it in the report before the panel says it: for
cross-lake generalisation this is **n = 6**, not 42 and not 900,000.

**Compute.** Minutes on a free Colab T4. Workable on the laptop CPU.

---

## 4. Plan for #2: the geo-fence

This is the deliverable that actually addresses "illegal encroachment", and the
research suggests it is stronger than the deep learning model as a scientific result.

**The fence, and the legal position.** This is contested and has changed twice:

- Bengaluru Master Plan: **30 m**, the long-standing default.
- National Green Tribunal, O.A. 222/2014 (order dated 4.5.2016): **75 m** for lakes,
  50/35/25 m for rajakaluves.
- Supreme Court, *Mantri Techzone Pvt. Ltd. v. Forward Foundation*, decided 5.3.2019:
  **set aside the NGT's general 75 m direction**, but sustained it specifically
  against the two encroaching parties at the Bellandur–Agara site. The common
  reporting that the Court "reinstated 30 m city-wide, prospectively" is a
  simplification the primary text does not support. **[verify before quoting]**
- Karnataka Act No. 19 of 2026, gazetted 18 February 2026: replaced the flat 30 m with
  a **size-tiered** buffer measured *from the revenue boundary*, 0 m (≤0.05 acre)
  rising to **30 m (>100 acres)**.

Under the 2026 tiers, four of the six lakes are in the top band:

| Lake | Footprint | Acres | Tier |
|---|---|---|---|
| Bellandur | 315.44 ha | 779.5 | 30 m |
| Varthur | 153.96 ha | 380.4 | 30 m |
| Madiwala | 54.88 ha | 135.6 | 30 m |
| Hebbal | 45.44 ha | 112.3 | 30 m |
| Ulsoor | 38.12 ha | 94.2 | below 100 — look up the tier |
| Sankey Tank | 12.40 ha | 30.6 | below 100 — look up the tier |

**What to do:** test **30 m and 75 m as bracketing cases**, disclose the 2026 tiering,
and state that lakes falling in the 1–12 m tiers have a legal buffer that is
**sub-pixel at 20 m and therefore untestable with Sentinel-2**. That last sentence is
a finding, not a failure.

**The intrusion test.** Vector footprints only — a 30 m buffer is 1.5 pixels wide, so
raster counting inside the fence is not viable. Use **overlap fraction**, not
centroid-in-polygon, and report a bracketed range (any-intersection vs >50% overlap)
rather than a single number. Index with `STRtree`.

**Source.** `GOOGLE/Research/open-buildings-temporal/v1` — annual 2016–2023, 4 m
effective resolution (7–8 pixels across a 30 m buffer), CC-BY 4.0 / ODbL. Covers
India. **It stops at 2023**, so 2024 and 2025 have no coverage and that gap must be
stated, not interpolated. Microsoft Global ML Building Footprints is **not**
multi-temporal and cannot help. OpenStreetMap history via ohsome can reconstruct past
snapshots but risks showing an import step-jump rather than real construction.

**One pre-flight check, before trusting any count.** A claim circulated in secondary
sources that Google's pipeline suppresses detections within 250 m of water bodies.
Four primary sources were checked — the Earth Engine catalogue, the official Open
Buildings site, and both Sirko et al. papers (arXiv 2107.12283, 2310.11622) — and
**none mentions any such rule**. It is recorded as checked-but-unconfirmed. Before
building on this dataset, run one lake with known nearby construction and confirm
non-zero detections exist in the 30–250 m ring. If the filter did exist, every lake
would return near-zero intrusion and the pipeline would look perfectly healthy while
being completely wrong.

**Boundary caveat.** The 2026 law measures from the *revenue boundary*. The project
uses an OpenStreetMap polygon, which has no statutory standing. No login-free
machine-readable authoritative boundary was found at KTCDA, Survey of India, Bhuvan,
National Wetland Atlas 2024 or the BBMP Lakes Monitoring System. Hand-digitising RTC
records is 0.5–1.5 days per lake. Either do that, or state the limitation precisely.

---

## 5. Plan for #3: the honest tabular model

**Use a linear mixed-effects model, not a tree ensemble.**

- Random intercept per lake, slope on year, rainfall as a time-varying covariate.
- `statsmodels` MixedLM, or `bambi`/PyMC for the Bayesian version. With Bayesian
  priors, use half-normal or half-Cauchy on the standard deviation — never
  inverse-gamma with this few groups.
- The constant OpenStreetMap features are **absorbed into the random intercept**, which
  is graceful degradation. In XGBoost the same columns become hidden lake-identity
  splits, which is silent failure.
- Be upfront: estimating between-lake variance from **6 clusters** is below the
  commonly cited 15–50 minimum, so the estimate is wide and imprecise. Report it that
  way. **[verify the exact thresholds in McNeish & Stapleton 2016]**

**Wrap it in conformal prediction.** Cheapest credibility win in the entire research.
Use **jackknife+ or CV+** (not split conformal — you cannot spare a calibration
hold-out from 42 rows), via `MAPIE`. Two things must be said every time an interval
is reported:

> The interval covers the project's **own derived label**, not true encroachment
> state, because no ground truth exists to calibrate against. Only marginal (pooled)
> coverage is claimable; the 6-lake clustering plausibly violates exchangeability, so
> per-lake conditional coverage is not.

Note also that jackknife+/CV+ guarantee only ≥1−2α in the worst case, so a nominal
90% target guarantees ≥80%.

---

## 6. Independent cross-checks

The project's outcome variable is derived from its own MNDWI threshold, so every model
trained on it is scored against that threshold rather than against reality. External
products break that loop. None of them is ground truth; all give *agreement*, not
*accuracy*.

- **`GOOGLE/DYNAMICWORLD/V1`** — the best of them. 10 m, computed per scene so it can
  be matched to the exact seven acquisition dates, and it has an explicit
  **`flooded_vegetation`** class, the closest ready-made proxy for Bellandur's 260 ha
  hyacinth mat found anywhere in this research. Caveat: trained on Sentinel-2, so it
  is algorithmically independent, not sensor-independent.
- **`JRC/GSW1_4/*`** — different sensor (Landsat), institution and algorithm. Its
  `seasonality` and `recurrence` layers are the best external evidence for arguing a
  one-year dip against high multi-decadal recurrence, i.e. the desilting case. **Ends
  in 2021** in Earth Engine, so 2022–2025 have no cross-check. A v1.5 extension to
  2024 is advertised on the JRC portal but was not confirmed as mirrored into Earth
  Engine — **[verify]**.
- **`ESA/WorldCover/v100` / `v200`** — the only genuinely sensor-independent option
  (uses Sentinel-1 too), but only 2020 and 2021 exist, and it has **no
  flooded-vegetation class**, so it will fold hyacinth into water, wetland or
  grassland. Disagreement with it would be a legend artefact, not a finding.

---

## 7. One line for each method the guide suggested, and each one rejected

**The guide's two asks:**

- **Deep learning for image-based model training.** Adopted, as a four-class U-Net
  segmenting water, hyacinth, bare lake bed and built-up from the project's own
  Sentinel-2 bands — trained per-pixel, where there is enough data, rather than
  per-lake-year, where there is not.
- **Geo-fencing.** Adopted, in the legal buffer-zone sense: the statutory buffer as
  the fence, and multi-temporal building footprints intersected with it as the
  intrusion test. The GPS/IoT real-time alerting sense of the term was considered and
  ruled out — it needs a deployment and alerting layer, not satellite analysis.

**Rejected, with reasons:**

- **XGBoost with SHAP.** 42 rows from 6 clusters, with constant per-lake features that
  act as a lake-identity label. It will fit in-sample, produce professional-looking
  SHAP plots, and generalise poorly. Usable only as a *demonstration of the leakage*.
- **ConvLSTM / LSTM.** Six independent sequences of length 7. Going per-pixel does not
  help — pixels within a lake-year share the same timestamps and are spatially
  autocorrelated, so the effective count stays at 6. Overfits by construction.
- **Segment Anything (SAM / SAM2).** Cleanest licence of the image methods, but needs
  RGB input, which discards the SWIR band the whole MNDWI baseline rests on, and is
  expected to over-segment exactly the amorphous hyacinth boundaries that matter.
  "Zero labels" is a bait-and-switch: the cost moves to manual prompt placement per
  lake-year.
- **Pretrained Siamese change detection.** LEVIR-CD (0.5 m), WHU-CD (~0.2 m) and
  S2Looking (0.5–0.8 m) are 25–100× sharper than 20 m Sentinel-2. Note that the "S2"
  in S2Looking does **not** mean Sentinel-2 — it is GaoFen/SuperView/BeiJing-2
  imagery. Their published IoU scores describe a different problem on a different
  sensor. A small from-scratch Siamese network is defensible; their checkpoints are not.
- **Positive-unlabelled learning.** Needs hundreds to tens of thousands of positives;
  single digits are obtainable at year-stamped granularity. Worse, the premise is
  inverted: the government survey figures (§10) put encroachment at **730 of 837**
  water bodies in Bengaluru Urban District, and **472 acres across 136 surveyed BBMP
  lakes**. "A few positives hidden in many negatives" is simply not the situation, and
  the six study lakes are selection-biased toward the famous, litigated ones, which
  violates the SCAR assumption every PU estimator depends on.
- **Survival analysis / Cox.** Six subjects, zero-to-one observed events. With zero
  events the partial likelihood is flat and the model cannot be fit at all. Censoring
  is not merely imprecisely dated — it is undefined, because "not yet encroached"
  cannot be verified except through the same confounded proxies. Worth one paragraph
  naming the framework and showing via the events-per-variable rule why it fails.
- **`JRC/GHSL/P2023A/GHS_BUILT_S`.** One 100 m cell covers 25 of the project's pixels
  and a 30 m buffer is under a third of one cell. Structurally unusable at buffer
  scale. And of the seven study years, only 2020 and 2025 sit on the epoch grid —
  2020 is interpolated and **2025 is a forward extrapolation computed before 2025
  happened**. Citing it would mean presenting a projection as a measurement.
- **VIIRS nighttime lights.** At 463.83 m/pixel, one pixel is ~21.5 ha. The 30 m ring
  around Sankey Tank is ~3.6 ha; around Bellandur, the largest lake, ~19 ha. **Less
  than one pixel for every lake in the study.** A clean null result at buffer scale.
- **BFAST / BFAST Monitor.** Decomposes a seasonal cycle within each year; the original
  work used ~23 observations per year. With one observation per year there is no
  seasonal component to decompose. Structurally disqualified, not merely underpowered.
- **Getis-Ord Gi\*.** ArcGIS's own documentation sets a ≥30-feature reliability floor;
  8–16 ring sectors per lake-year is below it. And ~336 pooled tests would produce
  ~16–17 false hotspots at uncorrected α = 0.05 by chance. If run at all, run it
  pooled with Benjamini-Hochberg correction and label it exploratory.
- **Snorkel weak supervision.** The label model assumes conditionally independent
  labelling functions; MNDWI, NDVI and GHSL all derive from the same reflectance, OSM
  is static, and VIIRS is far too coarse. Five labelling functions collapse to two or
  three, and the output would mostly restate *"MNDWI and NDVI already agreed"*.
  Defensible only with that disclosed.

---

## 8. Evaluation protocol, identical for every method

Any method not judged this way cannot be compared with the others:

- **Leave-one-lake-out**, six folds, for anything that generalises across lakes.
  Never random k-fold on 42 rows drawn from 6 clusters.
- **Leave-one-year-out** as a sensitivity check for anything temporal.
- Report the **effective sample size** (6 lakes) alongside the nominal one.
- Report **conformal intervals**, with the proxy-label caveat stated every time.
- Every external comparison reported as **agreement**, never as accuracy.
- **A null result is reported as a result.**

---

## 9. Setup

Keep deep learning dependencies out of the main pipeline so the existing run stays
light. A separate `requirements-dl.txt` holding `torch` (CPU locally, GPU on Colab),
`scikit-learn`, `statsmodels`, `mapie`, `pymannkendall`, and `geopandas`. Add
`earthengine-api` only if the Dynamic World and Open Buildings cross-checks are built.

**One fix needed first:** `src/02_fetch_satellite.py:49` caches
`["blue","green","red","nir","swir16","scl"]` — B12 (`swir22`) is absent. DeepWaterMap
and WatNet both require it, and several foundation encoders do too. One-line change
plus a re-fetch.

**One bug found during review, unrelated to this research:** `mean_mndwi`, `mean_ndwi`
and `mean_ndvi` in `features_lake_year.csv` are corrupted — Bellandur 2025 has
`mean_mndwi = −61.63`, Sankey 2023 has `mean_ndwi = −111.35`. After the −0.1 offset,
reflectance goes negative over dark pixels, so `g + s + EPS` lands near zero and the
ratio explodes. The masks are unaffected; the three mean columns are unusable as
features and currently feed the F25 correlation figure.

---

## 10. What is not verified

- `published_performance` is unconfirmed in **26 of 30** items.
- `applies_to_indian_urban_lakes` is unconfirmed in **17 of 30** — which, read the
  other way, is the project's novelty claim: almost none of these methods has been
  published on Indian urban lakes, and none on encroachment against a legal buffer.

---

## 11. Ground truth: what the government surveys actually say

The project's four stated limitations include "no real ground truth" and "only 42
rows". These are the same problem, and this section is the only route out of it.

**A circulating figure of "159 of 202 lakes encroached" could not be verified** and
should not be cited. No primary source, government page or news article at any tier
states it. What *is* sourced:

| Tier | Source | Finding |
|---|---|---|
| 1 | Times of India, 11 Jan 2026 | **730 of 837** water bodies in Bengaluru **Urban District** under illegal occupation. Survey by Minor Irrigation Dept + State Survey Settlement and Land Records (SSLR), launched March 2024, ~95% complete. Drone / satellite / ArcGIS / rover, with **per-lake GPS boundaries uploaded to an internal Web GIS platform** |
| 1 | New Indian Express, 29 Jul 2025 | **183 lakes** under BBMP jurisdiction, 136 surveyed, 33 pending, **472 acres encroached** (288 government agencies, ~184–188 private), citing Supreme Court directions of 21 Jul and 1 Aug 2023 |
| 2 | Bangalore Mirror, no named source in article | 202 lakes, 851.11 acres encroached, 19 disappeared |

**These are different populations and must not be conflated.** Bengaluru Urban
District (~834–837 water bodies) is a materially larger set than former BBMP city
limits (~183–211). There is no single authoritative lake count; statewide the figure
is ~41,849. One news reference indicates BBMP has been superseded by the Greater
Bengaluru Authority — **[uncertain]**, not confirmed against the Act.

**The highest-value, lowest-cost action in this entire research programme:** the SSLR
/ Minor Irrigation **Web GIS platform** described in the TOI report holds per-lake GPS
boundaries *and* encroachment status, is government-produced and court-supervised, and
would simultaneously solve the boundary problem (§4) and the ground-truth problem. Its
public or machine-readable availability is unconfirmed. **Submit a direct request or
RTI application to SSLR / Minor Irrigation Department.** A few rupees and a few weeks
of waiting, against a deliverable nothing else in this note can match.

**No machine-readable boundary set was found** for BBMP LMS (lms.bbmpgov.in was
unreachable, TLS error), KTCDA, or any Karnataka geoportal. Overpass extraction from
OpenStreetMap is feasible but known to be incomplete for smaller and informal tanks.

**On scaling, one correction to the obvious hope.** The marginal imagery and compute
cost of going city-wide is near zero — the whole city sits inside the MGRS tile
already indexed. The real cost is boundary curation: **40–100 student-days** for a
rigorous ~200-lake set, with a partial **20–40 lake scale-up (10–20 student-days)** as
the realistic fallback. But 200 lakes × 7 years is **not** 1,400 independent labelled
examples. The survey label is a single cross-sectional snapshot (~2024–2026), so the
independent unit count stays at ~183–211 lakes, and the clustering problem from the
current 42-row setup **reappears at larger scale** — now with a real label attached,
which is a genuine improvement, but not a full fix. Plan the mixed-effects structure
(§5) accordingly rather than assuming the sample-size problem dissolves.

---

## 12. The desilting confound cannot be solved with elevation

Tested and confirmed negative. This closes off an approach that looks obvious:

- **`USGS/SRTMGL1_003`** is fixed at a single February 2000 radar overflight. It cannot
  show any change across 2019–2025.
- **`JAXA/ALOS/AW3D30/V4_1`** and CartoDEM version increments are **reprocessing and
  void-filling of one fixed historical stereo-imagery epoch** each (ALOS PRISM
  ~2006–2011; the Cartosat-1 campaign). A higher version number does not mean a later
  elevation date — an easy and costly misreading.
- Even given a second epoch, public DEM vertical error of roughly **5–16 m**
  **[verify]** dwarfs the **1–3 m** bed-relief signal that desilting a shallow urban
  tank would produce. The hypsometric stage–area–volume approach therefore fails at
  this scale too.
- **ICESat-2 ATL13** Version 6 is retired in favour of Version 7 (active mission), but
  its six narrow, widely-spaced ground tracks make usable returns over lakes of
  12–315 ha a matter of chance. No validation study at this lake size was found.
  Likely sparse to unusable.

**Therefore:** resolve desilting versus encroachment by **documentary evidence** —
BBMP/BWSSB desilting tender records, works orders, contemporaneous news — or by the
**Sentinel-1 within-year V-shape** test (§2, #6). Not by elevation.

---

Raw research output: 30 JSON files, 35 fields each, in `results/`, produced by
independent agents on 17 September 2026. Each file lists the fields its agent could not
verify in an `uncertain` array; the same caution applies to every number marked
**[verify]** above.
