# Technical Language and Dataset Naming Rules

## 1. Overall Technical Level

This is a **B.Tech Computer Science capstone project**.

The project can and should use normal Computer Science and AI terminology where appropriate.

It is completely acceptable to use terms such as:

* Machine Learning
* Deep Learning
* Computer Vision
* Image Processing
* Regression
* Classification
* Random Forest
* XGBoost
* Support Vector Machine
* Neural Network
* CNN
* Feature Engineering
* Training Dataset
* Validation Dataset
* Test Dataset
* Accuracy
* Precision
* Recall
* F1-score
* RMSE
* Python
* NumPy
* OpenCV
* GeoPandas
* Google Earth Engine
* Satellite Imagery
* Remote Sensing
* GIS
* NDVI
* NDBI
* NDWI
* LST
* SAR

These are normal technical terms for a Computer Science capstone and do NOT need to be avoided.

The goal is not to make the project sound non-technical.

The goal is to make the explanation understandable to a B.Tech student and faculty member.

---

# 2. Where to Avoid Excessive Technical Depth

Avoid introducing highly specialised remote-sensing, GIS, or deep-learning terminology unless it is genuinely required by the implementation.

Examples of terms that should NOT be introduced unnecessarily:

* U-Net
* Mask R-CNN
* semantic segmentation
* instance segmentation
* POI
* spatial autocorrelation
* radiometric normalization
* spectral unmixing
* atmospheric correction algorithms
* BRDF correction
* latent representation
* feature manifold
* geostatistical modelling
* topological analysis
* morphological operators
* multi-scale feature pyramid
* hierarchical feature aggregation
* cross-domain generalization
* transformer-based multimodal fusion
* self-supervised representation learning

These terms may be used when they are actually part of the implemented system or when discussing a specific research paper.

Do not introduce them merely to make the project sound advanced.

---

# 3. Technical Term vs Technical Explanation

Use the proper technical term, but explain it simply.

### Good

"NDVI (Normalized Difference Vegetation Index) is used to estimate vegetation cover."

### Good

"Random Forest is a machine learning model that combines many decision trees to make a prediction."

### Good

"Deep learning can be used to classify satellite images automatically."

### Avoid

"NDVI enables spectral discrimination of photosynthetically active vegetation."

### Avoid

"Random Forest performs nonlinear ensemble-based feature-space partitioning."

The technical term is fine.

The unnecessarily complicated explanation is not.

---

# 4. Dataset Names MUST Be Exact

**Never rename, shorten, invent, or paraphrase official dataset names when identifying datasets.**

The dataset's official name, provider, and product ID must be preserved.

For example:

Correct:

* NOAA/VIIRS/DNB/ANNUAL_V22
* NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG
* NOAA/DMSP-OLS/NIGHTTIME_LIGHTS
* COPERNICUS/S2_SR_HARMONIZED
* COPERNICUS/S2_HARMONIZED
* LANDSAT/LC08/C02/T1_L2
* LANDSAT/LC09/C02/T1_L2
* LANDSAT/LE07/C02/T1_L2
* LANDSAT/LT05/C02/T1_L2
* NASA/HLS/HLSL30/v002
* NASA/HLS/HLSS30/v002
* MODIS/061/MOD11A1
* COPERNICUS/S1_GRD
* USGS/SRTMGL1_003
* JAXA/ALOS/AW3D30/V4_1
* GOOGLE/DYNAMICWORLD/V1

Do NOT replace them with invented names such as:

"VIIRS Economic Dataset"

"Sentinel Urban Dataset"

"Landsat Urban Growth Dataset"

Those may be used as descriptions, but not as dataset names.

---

# 5. Use Official Dataset Name + Simple Description

Preferred format:

**NOAA/VIIRS/DNB/ANNUAL_V22 — VIIRS annual nighttime-light dataset**

Then explain:

"We use its nighttime radiance values as an indirect indicator of human and economic activity."

Another example:

**COPERNICUS/S2_SR_HARMONIZED — Sentinel-2 Surface Reflectance Harmonized**

"We use this dataset to detect built-up areas and vegetation at relatively high spatial resolution."

This gives the reader both:

1. The exact technical dataset.
2. A simple explanation of why we use it.

---

# 6. Never Change Dataset IDs

Dataset IDs are part of the implementation.

If the code contains:

```text
NOAA/VIIRS/DNB/ANNUAL_V22
```

do not change it to:

```text
VIIRS_ANNUAL
```

If a more readable label is needed in the UI, use:

```text
VIIRS Annual Nighttime Lights
```

while keeping the official dataset ID internally documented.

---

# 7. Dataset Provider Names Must Also Be Correct

Use the actual provider when known.

Examples:

* NOAA — VIIRS/DNB
* USGS — Landsat
* NASA — Harmonized Landsat and Sentinel
* ESA/Copernicus — Sentinel-1 and Sentinel-2
* NASA LP DAAC — MODIS
* JAXA — ALOS
* NASA/USGS — SRTM
* Google — Dynamic World
* OpenStreetMap — OSM data

Do not invent an agency or attribute a dataset to the wrong organisation.

---

# 8. Machine Learning and Deep Learning Are Allowed

Machine learning and deep learning are completely valid parts of this project.

Claude may recommend or explain:

* Random Forest
* XGBoost
* SVM
* Logistic Regression
* Linear Regression
* Decision Trees
* Neural Networks
* CNNs
* Basic deep-learning approaches

However, each model must have a clear reason for being used.

Example:

"Random Forest can be used to classify pixels into built-up and non-built-up areas."

Example:

"A CNN could be used to classify satellite image patches if a sufficiently large labelled dataset is available."

Do not recommend an advanced architecture simply because it sounds impressive.

---

# 9. Do Not Force Deep Learning

The project does not need deep learning everywhere.

If a simpler machine-learning or image-processing method is suitable, prefer it.

For example:

* NDVI/NDBI thresholding may be enough for an initial analysis.
* Random Forest may be enough for land-cover classification.
* Regression may be enough for trend analysis.
* A CNN should only be introduced when there is a meaningful image-classification problem and enough training data.

The project should be judged by the quality of the analysis, not by how many AI models are used.

---

# 10. Explain Models at B.Tech Level

### Random Forest

"Random Forest is a machine learning algorithm that uses many decision trees and combines their results."

### XGBoost

"XGBoost is a machine learning algorithm that builds decision trees sequentially, with each new tree trying to improve the previous result."

### CNN

"A CNN (Convolutional Neural Network) is a deep-learning model commonly used for image analysis. It learns visual patterns directly from images."

This level of explanation is preferred.

Do not provide graduate-level mathematical explanations unless specifically requested.

---

# 11. Keep Satellite Concepts Technical but Understandable

Technical satellite concepts are allowed.

For example:

"Sentinel-2 provides multispectral imagery with 10 m resolution for several important bands. We can use the Near Infrared and Red bands to calculate NDVI."

This is good.

Avoid:

"Sentinel-2 provides high-dimensional spectral observations suitable for advanced spectral manifold analysis."

---

# 12. Avoid Unnecessary Acronyms

Technical acronyms that are standard for this project are allowed, but the first occurrence should include the full name.

Examples:

"NDVI (Normalized Difference Vegetation Index)"

"NDBI (Normalized Difference Built-up Index)"

"LST (Land Surface Temperature)"

"SAR (Synthetic Aperture Radar)"

"VIIRS (Visible Infrared Imaging Radiometer Suite)"

After the first explanation, the abbreviation can be used normally.

---

# 13. Do Not Simplify Official Scientific Terms Incorrectly

Do not replace scientifically meaningful terms with inaccurate simplified wording.

For example:

Correct:

"Land Surface Temperature (LST)"

Not:

"Ground heat"

Correct:

"Nighttime light radiance"

Not:

"City brightness"

Correct:

"Surface reflectance"

Not:

"Satellite colour"

Correct:

"Built-up area"

Not:

"Building pixels"

Simple explanation is good.

Changing the actual scientific meaning is not.

---

# 14. Project Explanations Should Follow This Pattern

For any technical component:

### What it is

Use the proper technical name.

### Why we use it

Explain the purpose in simple language.

### How it works in our project

Explain what happens to the Varanasi data.

### Result

Explain what the system produces.

Example:

### VIIRS Nighttime Lights

**What it is:**
`NOAA/VIIRS/DNB/ANNUAL_V22` is an annual nighttime-light dataset.

**Why we use it:**
Nighttime brightness can be used as an indirect indicator of human and economic activity.

**How we use it:**
We compare the change in nighttime radiance across different parts of Varanasi and compare it with changes in built-up area.

**Result:**
The system identifies areas where physical urban growth and nighttime activity are both increasing.

---

# 15. Research Papers

Use technical terminology from papers only when it is necessary to explain the paper.

Do not reproduce complicated research-paper wording.

Example:

Paper wording may say:

"multi-temporal semantic segmentation of impervious surfaces"

Explain it as:

"The researchers classified satellite images from different years to identify built-up areas and study how they changed."

Keep the proper technical concept, but explain it in normal language.

---

# 16. Never Confuse Dataset Names With Derived Features

For example:

**Dataset:**

`COPERNICUS/S2_SR_HARMONIZED`

**Derived feature:**

NDVI

**Derived output:**

Vegetation map

Similarly:

**Dataset:**

`NOAA/VIIRS/DNB/ANNUAL_V22`

**Raw variable:**

Nighttime radiance

**Derived output:**

Nighttime activity map

Claude must maintain this distinction.

---

# 17. Never Invent Dataset Capabilities

Before claiming a dataset provides something, verify the actual bands, variables, spatial resolution, temporal coverage, and product type.

For example:

Do not claim that a dataset directly contains:

* GDP
* income
* business activity
* individual buildings
* exact population
* illegal construction
* future urban growth

unless the dataset genuinely provides that information.

If something is derived or inferred, explicitly call it a:

* proxy
* derived metric
* estimate
* prediction
* inference

---

# 18. Preferred Tone

The tone should sound like:

"A technically capable final-year B.Tech student who understands the project and can explain it clearly."

Not:

"A remote-sensing researcher writing a journal paper."

Not:

"A marketing document."

Not:

"A beginner who does not understand the technical concepts."

The project should sound technically serious while remaining understandable.

---

# 19. One-Sentence Rule

Before finalising a technical sentence, ask:

**"Can this be said in simpler English without losing the technical meaning?"**

If yes, simplify it.

Do not simplify the dataset name, technical method name, metric name, or scientific quantity itself.

Simplify the surrounding explanation.

---

# 20. Final Rule

## Technical terms are allowed.

## Official dataset names are mandatory.

## Unnecessary jargon is not.

The target style is:

**Proper technical terminology + simple explanation + accurate claims.**

Never sacrifice technical accuracy just to make the writing simpler.
