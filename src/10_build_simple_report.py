"""
10_build_simple_report.py
=========================
Writes reports/eda_simple.html - the student-facing version of the EDA.

Five sections, plain language, one idea per figure:
  1. Where I got my data from
  2. What exactly I took from each source
  3. Looking at the data as it is
  4. My plan for the actual implementation
  5. A small look at my final dataset
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures_simple"
REP = ROOT / "reports"
REP.mkdir(parents=True, exist_ok=True)
MAX_W = 1450


def img(name: str, caption: str) -> str:
    p = FIG / f"{name}.png"
    if not p.exists():
        return f"<p style='color:#d03b3b'>[missing figure {name}]</p>"
    from PIL import Image
    im = Image.open(p)
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
    if im.mode == "RGBA":
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=84, optimize=True)
    src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return (f'<figure><img src="{src}" alt="{html.escape(caption)}">'
            f'<figcaption>{caption}</figcaption></figure>')


def table(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(
        f"<td>{'' if pd.isna(v) else html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in df.itertuples(index=False))
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


# --------------------------------------------------------------------------
FLOW = """
<svg viewBox="0 0 940 400" class="flow" role="img"
     aria-label="Flowchart of the method from three data sources to the final model">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
            markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--line)"/>
    </marker>
  </defs>

  <!-- inputs -->
  <rect x="8"  y="18"  width="196" height="66" rx="9" class="bx in"/>
  <text x="106" y="44"  class="t1">Sentinel-2 images</text>
  <text x="106" y="64"  class="t2">7 years, 6 bands</text>

  <rect x="8"  y="160" width="196" height="66" rx="9" class="bx in"/>
  <text x="106" y="186" class="t1">OpenStreetMap</text>
  <text x="106" y="206" class="t2">lake shapes, buildings</text>

  <rect x="8"  y="302" width="196" height="66" rx="9" class="bx in"/>
  <text x="106" y="328" class="t1">NASA POWER</text>
  <text x="106" y="348" class="t2">daily rainfall</text>

  <!-- processing -->
  <rect x="266" y="18"  width="196" height="66" rx="9" class="bx pr"/>
  <text x="364" y="44"  class="t1">Calculate MNDWI</text>
  <text x="364" y="64"  class="t2">water = MNDWI &gt; 0</text>

  <rect x="266" y="160" width="196" height="66" rx="9" class="bx pr"/>
  <text x="364" y="186" class="t1">Cut with lake shape</text>
  <text x="364" y="206" class="t2">count water pixels</text>

  <rect x="266" y="302" width="196" height="66" rx="9" class="bx pr"/>
  <text x="364" y="328" class="t1">Add rain + buildings</text>
  <text x="364" y="348" class="t2">extra columns</text>

  <!-- table -->
  <rect x="524" y="160" width="176" height="66" rx="9" class="bx tb"/>
  <text x="612" y="186" class="t1">Final table</text>
  <text x="612" y="206" class="t2">6 lakes x 7 years</text>

  <!-- model -->
  <rect x="756" y="88"  width="176" height="66" rx="9" class="bx md"/>
  <text x="844" y="114" class="t1">Make label</text>
  <text x="844" y="134" class="t2">risk / no risk</text>

  <rect x="756" y="232" width="176" height="66" rx="9" class="bx md"/>
  <text x="844" y="258" class="t1">Random Forest</text>
  <text x="844" y="278" class="t2">train + test</text>

  <!-- arrows -->
  <path d="M204,51  H262"  class="ln" marker-end="url(#ar)"/>
  <path d="M204,193 H262"  class="ln" marker-end="url(#ar)"/>
  <path d="M204,335 H262"  class="ln" marker-end="url(#ar)"/>
  <path d="M364,84  V150"  class="ln" marker-end="url(#ar)"/>
  <path d="M462,193 H520"  class="ln" marker-end="url(#ar)"/>
  <path d="M364,302 V240 H520 V200" class="ln" marker-end="url(#ar)" fill="none"/>
  <path d="M700,180 H728 V121 H752" class="ln" marker-end="url(#ar)" fill="none"/>
  <path d="M700,206 H728 V265 H752" class="ln" marker-end="url(#ar)" fill="none"/>
  <path d="M844,154 V228" class="ln" marker-end="url(#ar)"/>
</svg>
"""

CSS = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
:root{
  --bg:#fbfbf9; --card:#fff; --ink:#141413; --ink2:#4a4945; --muted:#8a8880;
  --line:#d9d7cf; --border:rgba(20,20,19,.11);
  --blue:#2a78d6; --orange:#eb6834; --green:#008300; --grey:#c3c2b7;
  --in:#e8f1fd; --pr:#fdeee7; --tb:#e8f6f0; --md:#efecfa;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme=light]){
    --bg:#111110; --card:#1b1b1a; --ink:#f5f5f2; --ink2:#c3c2b7; --muted:#8a8880;
    --line:#3a3a37; --border:rgba(255,255,255,.12);
    --blue:#3987e5; --orange:#d95926; --green:#0ca30c; --grey:#5a5954;
    --in:#16283e; --pr:#3a2318; --tb:#12302a; --md:#241f3d;
  }
}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.72 system-ui,-apple-system,"Segoe UI",sans-serif}
.page{max-width:960px;margin:0 auto;padding:46px 22px 90px}
h1{font-size:1.92rem;line-height:1.22;margin:0 0 8px;font-weight:660}
.sub{font-size:1.03rem;color:var(--ink2);margin:0 0 6px}
.by{font-size:.87rem;color:var(--muted);margin:0 0 26px}
h2{font-size:1.34rem;margin:52px 0 4px;font-weight:650;
   padding-top:22px;border-top:2px solid var(--line)}
h2 .n{color:var(--blue);margin-right:9px}
h3{font-size:1.03rem;margin:30px 0 6px;font-weight:640}
p{margin:0 0 15px;color:var(--ink2)}
ul,ol{color:var(--ink2);padding-left:22px;margin:0 0 15px}
li{margin:6px 0}
b,strong{color:var(--ink)}
code{font:.87em ui-monospace,Menlo,monospace;background:var(--card);
  border:1px solid var(--border);border-radius:4px;padding:1px 6px}
figure{margin:22px 0 26px}
figure img{width:100%;height:auto;display:block;border:1px solid var(--border);
  border-radius:10px;background:#fff}
figcaption{font-size:.88rem;color:var(--muted);margin-top:9px}
.tw{overflow-x:auto;border:1px solid var(--border);border-radius:10px;
  background:var(--card);margin:18px 0 24px}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--border);
  white-space:nowrap;vertical-align:top}
th{font-weight:640;color:var(--ink);background:var(--card);position:sticky;top:0}
td{color:var(--ink2)}
tbody tr:last-child td{border-bottom:none}
.box{background:var(--card);border:1px solid var(--border);border-left:4px solid var(--blue);
  border-radius:0 10px 10px 0;padding:14px 18px;margin:20px 0}
.box.warn{border-left-color:var(--orange)}
.box p:last-child{margin-bottom:0}
.steps{counter-reset:s;list-style:none;padding:0}
.steps li{counter-increment:s;position:relative;padding:11px 0 11px 46px;
  border-bottom:1px solid var(--border)}
.steps li:last-child{border-bottom:none}
.steps li::before{content:counter(s);position:absolute;left:0;top:11px;
  width:29px;height:29px;border-radius:50%;background:var(--blue);color:#fff;
  font-size:.85rem;font-weight:700;display:grid;place-items:center}
.flow{width:100%;height:auto;margin:20px 0 8px;display:block}
.flow .bx{stroke:var(--border);stroke-width:1.2}
.flow .in{fill:var(--in)} .flow .pr{fill:var(--pr)}
.flow .tb{fill:var(--tb)} .flow .md{fill:var(--md)}
.flow .t1{font:600 14px system-ui,sans-serif;fill:var(--ink);text-anchor:middle}
.flow .t2{font:12px system-ui,sans-serif;fill:var(--muted);text-anchor:middle}
.flow .ln{stroke:var(--line);stroke-width:1.8;fill:none}
.tags{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 18px}
.tag{background:var(--card);border:1px solid var(--border);border-radius:20px;
  padding:4px 13px;font-size:.83rem;color:var(--ink2)}
footer{margin-top:52px;padding-top:20px;border-top:2px solid var(--line);
  font-size:.86rem;color:var(--muted)}
@media(max-width:620px){.page{padding:28px 15px 60px}h1{font-size:1.5rem}}
"""


def build() -> str:
    feats = pd.read_csv(PROC / "features_lake_year.csv")
    wx = pd.read_csv(PROC / "weather_daily.csv")
    area = pd.read_csv(PROC / "lake_water_area.csv")
    scenes = pd.read_csv(PROC / "s2_scenes.csv", parse_dates=["datetime"])

    # --- section 1 table: the sources
    sources = pd.DataFrame([
        ["Sentinel-2 (ESA Copernicus)", "European Space Agency", "Satellite photo",
         "Free", "AWS Open Data (STAC API)"],
        ["OpenStreetMap", "OSM community", "Map data (shapes)",
         "Free", "Overpass API"],
        ["NASA POWER", "NASA", "Weather numbers", "Free", "Web API"],
        ["NASA GIBS", "NASA", "Ready-made satellite maps", "Free", "WMS service"],
    ], columns=["Source", "Who made it", "What kind of data", "Cost", "How I got it"])

    # --- section 2 table: what exactly I took
    taken = pd.DataFrame([
        ["Sentinel-2", "7 satellite images (1 per year, 2019 to 2025)",
         "Bengaluru, tile 43PGQ", "7 images"],
        ["Sentinel-2", "Bands B02, B03, B04, B08, B11 and SCL",
         "Blue, Green, Red, NIR, SWIR, cloud mask", "6 bands each"],
        ["OpenStreetMap", "Lake boundaries", "6 lakes in Bengaluru", "6 shapes"],
        ["OpenStreetMap", "Building footprints", "Same area as the images",
         "4,20,080 buildings"],
        ["OpenStreetMap", "Roads and drains", "Same area", "51,123 roads / 1,232 drains"],
        ["NASA POWER", "Daily rainfall and temperature", "8 cities, 2018 to 2025",
         "23,376 rows"],
        ["NASA GIBS", "MODIS and VIIRS map images", "For background maps", "4 images"],
    ], columns=["From which source", "What I took", "Details", "How much"])

    bands = pd.DataFrame([
        ["B02", "Blue", "10 m", "For normal colour photo"],
        ["B03", "Green", "10 m", "Used in MNDWI formula"],
        ["B04", "Red", "10 m", "For normal colour photo and NDVI"],
        ["B08", "NIR", "10 m", "For NDVI (to find plants)"],
        ["B11", "SWIR", "20 m", "Used in MNDWI formula"],
        ["SCL", "Scene class", "20 m", "Tells which pixel is cloud"],
    ], columns=["Band", "What it is", "Resolution", "Why I need it"])

    lakes_tbl = (area.groupby("lake")
                 .agg(**{"Lake size (hectare)": ("ref_ha", "first"),
                         "Water in 2019 (ha)": ("water_ha", "first"),
                         "Water in 2025 (ha)": ("water_ha", "last")})
                 .round(1).reset_index().rename(columns={"lake": "Lake"}))

    raw_feat = feats[["lake", "year", "ref_ha", "water_ha", "veg_ha", "other_ha",
                      "rain_365d", "bldg_per_ha_250m"]].head(8).round(2)
    raw_feat.columns = ["lake", "year", "ref_ha", "water_ha", "veg_ha", "other_ha",
                        "rain_365d", "bldg_per_ha_250m"]
    raw_wx = wx[["city", "date", "PRECTOTCORR", "T2M", "T2M_MAX", "RH2M"]].head(6)

    sc = scenes[["year", "datetime", "cloud", "tile"]].copy()
    sc["datetime"] = sc.datetime.dt.strftime("%d %b %Y")
    sc["cloud"] = sc.cloud.round(4)
    sc.columns = ["Year", "Date of photo", "Cloud %", "Tile"]

    return f"""<div class="page">

<h1>Exploratory Data Analysis — Illegal Lake Encroachment Prediction</h1>
<p class="sub">Finding out which lakes in Bengaluru are losing their water area,
using free satellite images and open map data.</p>
<p class="by">Data collected on 3 August 2026. All sources are free and no login is needed.</p>

<div class="tags">
  <span class="tag">6 lakes</span><span class="tag">7 years (2019–2025)</span>
  <span class="tag">4 data sources</span><span class="tag">42 rows in final table</span>
  <span class="tag">Python + rasterio + pandas</span>
</div>

<p>Before starting the model I wanted to first check what data is actually available
and whether it is usable or not. That is what this document is. First I will show where
the data comes from, then what exactly I picked, then how the raw data looks, then my
plan for the implementation, and at the end a small look at the dataset I made.</p>

<!-- ======================= 1 ======================= -->
<h2><span class="n">1.</span>Where I got my data from</h2>
<p>I have used <b>4 sources</b>. All of them are free, no payment and no registration
needed. I did not use any Kaggle file, everything is downloaded directly from the
official servers using Python.</p>

{table(sources)}

<h3>Small explanation of each one</h3>
<ul>
<li><b>Sentinel-2</b> is a satellite of the European Space Agency. It passes over
Bengaluru every 5 days and takes a photo. The photo is not only normal colour, it also
has infrared bands which we cannot see with our eyes. Water looks very different in
those bands, so this is the main thing I am using.</li>
<li><b>OpenStreetMap</b> is like a free version of Google Maps where anyone can edit.
From here I got the boundary shape of each lake, and also all the buildings and roads
near the lakes.</li>
<li><b>NASA POWER</b> gives daily rainfall and temperature for any latitude and
longitude in the world. I used it to get rainfall for Bengaluru.</li>
<li><b>NASA GIBS</b> gives ready-made satellite map images. I used it only for
background maps, not for calculation.</li>
</ul>

{img("S01_sources", "Figure 1 — The 4 sources and how many files I downloaded from each.")}

<!-- ======================= 2 ======================= -->
<h2><span class="n">2.</span>What exactly I selected from these sources</h2>
<p>Each source has a lot of data. I did not take everything, only the part I need.
Below is the exact list.</p>

{table(taken)}

<h3>Which satellite bands I selected and why</h3>
<p>One Sentinel-2 image has 13 bands. Downloading all 13 is a waste of time and space,
so I selected only 6.</p>

{table(bands)}

<h3>Which 7 images I selected</h3>
<p>For every year I picked <b>one</b> image from the dry season (January to April) with
almost zero clouds. Reason is simple. If I take one photo in monsoon and one photo in
summer, the lake will obviously look bigger in monsoon, and then I will wrongly think
the lake has grown. So I keep the season same for all years and only then the comparison
is fair.</p>

{table(sc)}

<h3>Which 6 lakes I selected</h3>
<p>I selected 6 lakes in Bengaluru of different sizes, so that my dataset is not only
big lakes or only small lakes. Bellandur and Varthur are the famous ones which come in
news for froth and pollution.</p>

{table(lakes_tbl)}

<!-- ======================= 3 ======================= -->
<h2><span class="n">3.</span>Looking at the data as it is</h2>
<p>Here I am just showing the raw data without any processing, so that it is clear what
I am actually working with.</p>

<h3>3.1 The satellite image</h3>
{img("S03_raw_image", "Figure 2 — Raw Sentinel-2 photo of Bellandur and Varthur lake. "
                      "Left dark patch is Bellandur, right one is Varthur. White area is city.")}

<h3>3.2 Where the lakes are</h3>
{img("S02_lake_map", "Figure 3 — The 6 lakes I selected, drawn using the boundary shapes "
                     "taken from OpenStreetMap.")}

<h3>3.3 The rainfall data</h3>
<p>This is how the NASA POWER data looks when I open the CSV file. One row is one day
for one city.</p>
{table(raw_wx)}
{img("S05_rainfall", "Figure 4 — Average rainfall in Bengaluru month by month. "
                     "Orange bars are the dry months where I take my satellite images.")}

<h3>3.4 My final table</h3>
<p>After processing everything I get one table. One row is one lake in one year.
First 8 rows are shown below.</p>
{table(raw_feat)}

<!-- ======================= 4 ======================= -->
<h2><span class="n">4.</span>My plan for the actual implementation</h2>
<p>Nothing fancy. The full method is shown in the flowchart, then I have written the
same thing as steps.</p>

{FLOW}
<p class="by" style="margin-top:0">Figure 5 — My method from start to end.</p>

<h3>4.1 The algorithm, step by step</h3>
<ol class="steps">
<li><b>Get the lake boundary.</b> Download the shape of each lake from OpenStreetMap.
This shape is fixed and I use the same one for all 7 years, so I am always measuring
inside the same area.</li>
<li><b>Download the satellite image.</b> For each year, search for a dry season image
with less than 1% cloud and download only the 6 bands I need.</li>
<li><b>Convert to reflectance.</b> The downloaded values are integers, not real values.
The formula is <code>reflectance = DN × 0.0001</code>. Whether an extra −0.1 belongs in
this formula is the single most important detail in the whole project, and I explain it
in section 4.3.</li>
<li><b>Calculate MNDWI</b> for every pixel using
<code>MNDWI = (Green − SWIR) / (Green + SWIR)</code>. Water gives a positive value
because water absorbs SWIR light. Land and buildings give negative value.</li>
<li><b>Make the water mask.</b> If MNDWI &gt; 0 then that pixel is water, otherwise not.</li>
<li><b>Also calculate NDVI</b> using <code>NDVI = (NIR − Red) / (NIR + Red)</code>.
If NDVI &gt; 0.3 then that pixel is plants. I need this because these lakes are full of
water hyacinth and I should not count that as water.</li>
<li><b>Count the pixels inside the lake shape.</b> Each pixel is 20 m × 20 m = 0.04
hectare. So water area = number of water pixels × 0.04.</li>
<li><b>Add rainfall.</b> From NASA POWER, add total rainfall of the last 30, 90 and 365
days before the photo date.</li>
<li><b>Add building pressure.</b> Count how many buildings are within 100 m, 250 m and
500 m of the lake boundary.</li>
<li><b>Make the final table</b> with all these columns, one row per lake per year.</li>
<li><b>Create the label.</b> Mark a lake-year as <i>risk = 1</i> if water area dropped
more than 10% compared to its own average, AND building count near the lake is high,
AND rainfall was normal (not a drought year).</li>
<li><b>Train the model.</b> Random Forest classifier, because my features are mixed
types and Random Forest handles that well without scaling. I will also try Logistic
Regression as a simple baseline to compare.</li>
<li><b>Test it properly.</b> Split by lake, not randomly. Train on some lakes and test
on lakes the model has never seen. If I split randomly then the same lake will be in
train and test and my accuracy will look falsely high.</li>
<li><b>Check the result</b> using Precision, Recall and F1 score. Accuracy alone is not
useful here because encroachment cases are very few compared to normal cases.</li>
</ol>

<h3>4.2 How the water finding actually looks</h3>
{img("S04_mndwi", "Figure 6 — Step 4 and step 5 of my algorithm on the 2025 image. "
                  "Top is the MNDWI value, bottom is the final water mask.")}

<h3>4.3 Three problems I found and how I am handling them</h3>

<div class="box warn">
<p><b>Problem 1 — the minus 0.1 offset, applied twice.</b> Sentinel-2 data from 2022
onwards stores reflectance with a −0.1 offset, and the band metadata of every scene says
to apply it. So I applied it. But the scenes also carry a property called
<code>earthsearch:boa_offset_applied: true</code>, which means the provider had already
subtracted that offset before giving me the file. I was subtracting it a second time.</p>
<p>The effect was severe and easy to miss. Over water, the raw values are small, so
subtracting 0.1 twice made the reflectance negative. A formula like
<code>(Green − SWIR) / (Green + SWIR)</code> flips sign when both bands are negative, so
open water was reading as NDVI ≈ +0.9, which means "dense vegetation". My water test
<code>MNDWI &gt; 0</code> was finding only 4&ndash;16% of the pixels that the satellite's
own classification layer calls water. After removing the double offset it finds
82&ndash;96%. Sankey Tank, which is a small clean full tank, had been reported as about
1 hectare of water out of 12.4.</p>
<p>How I confirmed it: the raw values over water pixels are 50&ndash;1200, so applying
the offset would make the median water pixel have negative reflectance, which is
physically impossible. I also checked the metadata of the 2019, 2021, 2023 and 2025
scenes directly, and all four say the offset is already applied. Lesson: metadata can
contradict itself, and a physical sanity check on the actual numbers settles it.</p>
</div>

<div class="box warn">
<p><b>Problem 2 — the lake is not one single shape.</b> In OpenStreetMap, Bellandur lake
is stored as 4 separate parts. My first code was taking only the biggest part, which is
119 hectare, but the actual full lake is 317 hectare. So I was missing two-thirds of the
lake. Now I join all the parts together. Madiwala lake also has islands inside which I
have to remove.</p>
</div>

<div class="box warn">
<p><b>Problem 3 — less water does not mean encroachment.</b> This is the most important
one. In 2021 Bellandur showed almost no water, only 0.4 hectare. But if I look at the
photo, the lake bed is dry and empty, and the satellite's own classification also says
it is bare soil. And by 2025 the water is back to 107 hectare. Encroachment does not go
away like that. So the lake was drained for cleaning work, it was not encroached. If my
label only checks "water reduced", it will mark 2021 as encroachment which is wrong.
This is why in step 11 I am also checking buildings and rainfall, not only water.</p>
</div>

<!-- ======================= 5 ======================= -->
<h2><span class="n">5.</span>A small look at my final dataset</h2>
<p>My final dataset has <b>42 rows</b> (6 lakes × 7 years) and <b>35 columns</b>.
Below are two simple charts from it.</p>

{img("S06_water_series", "Figure 7 — Water area of each lake for every year. "
                         "This is the main column of my dataset.")}

<p>From this chart we can see the water is going up and down a lot, it is not a smooth
line. Bellandur and Varthur become almost empty in the middle years and then come back
in 2025. Sankey Tank and Ulsoor stay more or less same because they are small and well
maintained.</p>

{img("S07_inside_lake", "Figure 8 — What is inside Bellandur lake. Blue is water, "
                        "green is water hyacinth plants, grey is dry ground.")}

<p>This chart is showing why I cannot use only water area. In 2019 the lake has just 16
hectare of water but 260 hectare of plants. The lake is not empty, it is covered by water
hyacinth. If I count only water I will say the lake is finished, which is not correct.
So in my dataset I am keeping water, plants and dry ground as three separate columns.</p>

<h3>5.1 What is good and what is missing</h3>
<p><b>Good points:</b></p>
<ul>
<li>All data is free and the code can run again anytime, nothing is manual.</li>
<li>All 7 images are from the same season and same satellite tile, so comparison is fair.</li>
<li>Water, plants and dry ground are separated, not mixed together.</li>
</ul>
<p><b>Missing points (I will mention these in my report as limitations):</b></p>
<ul>
<li><b>Only 42 rows.</b> This is very less for training a model. Next step is to run the
same code on all the lakes of Bengaluru, which will give a few hundred rows.</li>
<li><b>OpenStreetMap data is only of one date.</b> So my building count is same for all
7 years and cannot show growth. For that I need old OSM data or the GHSL built-up layer.</li>
<li><b>No real ground truth.</b> Nobody has given me a list of which lake is actually
encroached. So my label is only a guess made from the data. I should check at least a
few lakes manually on Google Earth to confirm.</li>
<li><b>One image per year</b> only shows the dry season minimum, not the full year.</li>
</ul>

<footer>
<p>Made using Python (rasterio, pandas, shapely, matplotlib). Satellite data from ESA
Copernicus Sentinel-2 through AWS Open Data. Weather from NASA POWER. Map data from
OpenStreetMap contributors (ODbL licence).</p>
<p>Detailed version with all 25 figures: <code>reports/eda_report.html</code>.
Interactive version: <code>reports/dashboard.html</code>.</p>
</footer>
</div>"""


if __name__ == "__main__":
    body = build()
    out = REP / "eda_simple.html"
    out.write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>EDA — Lake Encroachment Prediction</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>",
        encoding="utf-8")
    print(f"  {out.relative_to(ROOT)}   {out.stat().st_size/1e6:.2f} MB")
