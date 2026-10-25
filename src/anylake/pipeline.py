"""
anylake/pipeline.py
===================
Every step for one lake, timed, with all results written to
outputs/anylake/<slug>/.

    from anylake import pipeline, search
    rec = search.search("Hussain Sagar")[0]
    summary = pipeline.run(rec)

Steps
-----
 1  dataset      lake outline (OpenStreetMap) + Sentinel-2 scenes, 12 bands x 7 years
 2  enhance      image enhancement, before/after pictures and quality numbers
 3  vectors      image -> vector: per-pixel feature table + spectral signatures
 4  unet_raw     U-Net on raw bands, every year, scored against this lake's seeds
 5  unet_enh     U-Net on enhanced bands, the same
 6  polygons     raster -> vector: water, built-up and growth as GeoJSON
 7  geofence     buildings and built-up inside the 30 m / 75 m buffers
 8  forecast     Random Forest projection of built-up 3 years ahead
 9  render       vector -> image: predicted polygons burned onto the image
"""

from __future__ import annotations

import json
import time

import numpy as np

from . import HORIZON, MODELS, OUT
from . import fetch as F
from . import geofence as G
from . import render as R
from . import vectorize as V
from .lake import CLASS_NAMES, Lake

STEP_NAMES = {
    "dataset": "Find the lake and download Sentinel-2 (12 bands x 7 years)",
    "enhance": "Image enhancement (before / after)",
    "unet_raw": "U-Net on raw bands (7 years)",
    "unet_enh": "U-Net on enhanced bands (7 years)",
    "vectors": "Image -> vector: pixel feature vectors",
    "polygons": "Raster -> vector: GeoJSON polygons",
    "geofence": "Geo-fence (30 m / 75 m)",
    "forecast": f"Forecast model: train, validate, predict +{HORIZON} years",
    "render": "Vector -> image: polygons onto the satellite image",
}


def ready() -> bool:
    return all((MODELS / f"unet_{v}.pt").exists() for v in ("raw", "enh"))


def experiment_summary() -> dict:
    p = OUT / "enhancement_experiment.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def library() -> list[str]:
    p = OUT / "library.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def out_dir(slug: str):
    d = OUT / slug
    (d / "vectors").mkdir(parents=True, exist_ok=True)
    return d


def load_summary(slug: str) -> dict | None:
    p = OUT / slug / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_maps(slug: str, variant: str) -> dict:
    with np.load(OUT / slug / f"maps_{variant}.npz") as z:
        return {int(k[1:]): z[k] for k in z.files}


def run(rec_or_slug, log=print, on_step=None, on_start=None) -> dict:
    """
    rec_or_slug: a search result (dict) or the slug of a lake already registered.
    on_start(key) and on_step(key, seconds) are called around every step, for a
    progress display.
    """
    from . import unet_lib as U
    from .forecast import FEATURE_LABELS, LakeFeatures, run as forecast

    timings = {}

    def begin(key):
        if on_start:
            on_start(key)
        return time.perf_counter()

    def done(key, t0):
        timings[key] = round(time.perf_counter() - t0, 2)
        log(f"  {STEP_NAMES[key]}: {timings[key]:.1f}s")
        if on_step:
            on_step(key, timings[key])

    # 1 dataset ---------------------------------------------------------------
    t0 = begin("dataset")
    if isinstance(rec_or_slug, dict):
        meta = F.register(rec_or_slug)
    else:
        meta = json.loads((F.LAKES / rec_or_slug / "meta.json").read_text(encoding="utf-8"))
    meta = F.fetch(meta, log)
    lake = Lake(meta["slug"])
    if len(lake.years) < 5:
        raise RuntimeError(f"only {len(lake.years)} usable years of imagery for this lake")
    for y in lake.years:
        lake.labels(y)
    d = out_dir(lake.slug)
    win = lake.window()
    done("dataset", t0)

    # 2 enhancement -----------------------------------------------------------
    from .enhance import before_after
    from .lake import reflectance_bands

    t0 = begin("enhance")
    y_last, y_first = lake.years[-1], lake.years[0]
    before, after, quality = before_after(reflectance_bands(lake.stack(y_last)), win)
    R.plain(before, lake, win).save(d / "enh_before.png")
    R.plain(after, lake, win).save(d / "enh_after.png")
    done("enhance", t0)

    # 4-5 U-Net, both variants --------------------------------------------------
    is_library = lake.slug in library()
    maps, scores = {}, {}
    for v in ("raw", "enh"):
        t0 = begin(f"unet_{v}")
        if is_library and all((lake.dir / f"oof_{v}_{y}.npy").exists() for y in lake.years):
            # A library lake was in the deployable model's training set, so use the
            # map from the experiment fold that held it out instead.
            maps[v] = {y: np.load(lake.dir / f"oof_{v}_{y}.npy") for y in lake.years}
            src = "held-out fold of the 6-lake experiment"
        else:
            bundle = U.load_bundle(MODELS / f"unet_{v}.pt")
            maps[v], _ = U.map_lake(bundle, lake, v)
            src = "deployable model trained on the 6 Bengaluru lakes"
        cm = np.zeros((4, 4), "int64")
        for y in lake.years:
            lab = lake.labels(y).copy()
            lab[lake.bad(lake.stack(y))] = 255
            cm += U.score(maps[v][y], lab, lake.masks["region"])
        from dl import metrics as M

        scores[v] = {k: (None if isinstance(val, float) and np.isnan(val) else val)
                     for k, val in M.metrics_from_cm(cm).items()}
        scores[v]["source"] = src
        np.savez_compressed(d / f"maps_{v}.npz", **{f"y{y}": m for y, m in maps[v].items()})
        done(f"unet_{v}", t0)

    exp = experiment_summary()
    chosen = exp.get("chosen", "raw")
    cmap = maps[chosen]

    # 3 image -> vector (after the U-Net so each row carries its class) --------
    t0 = begin("vectors")
    table = V.feature_table(lake, y_last, cmap[y_last])
    table.to_csv(d / f"pixel_vectors_{y_last}.csv.gz", index=False, compression="gzip")
    sig = V.signatures(table)
    V.signature_chart(sig, d / "spectral_signatures.png")
    done("vectors", t0)

    # 6 raster -> polygons ----------------------------------------------------
    t0 = begin("polygons")
    from .lake import BUILT, WATER

    region = lake.masks["region"]
    growth = G.growth_mask(cmap, region, y_first, y_last)
    layers = {
        f"water_{y_first}": region & (cmap[y_first] == WATER),
        f"water_{y_last}": region & (cmap[y_last] == WATER),
        f"built_{y_last}": region & (cmap[y_last] == BUILT),
        f"growth_{y_first}_{y_last}": growth,
    }
    vec_stats = {}
    for name, mask in layers.items():
        polys = V.polygons(mask, lake, min_px=1, simplify_m=0)
        fc = V.to_geojson(polys, lake, {"layer": name}, d / "vectors" / f"{name}.geojson")
        vec_stats[name] = {"polygons": len(fc["features"]),
                           "vector_ha": round(sum(p.area for p in polys) / 1e4, 2),
                           "raster_ha": round(float(mask.sum()) * lake.px_ha, 2)}
    done("polygons", t0)

    # 7 geo-fence -------------------------------------------------------------
    t0 = begin("geofence")
    fence = G.buildings_in_fences(lake)
    per_year = G.per_year(lake, cmap)
    per_year.to_csv(d / "geofence_per_year.csv", index=False)
    growth_zones = G.zone_hectares(growth, lake)
    done("geofence", t0)

    # 8 forecast --------------------------------------------------------------
    t0 = begin("forecast")
    target = LakeFeatures(lake, cmap)
    pool = []
    for s in library():
        if s == lake.slug:
            continue
        L = Lake(s)
        if all((L.dir / f"oof_{chosen}_{y}.npy").exists() for y in L.years):
            pool.append(LakeFeatures(L, {y: np.load(L.dir / f"oof_{chosen}_{y}.npy") for y in L.years}))
    fc_err = None
    try:
        fres = forecast(target, pool, log=log)
        fmask = fres["mask"] & region
        np.save(d / "forecast_prob.npy", fres["prob"])
        finfo = fres["info"]
        finfo["importance"] = [(FEATURE_LABELS.get(k, k.upper() if k in ("ndvi", "ndbi", "ndwi", "mndwi")
                                                   else k), v) for k, v in finfo["importance"]]
        future_zones = G.zone_hectares(fmask, lake)
    except RuntimeError as e:
        fc_err, finfo, fmask = str(e), {}, np.zeros(lake.shape, bool)
        future_zones = G.zone_hectares(fmask, lake)
    done("forecast", t0)

    # 9 vector -> image -------------------------------------------------------
    t0 = begin("render")
    y_future = y_last + HORIZON
    fpolys = V.polygons(fmask, lake, min_px=2)
    V.to_geojson(fpolys, lake, {"layer": f"predicted_encroachment_{y_future}",
                                "note": "model prediction, not observed construction"},
                 d / "vectors" / f"predicted_encroachment_{y_future}.geojson")
    img, rgb = R.vector_in_image(lake, y_last, fpolys, win)
    img.save(d / f"future_{y_future}.png")
    R.write_geotiff(rgb, lake, win, d / f"future_{y_future}.tif")
    R.class_map(lake, cmap[y_last], y_last, win).save(d / f"landcover_{y_last}.png")
    R.with_mask(lake, y_last, growth, win).save(d / f"growth_{y_first}_{y_last}.png")
    done("render", t0)

    # Reliability checks. The models learned from six Bengaluru lakes; elsewhere
    # the U-Net can misread built-up land (a change of region, or one unusual
    # scene). These two checks catch the cases where that breaks the results.
    warnings = []
    f1b = scores[chosen].get("f1_built")
    if f1b is not None and f1b < 0.30:
        warnings.append(f"The U-Net scores only {f1b:.2f} F1 on built-up land at this lake (against "
                        f"OpenStreetMap buildings), so it is misreading the built-up area here - most likely "
                        f"because the landscape differs from the six Bengaluru lakes it learned from. Treat "
                        f"growth and the forecast as unreliable for this lake.")
    built_px = {y: int((region & (cmap[y] == BUILT)).sum()) for y in lake.years}
    earlier = float(np.median([built_px[y] for y in lake.years[:-1]]))
    if earlier >= 50 and built_px[y_last] < 0.25 * earlier:
        warnings.append(f"Built-up mapped in {y_last} ({built_px[y_last] * lake.px_ha:.1f} ha) is far below "
                        f"the earlier years (median {earlier * lake.px_ha:.1f} ha). One unusual {y_last} scene "
                        f"is the likely cause, not demolition, so growth to {y_last} and the {y_future} "
                        f"forecast are understated here.")

    hect = {n: round(float((lake.masks["inside"] & (cmap[y_last] == k)).sum()) * lake.px_ha, 2)
            for k, n in enumerate(CLASS_NAMES)}
    hect0 = {n: round(float((lake.masks["inside"] & (cmap[y_first] == k)).sum()) * lake.px_ha, 2)
             for k, n in enumerate(CLASS_NAMES)}
    summary = {
        "lake": {k: meta.get(k) for k in ("name", "display", "district", "state", "lat", "lon",
                                          "osm_type", "osm_id", "slug")},
        "area_ha": round(lake.geom.area / 1e4, 1),
        "grid": meta["grid"], "library_lake": is_library,
        "years": lake.years, "first_year": y_first, "last_year": y_last, "future_year": y_future,
        "scenes": meta["scenes"],
        "timings": timings, "total_seconds": round(sum(timings.values()), 1),
        "enhancement_quality": quality,
        "unet_scores": scores, "chosen_variant": chosen, "experiment": exp,
        "agreement_raw_enh": round(float(np.mean([(maps["raw"][y] == maps["enh"][y])[region].mean()
                                                  for y in lake.years])), 4),
        "hectares_inside_last": hect, "hectares_inside_first": hect0,
        "growth_ha": round(float(growth.sum()) * lake.px_ha, 2), "growth_zones": growth_zones,
        "fence": fence, "vectors": vec_stats,
        "forecast": finfo, "forecast_error": fc_err, "future_zones": future_zones,
        "future_polygons": len(fpolys),
        "pixels_in_vector_table": int(len(table)),
        "warnings": warnings, "built_px_by_year": built_px,
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    log(f"  total {summary['total_seconds']:.0f}s -> {d}")
    return summary
