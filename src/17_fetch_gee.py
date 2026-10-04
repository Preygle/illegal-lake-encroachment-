"""
17_fetch_gee.py
===============
The Earth Engine layers: a per-year building count inside each fence, and two
independent cross-checks on the segmentation.

  1. GOOGLE/Research/open-buildings-temporal/v1 - Google Open Buildings Temporal
     Buildings inside the footprint, the 30 m ring and the 75 m ring, for every
     year 2016-2023. Computed server-side (zonal statistics), so nothing large
     is downloaded. Starts with the pre-flight check the design requires.

  2. GOOGLE/DYNAMICWORLD/V1 - Google Dynamic World
     The Dynamic World map made from the same Sentinel-2 scene, for each of the
     seven scene dates, compared pixel for pixel with the segmentation.

  3. JRC/GSW1_4/YearlyHistory and JRC/GSW1_4/GlobalSurfaceWater - JRC Global
     Surface Water. Yearly water class for 2019-2021 (the dataset ends in 2021)
     and long-term recurrence, compared with the segmentation's open water.

None of these is ground truth. The comparisons are reported as agreement, never
as accuracy.

    python src/17_fetch_gee.py
    python src/17_fetch_gee.py --only buildings

Writes outputs/dl/gee/{open_buildings_intrusion.csv, preflight.csv,
crosscheck_dynamic_world.csv, crosscheck_gsw.csv, *.tif}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dl import common as C
from dl import gee as G

OUT = C.DL / "gee"

# Open Buildings reports a per-pixel presence score, not a yes/no building
# mask. 0.5 is used as the presence threshold for "built area"; the building
# COUNT comes from the fractional-count band and needs no threshold.
PRESENCE_THRESHOLD = 0.5
PREFLIGHT_RING = (30, 250)


# ==========================================================================
# 1. Open Buildings Temporal
# ==========================================================================
def zones(poly) -> dict:
    zs = {"footprint": poly}
    for m in C.FENCES_M:
        zs[f"ring_{m}m"] = poly.buffer(m).difference(poly)
    zs["preflight"] = (poly.buffer(PREFLIGHT_RING[1])
                       .difference(poly.buffer(PREFLIGHT_RING[0])))
    return zs


def building_stats(img, geom, scale):
    ee = G.ee_init()
    count = img.select("building_fractional_count")
    built = img.select("building_presence").gte(PRESENCE_THRESHOLD)
    px_area = ee.Image.pixelArea()
    stack = ee.Image.cat([count.rename("count"),
                          built.multiply(px_area).rename("built_m2")])
    return stack.reduceRegion(reducer=ee.Reducer.sum(), geometry=geom,
                              scale=scale, maxPixels=1e9)


def open_buildings():
    print("[1/3] GOOGLE/Research/open-buildings-temporal/v1")
    years = G.open_buildings_years()
    scale = G.open_buildings_scale()
    print(f"  years available: {sorted(years)}   native scale {scale:.2f} m")

    geoms = C.lake_geoms()
    rows = []
    for lake in C.lake_order():
        for zone, g in zones(geoms[lake]).items():
            eg = G.ee_geom(g)
            for year, img in sorted(years.items()):
                st = building_stats(img, eg, scale).getInfo()
                rows.append({"lake": lake, "zone": zone, "year": year,
                             "buildings": round(float(st.get("count") or 0), 1),
                             "built_ha": round(float(st.get("built_m2") or 0) / 1e4, 3),
                             "zone_ha": round(g.area / 1e4, 2)})
        print(f"  {lake:16s} done")
    df = pd.DataFrame(rows)

    # --- pre-flight: are there detections near water at all? ----------------
    # A secondary source claimed the Open Buildings pipeline suppresses
    # detections within 250 m of water. The primary sources checked during the
    # research say nothing of the kind, but if it were true every fence would
    # read zero and the layer would look healthy while being useless. So: in
    # the 30-250 m ring, the OpenStreetMap snapshot shows plenty of buildings
    # around every lake; Open Buildings must see some too.
    pf = df[df.zone == "preflight"].groupby("lake").buildings.max()
    pre = pd.DataFrame({"lake": pf.index, "max_buildings_30_250m": pf.values})
    pre["passes"] = pre.max_buildings_30_250m > 1
    pre.to_csv(OUT / "preflight.csv", index=False)
    print("\n  pre-flight, buildings detected 30-250 m from the shoreline:")
    for r in pre.itertuples():
        print(f"    {r.lake:16s} {r.max_buildings_30_250m:8.1f}   "
              f"{'OK' if r.passes else 'FAIL - near-water suppression?'}")
    if not pre.passes.all():
        print("  PRE-FLIGHT FAILED for at least one lake - do not use its fence "
              "counts until this is explained.")

    wide = df[df.zone != "preflight"].pivot_table(
        index=["lake", "year"], columns="zone", values=["buildings", "built_ha"])
    wide.columns = [f"ob_{v}_{z}" for v, z in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(OUT / "open_buildings_intrusion.csv", index=False)
    df.to_csv(OUT / "open_buildings_long.csv", index=False)

    print("\n  buildings inside the 30 m / 75 m fence, by year")
    piv30 = wide.pivot(index="lake", columns="year", values="ob_buildings_ring_30m")
    piv75 = wide.pivot(index="lake", columns="year", values="ob_buildings_ring_75m")
    print("  30 m:\n" + piv30.round(1).to_string())
    print("  75 m:\n" + piv75.round(1).to_string())
    return wide, pre


# ==========================================================================
# 2. Dynamic World cross-check
# ==========================================================================
def dynamic_world(rasters):
    print("\n[2/3] GOOGLE/DYNAMICWORLD/V1 on the seven scene dates")
    scenes = pd.read_csv(C.PROC / "s2_scenes.csv")
    rows = []
    for sc in scenes.itertuples():
        year, date, tile = int(sc.year), str(sc.datetime)[:10], str(sc.tile)
        img, idx = G.dynamic_world_for(date, tile)
        if img is None:
            print(f"  {year} {date}: no Dynamic World image for {tile} "
                  f"(found {idx})")
            continue
        dw = G.download(img.select("label").toByte(),
                        OUT / f"dynamic_world_{year}.tif")[0]
        pred_p = C.DL / "pred" / f"landcover_{year}.npy"
        pred = np.load(pred_p) if pred_p.exists() else None

        mapped = np.full(dw.shape, C.IGNORE, "uint8")
        for k, v in G.DW_TO_PROJECT.items():
            mapped[dw == k] = v

        for lake, r in rasters.items():
            inside = r["inside"]
            n = int(inside.sum())
            rec = {"lake": lake, "year": year, "scene": idx}
            for k, name in enumerate(G.DW_CLASSES):
                rec[f"dw_{name}_frac"] = round(float((inside & (dw == k)).sum()) / n, 4)
            if pred is not None:
                ok = inside & (pred != C.IGNORE) & (mapped != C.IGNORE)
                rec["agreement"] = (round(float((pred[ok] == mapped[ok]).mean()), 4)
                                    if ok.any() else np.nan)
                for cls, name in enumerate(C.CLASS_NAMES):
                    rec[f"ours_{name}_frac"] = round(
                        float((inside & (pred == cls)).sum()) / n, 4)
                # Of what we call vegetation inside the lake, how much does
                # Dynamic World call flooded vegetation? This is the hyacinth
                # question asked of an independent product.
                ours_veg = inside & (pred == C.VEG)
                rec["dw_flooded_of_our_veg"] = (
                    round(float((ours_veg & (dw == 3)).sum()) / ours_veg.sum(), 4)
                    if ours_veg.any() else np.nan)
            rows.append(rec)
        print(f"  {year} {date}  {idx}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "crosscheck_dynamic_world.csv", index=False)
    if "agreement" in df:
        print("\n  agreement with Dynamic World inside each footprint (all 7 years)")
        s = df.groupby("lake").agg(agreement=("agreement", "mean"),
                                   dw_flooded=("dw_flooded_vegetation_frac", "mean"),
                                   flooded_of_our_veg=("dw_flooded_of_our_veg", "mean"))
        print(s.round(3).reindex(C.lake_order()).to_string())
    return df


# ==========================================================================
# 3. JRC Global Surface Water cross-check
# ==========================================================================
def global_surface_water(rasters):
    print("\n[3/3] JRC/GSW1_4 (Landsat, ends 2021)")
    ee = G.ee_init()
    gsw = ee.Image(G.GSW)
    rec_img = G.download(gsw.select("recurrence").unmask(0).toByte(),
                         OUT / "gsw_recurrence.tif")[0]
    occ_img = G.download(gsw.select("occurrence").unmask(0).toByte(),
                         OUT / "gsw_occurrence.tif")[0]
    yearly = ee.ImageCollection(G.GSW_YEARLY)

    rows = []
    for year in (2019, 2020, 2021):
        img = yearly.filter(ee.Filter.eq("year", year)).first()
        wc = G.download(img.select("waterClass").unmask(0).toByte(),
                        OUT / f"gsw_yearly_{year}.tif")[0]
        pred_p = C.DL / "pred" / f"landcover_{year}.npy"
        pred = np.load(pred_p) if pred_p.exists() else None
        for lake, r in rasters.items():
            inside = r["inside"]
            n = int(inside.sum())
            valid = inside & (wc > 0)
            rec = {"lake": lake, "year": year,
                   "gsw_valid_frac": round(float(valid.sum()) / n, 4),
                   # 2 = seasonal water, 3 = permanent water, 1 = not water
                   "gsw_permanent_frac": round(float((inside & (wc == 3)).sum()) / n, 4),
                   "gsw_any_water_frac": round(float((inside & (wc >= 2)).sum()) / n, 4),
                   "gsw_mean_occurrence": round(float(occ_img[inside].mean()), 1),
                   "gsw_mean_recurrence": round(float(rec_img[inside].mean()), 1)}
            if pred is not None:
                rec["ours_water_frac"] = round(
                    float((inside & (pred == C.WATER)).sum()) / n, 4)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "crosscheck_gsw.csv", index=False)
    print("  GSW 'any water in the year' vs our dry-season open water")
    cols = ["year", "gsw_any_water_frac", "gsw_permanent_frac"] + (
        ["ours_water_frac"] if "ours_water_frac" in df else [])
    for lake in C.lake_order():
        s = df[df.lake == lake][cols]
        print(f"  {lake}\n" + s.to_string(index=False))
    print("  GSW counts water seen at ANY time in the year; ours is one dry-"
          "season date,\n  so GSW should read higher. A GSW reading LOWER than "
          "ours would be the anomaly.")
    return df


def main(only):
    OUT.mkdir(parents=True, exist_ok=True)
    G.ee_init()
    rasters = C.lake_rasters()
    if only in (None, "buildings"):
        open_buildings()
    if only in (None, "dynamic_world"):
        dynamic_world(rasters)
    if only in (None, "gsw"):
        global_surface_water(rasters)
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["buildings", "dynamic_world", "gsw"])
    main(ap.parse_args().only)
