"""
15_geofence.py
==============
Layer B: turns the land-cover map into encroachment evidence.

The land-cover map is not the result. The result is what sits inside the fence.
Water loss has two innocent explanations - a lake drained for desilting, and a
weak monsoon - and only one guilty one. A structure standing inside the
statutory buffer has essentially one.

Two fences, reported as bracketing cases rather than as one "correct" number:

  * **30 m** - the Bengaluru Master Plan figure, and the Karnataka Tank
    Conservation and Development Authority figure. The February 2026 amendment
    (Karnataka Act 19 of 2026) replaced the flat 30 m with a size-tiered table
    measured from the *revenue* boundary; lakes over 100 acres stay at 30 m,
    which covers Bellandur, Varthur, Madiwala and Hebbal. Ulsoor (94.2 acres)
    and Sankey Tank (30.6 acres) fall in lower tiers.
  * **75 m** - the National Green Tribunal figure from O.A. 222/2014, whose
    general application the Supreme Court set aside in 2019.

Two intrusion measures, kept separate because they are not equally strong:

  * `built_frac_*` from the segmentation map - covers all seven years, but at
    20 m a 30 m ring is about 1.5 pixels wide, so it is coarse. And because the
    built-up training labels come from a single 2026 OpenStreetMap snapshot,
    this measures *where* buildings are, not *when* they appeared.
  * `bldg_in_*` from OpenStreetMap building centroids - a single 2026
    snapshot, so constant across years, and written that way rather than
    dressed up as a trend.
  * `ob_buildings_*` / `ob_built_ha_*` from GOOGLE/Research/open-buildings-
    temporal/v1 (run 17_fetch_gee.py first) - the only genuinely per-year
    building signal, 2016-2023. 2024 and 2025 are left empty: the dataset ends
    in 2023 and the gap is never interpolated.

    python src/15_geofence.py

Writes outputs/dl/geofence/lake_year_geofence.csv and, when the segmentation
maps exist, data/processed/features_lake_year_dl.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dl import common as C

OUT = C.DL / "geofence"

# Recorded acreage decides the KTCDA tier. Derived from the OpenStreetMap
# footprint, which is NOT the revenue boundary the 2026 Act measures from - the
# tier column below is therefore indicative and must be checked against the
# revenue record before it is quoted as a legal fact.
ACRES_PER_HA = 2.471054


def buffer_rings():
    """Boolean ring masks per lake per fence distance, plus the footprint."""
    return C.lake_rasters()


def building_points():
    """OpenStreetMap building centroids in grid coordinates."""
    from pyproj import Transformer

    _, _, crs, _ = C.grid()
    pts = np.load(C.PROC / "osm_building_pts.npy")
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tf.transform(pts[:, 0], pts[:, 1])
    return np.column_stack([x, y])


def vector_intrusion(rasters, xy) -> pd.DataFrame:
    """
    Buildings inside each fence, counted from vector points.

    Point-in-polygon on centroids, not overlap-fraction on footprints: the
    Overpass query requests `out center`, so the OpenStreetMap dump has one
    point per building and no outlines. A building straddling the fence line
    counts only if its centre is inside, which on a ring 30 m wide is not a
    rounding error - so the count is indicative. This is also a single 2026
    snapshot; the per-year counts come from Open Buildings Temporal
    (17_fetch_gee.py).
    """
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    tree = STRtree([Point(p) for p in xy])
    rows = []
    for lake, r in rasters.items():
        poly = r["geom"]
        rec = {"lake": lake}
        for m in C.FENCES_M:
            ring = poly.buffer(m).difference(poly)
            rec[f"bldg_in_{m}m"] = int(len(tree.query(ring, predicate="intersects")))
            rec[f"ring_{m}m_ha"] = round(ring.area / 1e4, 2)
        rec["bldg_inside_footprint"] = int(
            len(tree.query(poly, predicate="intersects")))
        rec["footprint_ha"] = round(poly.area / 1e4, 2)
        rec["footprint_acres"] = round(poly.area / 1e4 * ACRES_PER_HA, 1)
        rec["ktcda_tier_30m"] = bool(rec["footprint_acres"] > 100)

        # How much of the lake the analysis grid actually covers. Until the AOI
        # was widened in September 2026, Madiwala's southern lobe fell below the
        # grid and only 66% of the lake was measured. Kept as a guard so a
        # future change to the AOI cannot silently clip a lake again.
        rec["on_grid_ha"] = round(int(r["inside"].sum()) * C.pixel_area_ha(), 2)
        rec["on_grid_frac"] = (round(rec["on_grid_ha"] / rec["footprint_ha"], 3)
                               if rec["footprint_ha"] else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def composition(rasters) -> pd.DataFrame:
    """
    Per lake-year: what the segmentation map says is inside the footprint, and
    how much built-up cover sits inside each fence.
    """
    px_ha = C.pixel_area_ha()
    rows = []
    for year in C.YEARS:
        path = C.DL / "pred" / f"landcover_{year}.npy"
        if not path.exists():
            continue
        pred = np.load(path)
        for lake, r in rasters.items():
            inside = r["inside"] & (pred != C.IGNORE)
            n = int(inside.sum())
            if n == 0:
                continue
            rec = {"lake": lake, "year": year,
                   "valid_px": n,
                   "valid_frac": round(n / max(int(r["inside"].sum()), 1), 3)}
            for cls, name in enumerate(C.CLASS_NAMES):
                sel = inside & (pred == cls)
                rec[f"{name}_ha"] = round(int(sel.sum()) * px_ha, 2)
                rec[f"{name}_frac"] = round(float(sel.sum()) / n, 4)
            for m in C.FENCES_M:
                ring = r[f"ring_{m}m"] & (pred != C.IGNORE)
                k = int(ring.sum())
                rec[f"built_frac_{m}m"] = (
                    round(float((ring & (pred == C.BUILT)).sum()) / k, 4)
                    if k else np.nan)
                rec[f"ring_{m}m_px"] = k
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rasters = buffer_rings()

    print("[1/3] vector building intrusion")
    vec = vector_intrusion(rasters, building_points())
    for r in vec.itertuples():
        tier = "30 m tier" if r.ktcda_tier_30m else "sub-100-acre: check gazette"
        flag = "" if r.on_grid_frac >= 0.98 else f"  <-- only {r.on_grid_frac:.0%} on grid"
        print(f"  {r.lake:16s} {r.footprint_acres:7.1f} acres  {tier:28s} "
              f"30 m ring: {r.bldg_in_30m:5,d}   75 m: {r.bldg_in_75m:5,d}{flag}")
    vec.to_csv(OUT / "vector_intrusion.csv", index=False)

    clipped = vec[vec.on_grid_frac < 0.98]
    if not clipped.empty:
        print("\n  WARNING: the analysis grid does not cover these lakes in full.")
        for r in clipped.itertuples():
            print(f"    {r.lake}: {r.on_grid_ha:.2f} of {r.footprint_ha:.2f} ha "
                  f"({r.on_grid_frac:.0%}). Every per-pixel figure for this lake, "
                  f"here and in lake_water_area.csv, describes only that part.")
        print("    Fix by widening AOI in src/02_fetch_satellite.py and re-fetching.")

    print("\n[2/3] land-cover composition per lake-year")
    comp = composition(rasters)
    if comp.empty:
        print("  no segmentation maps yet - run src/14_train_unet.py first")
        return vec
    for lake in C.lake_order():
        s = comp[comp.lake == lake].sort_values("year")
        if s.empty:
            continue
        print(f"  {lake}")
        print("     year  water_ha  hyacinth_ha  bare_ha  built_ha   built_frac_30m")
        for r in s.itertuples():
            b30 = "   n/a" if np.isnan(r.built_frac_30m) else f"{r.built_frac_30m:6.3f}"
            print(f"     {r.year}  {r.water_ha:8.2f}  {r.vegetation_ha:11.2f}  "
                  f"{r.bare_ha:7.2f}  {r.built_ha:8.2f}   {b30}")

    print("\n[3/3] merging with rainfall and the existing table")
    merged = comp.merge(vec, on="lake", how="left")
    feat = C.PROC / "features_lake_year.csv"
    if feat.exists():
        old = pd.read_csv(feat)
        keep = [c for c in ["lake", "year", "ref_ha", "scene_date", "rain_30d",
                            "rain_90d", "rain_365d", "rain_dry_since_oct",
                            "tmax_30d", "lat", "lon", "nominal_area_ha"]
                if c in old.columns]
        merged = merged.merge(old[keep], on=["lake", "year"], how="left")

    ob = C.DL / "gee" / "open_buildings_intrusion.csv"
    if ob.exists():
        obd = pd.read_csv(ob)
        merged = merged.merge(obd, on=["lake", "year"], how="left")
        print(f"  + Open Buildings Temporal per-year counts "
              f"({obd.year.min()}-{obd.year.max()}; later years left empty)")
    else:
        print("  (no Open Buildings counts - run src/17_fetch_gee.py for the "
              "per-year building series)")

    merged = merged.sort_values(["lake", "year"])
    merged.to_csv(OUT / "lake_year_geofence.csv", index=False)
    merged.to_csv(C.PROC / "features_lake_year_dl.csv", index=False)
    print(f"  {merged.shape[0]} rows x {merged.shape[1]} cols")
    print(f"  -> {OUT / 'lake_year_geofence.csv'}")
    print(f"  -> {C.PROC / 'features_lake_year_dl.csv'}")

    (OUT / "provenance.json").write_text(json.dumps({
        "fences_m": list(C.FENCES_M),
        "built_frac_is_time_invariant": True,
        "per_year_building_source": "GOOGLE/Research/open-buildings-temporal/v1 "
                                    "(2016-2023), columns ob_*",
        "built_label_source": "OpenStreetMap snapshot, single epoch (2026)",
        "caveat": "built_frac_* and bldg_in_* do not vary with year. The built-up "
                  "training labels and the vector footprints both come from one "
                  "OpenStreetMap snapshot, so these columns say WHERE buildings "
                  "are, not WHEN they appeared. A genuine per-year building "
                  "series needs GOOGLE/Research/open-buildings-temporal/v1 "
                  "(annual 2016-2023).",
        "legal": "30 m = Bengaluru Master Plan / KTCDA; 75 m = NGT O.A. 222/2014, "
                 "general application set aside by the Supreme Court in 2019. "
                 "Karnataka Act 19 of 2026 tiers the buffer by tank size, "
                 "measured from the revenue boundary, which is not the "
                 "OpenStreetMap polygon used here.",
    }, indent=2), encoding="utf-8")
    return merged


if __name__ == "__main__":
    main()
