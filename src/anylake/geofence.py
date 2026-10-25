"""
anylake/geofence.py
===================
The geo-fence for any lake: what sits inside the 30 m and 75 m buffers.

Same logic as 15_geofence.py:

  * 30 m - the Bengaluru Master Plan / KTCDA buffer; 75 m - the National Green
    Tribunal figure (O.A. 222/2014). Other states set their own buffers, so
    both are reported as bracketing reference distances, not as the law for
    every lake.
  * Buildings in the fence are counted from OpenStreetMap building centroids
    (point-in-polygon) - a single current snapshot, so one number, not a trend.
  * Built-up area in the fence per year comes from the U-Net map.
  * Growth uses the persistence rule: built-up in the later year and every year
    after it, not built-up in the earlier year.
"""

from __future__ import annotations

import numpy as np

from .lake import BUILT, CLASS_NAMES, FENCES_M, WATER, Lake


def growth_mask(maps: dict, region: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """Built-up in y1 (and every later year) that was not built-up in y0."""
    new = region & (maps[y0] != BUILT) & (maps[y1] == BUILT)
    for y in maps:
        if y > y1:
            new &= maps[y] == BUILT
    return new


def buildings_in_fences(lake: Lake) -> dict:
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    xy = lake.buildings_xy
    out = {"footprint_ha": round(lake.geom.area / 1e4, 2)}
    if not len(xy):
        out.update({f"osm_buildings_{m}m": None for m in FENCES_M}, osm_buildings_inside=None)
        return out
    tree = STRtree([Point(p) for p in xy])
    for m in FENCES_M:
        ring = lake.geom.buffer(m).difference(lake.geom)
        out[f"osm_buildings_{m}m"] = int(len(tree.query(ring, predicate="intersects")))
    out["osm_buildings_inside"] = int(len(tree.query(lake.geom, predicate="intersects")))
    return out


def per_year(lake: Lake, maps: dict):
    """Water and built-up inside the outline, and built-up inside each ring, per year."""
    import pandas as pd

    rows = []
    for y, m in sorted(maps.items()):
        rec = {"year": y}
        for k, n in enumerate(CLASS_NAMES):
            rec[f"{n}_ha_inside"] = round(float((lake.masks["inside"] & (m == k)).sum()) * lake.px_ha, 2)
        for f in FENCES_M:
            ring = lake.masks[f"ring_{f}m"]
            rec[f"built_ha_{f}m"] = round(float((ring & (m == BUILT)).sum()) * lake.px_ha, 2)
            rec[f"ring_{f}m_ha"] = round(float(ring.sum()) * lake.px_ha, 2)
        rows.append(rec)
    return pd.DataFrame(rows)


def zone_hectares(mask: np.ndarray, lake: Lake) -> dict:
    """How much of `mask` falls inside the lake, the 30 m ring, the 75 m ring, the rest."""
    m = lake.masks
    within_75 = m["inside"] | m["ring_75m"]
    return {"inside_lake": round(float((mask & m["inside"]).sum()) * lake.px_ha, 2),
            "fence_30m": round(float((mask & m["ring_30m"]).sum()) * lake.px_ha, 2),
            "fence_75m": round(float((mask & m["ring_75m"]).sum()) * lake.px_ha, 2),
            "beyond_75m": round(float((mask & m["region"] & ~within_75).sum()) * lake.px_ha, 2)}


def water_change(maps: dict, lake: Lake) -> float:
    ys = sorted(maps)
    w = [float((lake.masks["inside"] & (maps[y] == WATER)).sum()) * lake.px_ha for y in ys]
    return w[-1] - w[0]
