"""
anylake/search.py
=================
Find a lake anywhere in India and return its outline.

Source: OpenStreetMap, through two public services -
  * Nominatim (the OSM geocoder) for name search, restricted to India
    (`countrycodes=in`), with the outline returned as GeoJSON;
  * Overpass for a typed "lat, lon", or when Nominatim gives only a point.

Both are volunteer-run. Their usage policies ask for an identifying
User-Agent, at most one request per second, and caching - all three are done
here. Results are cached on disk, so searching the same name twice costs no
request.
"""

from __future__ import annotations

import hashlib
import json
import re
import time

import requests

from . import LAKES

UA = {"User-Agent": "PredictiveDA-LakeSearch/1.0 "
                    "(B.Tech capstone; github.com/Preygle/illegal-lake-encroachment-)"}
NOMINATIM = "https://nominatim.openstreetmap.org/search"
REVERSE = "https://nominatim.openstreetmap.org/reverse"
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]
CACHE = LAKES / "_search_cache"

# OSM tags that describe standing water. Nominatim reports them as
# (category, type), e.g. ("water", "lake") or ("natural", "water").
WATER_TYPES = {"lake", "reservoir", "water", "pond", "lagoon", "basin",
               "oxbow", "wetland", "tank", "lake;pond"}
INDIA = (6.0, 68.0, 37.6, 97.6)            # S, W, N, E - rough national box

_last_call = [0.0]


def _polite_wait():
    dt = time.time() - _last_call[0]
    if dt < 1.1:
        time.sleep(1.1 - dt)
    _last_call[0] = time.time()


def _cached(key: str):
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".json")


def slugify(name: str, osm_type: str = "", osm_id=None) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "lake"
    return f"{s}-{osm_type[:1]}{osm_id}" if osm_id is not None else s


def area_ha(geom_ll) -> float:
    """Area of a lon/lat shapely geometry in hectares (equal-area projection)."""
    from pyproj import Transformer
    from shapely.ops import transform

    c = geom_ll.centroid
    tf = Transformer.from_crs("EPSG:4326",
                              f"+proj=laea +lat_0={c.y} +lon_0={c.x}",
                              always_xy=True).transform
    return transform(tf, geom_ll).area / 1e4


# ==========================================================================
# Overpass helpers
# ==========================================================================
def overpass(query: str) -> dict:
    """Run an Overpass query, trying two mirrors with backoff. Successful answers are cached."""
    path = _cached("overpass:" + query)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    last = None
    for attempt in range(4):
        for url in OVERPASS:
            try:
                r = requests.post(url, data={"data": query}, headers=UA, timeout=180)
                if r.status_code == 200 and r.text.lstrip().startswith("{"):
                    d = r.json()
                    if "remark" not in d or d.get("elements"):     # a remark means a timeout/error
                        path.write_text(r.text, encoding="utf-8")
                        return d
                    last = d["remark"][:80]
                else:
                    last = f"HTTP {r.status_code}"
            except Exception as e:                                # noqa: BLE001
                last = type(e).__name__
        time.sleep(8 * 2 ** attempt)                              # 8, 16, 32, 64 s
    raise RuntimeError(f"Overpass is not answering ({last}). It is a shared public "
                       f"server and is sometimes at capacity - try again in a few minutes.")


def nominatim(url: str, params: dict):
    path = _cached(url + json.dumps(params, sort_keys=True))
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    _polite_wait()
    r = requests.get(url, params=params, headers=UA, timeout=60)
    r.raise_for_status()
    d = r.json()
    path.write_text(json.dumps(d), encoding="utf-8")
    return d


def element_geometry(el: dict):
    """
    Shapely polygon for an Overpass way or relation fetched with `out geom`.

    Large lakes are multipolygon relations whose outer ring comes as several
    way fragments; they are stitched with linemerge + polygonize, and islands
    (inner rings) are cut out. Same approach as 02_fetch_satellite.py.
    """
    from shapely.geometry import LineString, Polygon
    from shapely.ops import linemerge, polygonize, unary_union

    if el["type"] == "way":
        g = el.get("geometry") or []
        if len(g) < 4:
            return None
        p = Polygon([(v["lon"], v["lat"]) for v in g])
        return p if p.is_valid and p.area > 0 else p.buffer(0)

    def rings(role):
        ls = [LineString([(v["lon"], v["lat"]) for v in m["geometry"]])
              for m in el.get("members", [])
              if m.get("role") == role and len(m.get("geometry", [])) >= 2]
        if not ls:
            return []
        return [p for p in polygonize(linemerge(ls)) if p.is_valid and p.area > 0]

    outer = rings("outer")
    if not outer:
        return None
    geom = unary_union(outer)
    inner = rings("inner")
    if inner:
        geom = geom.difference(unary_union(inner))
    return None if geom.is_empty else geom


def water_at(lat: float, lon: float, radius_m: int = 300) -> list[dict]:
    """
    The water body at a point. Nominatim reverse geocoding first (fast, and it
    returns the outline); Overpass as the fallback, with a containment query
    (is_in) for points in the middle of a big lake, plus water within
    `radius_m` for points just off the shore.
    """
    from shapely.geometry import Point, shape

    pt = Point(lon, lat)
    h = nominatim(REVERSE, {"lat": lat, "lon": lon, "format": "jsonv2", "layer": "natural",
                            "zoom": 14, "polygon_geojson": 1, "addressdetails": 1})
    if isinstance(h, dict) and h.get("geojson") and (h.get("type") in WATER_TYPES
                                                     or h.get("addresstype") in WATER_TYPES):
        g = shape(h["geojson"])
        if g.geom_type in ("Polygon", "MultiPolygon") and g.buffer(0.003).contains(pt):
            a = h.get("address", {})
            return [_record(h.get("name") or "Unnamed water body", h["display_name"],
                            a.get("state_district") or a.get("county") or a.get("city") or "",
                            a.get("state", ""), h["osm_type"], h["osm_id"],
                            g if g.is_valid else g.buffer(0))]

    sel = '["natural"="water"]', '["landuse"="reservoir"]'
    parts = "".join(f"way(pivot.a){t};relation(pivot.a){t};"
                    f"way(around:{radius_m},{lat},{lon}){t};" for t in sel)
    q = f"[out:json][timeout:90];is_in({lat},{lon})->.a;({parts});out geom;"
    out = []
    for el in overpass(q).get("elements", []):
        g = element_geometry(el)
        if g is None:
            continue
        tags = el.get("tags", {})
        name = tags.get("name:en") or tags.get("name") or "Unnamed water body"
        out.append(_record(name, f"{name} (OSM {el['type']} {el['id']})", "", "",
                           el["type"], el["id"], g))
    out.sort(key=lambda r: (not r["_geom"].contains(pt), r["_geom"].distance(pt)))
    return out


def by_osm_id(osm_type: str, osm_id: int):
    el = overpass(f"[out:json][timeout:90];{osm_type}({osm_id});out geom;").get("elements", [])
    return element_geometry(el[0]) if el else None


# ==========================================================================
# Public API
# ==========================================================================
def _record(name, display, district, state, osm_type, osm_id, geom) -> dict:
    from shapely.geometry import mapping

    c = geom.representative_point()
    return {"name": name, "display": display, "district": district, "state": state,
            "lat": round(c.y, 6), "lon": round(c.x, 6),
            "osm_type": osm_type, "osm_id": int(osm_id),
            "area_ha": round(area_ha(geom), 1),
            "slug": slugify(name, osm_type, osm_id),
            "geometry": mapping(geom), "_geom": geom}


def search(query: str, limit: int = 8) -> list[dict]:
    """
    Lakes in India matching `query`, largest first.

    `query` is a name ("Hussain Sagar", "Ousteri Lake, Puducherry") or a
    coordinate pair ("17.4239, 78.4738").
    """
    from shapely.geometry import shape

    query = query.strip()
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)", query)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        s, w, n, e = INDIA
        if not (s <= lat <= n and w <= lon <= e):
            raise ValueError("That point is outside India.")
        return water_at(lat, lon)[:limit]

    params = {"q": query, "countrycodes": "in", "format": "jsonv2", "limit": 20,
              "polygon_geojson": 1, "polygon_threshold": 0.00005, "addressdetails": 1}
    hits = nominatim(NOMINATIM, params)

    out, seen = [], set()
    for h in hits:
        is_water = (h.get("category") in ("water", "natural", "landuse", "waterway")
                    and h.get("type") in WATER_TYPES) or h.get("addresstype") in WATER_TYPES
        if not is_water or (h["osm_type"], h["osm_id"]) in seen:
            continue
        seen.add((h["osm_type"], h["osm_id"]))
        geom = shape(h["geojson"]) if h.get("geojson") else None
        if geom is None or geom.geom_type not in ("Polygon", "MultiPolygon"):
            # Nominatim sometimes has only the centre; ask Overpass for the outline.
            try:
                geom = by_osm_id(h["osm_type"], h["osm_id"])
            except Exception:                                     # noqa: BLE001
                geom = None
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        a = h.get("address", {})
        district = a.get("state_district") or a.get("county") or a.get("city") or ""
        out.append(_record(h.get("name") or query, h["display_name"], district,
                           a.get("state", ""), h["osm_type"], h["osm_id"], geom))
    out.sort(key=lambda r: -r["area_ha"])
    return out[:limit]


def public(rec: dict) -> dict:
    """The record without the shapely object, for JSON."""
    return {k: v for k, v in rec.items() if not k.startswith("_")}


if __name__ == "__main__":
    import sys

    for r in search(" ".join(sys.argv[1:]) or "Hussain Sagar"):
        print(f"{r['name'][:30]:30s} {r['state'][:18]:18s} {r['lat']:9.4f} {r['lon']:9.4f}"
              f" {r['area_ha']:9.1f} ha  {r['slug']}")
