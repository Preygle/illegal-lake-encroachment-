"""
anylake/forecast.py
===================
Predicts where built-up land may appear in the next three years.

What it predicts
----------------
For every pixel in the lake + 500 m region that is not built-up today: the
probability that it becomes built-up within HORIZON (3) years and stays that
way. This is an estimate learned from what happened around these lakes
between 2019 and 2025, not an observation and not a legal finding.

Model
-----
Random Forest - many decision trees, each trained on a random part of the
data, whose votes are averaged. It handles mixed features (reflectance,
distances, yes/no flags) without scaling, and does not need much tuning.

Each pixel is described by a feature vector (the image -> vector step):
  * the 16 input channels in year t (12 bands + NDWI, MNDWI, NDVI, NDBI)
  * change since t-1 in NDVI, NDBI and MNDWI
  * share of built-up within 100 m and 300 m, distance to nearest built-up,
    and how many earlier years the pixel itself was mapped built-up
  * signed distance to the shoreline (negative = inside the lake), and
    whether it is inside the lake / the 30 m fence / the 75 m fence
  * distance to the nearest OpenStreetMap road

Target: built-up at t+3 and in every later year we have (the persistence rule,
so a one-year misclassification of dry lake bed is not counted as growth).

Honest evaluation: train on the transitions 2020->2023 and 2021->2024, test
on 2022->2025, which the model has not seen. Then refit on all three and
project 2025 -> 2028.
"""

from __future__ import annotations

import numpy as np

from . import HORIZON
from .lake import BUILT, CHANNELS, IGNORE, Lake

TRAIN_T = [2020, 2021]
VAL_T = 2022
CAP_PER_LAKE_YEAR = 60_000

FEATURES = (CHANNELS + ["d_ndvi", "d_ndbi", "d_mndwi", "built_share_100m", "built_share_300m",
                        "dist_built_m", "built_years_before", "shore_m", "in_lake", "in_30m",
                        "in_75m", "dist_road_m"])
FEATURE_LABELS = {"built_share_300m": "Built-up share within 300 m",
                  "built_share_100m": "Built-up share within 100 m",
                  "dist_built_m": "Distance to built-up", "dist_road_m": "Distance to road",
                  "shore_m": "Distance to shoreline", "built_years_before": "Years mapped built before",
                  "d_ndbi": "Change in NDBI", "d_ndvi": "Change in NDVI", "d_mndwi": "Change in MNDWI"}


class LakeFeatures:
    """Feature planes for one lake, computed once per year and reused."""

    def __init__(self, lake: Lake, maps: dict):
        from scipy.ndimage import distance_transform_edt as edt

        self.lake, self.maps = lake, maps
        m, res = lake.masks, lake.res
        self.static = {
            "shore_m": np.where(m["inside"], -edt(m["inside"]), edt(~m["inside"])).astype("float32") * res,
            "in_lake": m["inside"].astype("float32"),
            "in_30m": m["ring_30m"].astype("float32"),
            "in_75m": m["ring_75m"].astype("float32"),
            "dist_road_m": (np.minimum(edt(~lake.road_mask) * res, 3000).astype("float32")
                            if lake.road_mask.any() else np.full(lake.shape, 3000, "float32")),
        }
        self._chan = {}

    def chan(self, y):
        if y not in self._chan:
            self._chan[y] = self.lake.channels(y, "raw")
        return self._chan[y]

    def planes(self, t: int):
        import cv2
        from scipy.ndimage import distance_transform_edt as edt

        x, bad = self.chan(t)
        xp, _ = self.chan(t - 1) if (t - 1) in self.maps else (x, bad)
        built = (self.maps[t] == BUILT)
        bf = built.astype("float32")
        p = {c: x[i] for i, c in enumerate(CHANNELS)}
        for c in ("ndvi", "ndbi", "mndwi"):
            i = CHANNELS.index(c)
            p[f"d_{c}"] = x[i] - xp[i]
        p["built_share_100m"] = cv2.blur(bf, (11, 11))
        p["built_share_300m"] = cv2.blur(bf, (31, 31))
        p["dist_built_m"] = (np.minimum(edt(~built) * self.lake.res, 2000).astype("float32")
                             if built.any() else np.full(self.lake.shape, 2000, "float32"))
        p["built_years_before"] = sum((self.maps[y] == BUILT).astype("float32")
                                      for y in self.maps if y < t)
        p.update(self.static)
        valid = (self.lake.masks["region"] & ~bad & (self.maps[t] != IGNORE) & ~built)
        return p, valid

    def target(self, t: int) -> np.ndarray:
        y1 = t + HORIZON
        tgt = self.maps[y1] == BUILT
        for y in self.maps:
            if y > y1:
                tgt &= self.maps[y] == BUILT
        return tgt

    def sample(self, t: int, cap: int | None, rng):
        p, valid = self.planes(t)
        rows, cols = np.nonzero(valid)
        if cap and len(rows) > cap:
            k = rng.choice(len(rows), cap, replace=False)
            rows, cols = rows[k], cols[k]
        X = np.column_stack([p[f][rows, cols] for f in FEATURES]).astype("float32")
        y = self.target(t)[rows, cols] if (t + HORIZON) in self.maps else None
        return X, y, (rows, cols)


def _stack(parts):
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    return X, y


def run(target: LakeFeatures, pool: list[LakeFeatures], seed: int = 42, log=print) -> dict:
    """
    Train, validate on the held-out transition, refit, and project the target
    lake HORIZON years past its last year.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    rng = np.random.default_rng(seed)
    everyone = [target] + [p for p in pool if p.lake.slug != target.lake.slug]
    have = lambda lf, t: t in lf.maps and (t + HORIZON) in lf.maps and (t - 1) in lf.maps  # noqa: E731

    train = [lf.sample(t, CAP_PER_LAKE_YEAR, rng)[:2] for lf in everyone for t in TRAIN_T if have(lf, t)]
    val = {lf.lake.slug: lf.sample(VAL_T, None, rng)[:2] for lf in everyone if have(lf, VAL_T)}
    if not train:
        raise RuntimeError("not enough years of imagery for a forecast")
    Xtr, ytr = _stack(train)
    log(f"    training pixels {len(ytr):,d}  ({ytr.mean():.2%} became built-up)")

    def model():
        return RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=20,
                                      class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=seed)

    out = {"horizon": HORIZON, "train_transitions": [f"{t}->{t + HORIZON}" for t in TRAIN_T],
           "val_transition": f"{VAL_T}->{VAL_T + HORIZON}", "n_train": int(len(ytr))}
    thr = 0.5
    if ytr.sum() >= 20 and val:
        rf = model().fit(Xtr, ytr)
        Xv, yv = _stack(list(val.values()))
        pv = rf.predict_proba(Xv)[:, 1]
        if 0 < yv.sum() < len(yv):
            prec, rec, th = precision_recall_curve(yv, pv)
            f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
            i = int(np.nanargmax(f1[:-1]))
            thr = float(th[i])
            out.update(val_auc=round(float(roc_auc_score(yv, pv)), 4),
                       val_ap=round(float(average_precision_score(yv, pv)), 4),
                       val_base_rate=round(float(yv.mean()), 4),
                       val_precision=round(float(prec[i]), 4), val_recall=round(float(rec[i]), 4),
                       val_f1=round(float(f1[i]), 4))
        if target.lake.slug in val:
            Xt, yt = val[target.lake.slug]
            pt = rf.predict_proba(Xt)[:, 1]
            if 0 < yt.sum() < len(yt):
                out["val_auc_this_lake"] = round(float(roc_auc_score(yt, pt)), 4)
                out["val_ap_this_lake"] = round(float(average_precision_score(yt, pt)), 4)
            out["val_new_built_ha_this_lake"] = round(float(yt.sum()) * target.lake.px_ha, 2)
            out["val_predicted_ha_this_lake"] = round(float((pt >= thr).sum()) * target.lake.px_ha, 2)
    out["threshold"] = round(thr, 4)

    # refit on every transition, then project from the last year
    allp = train + [v for v in val.values() if v[1] is not None]
    Xa, ya = _stack(allp)
    if ya.sum() < 20:
        raise RuntimeError("too little persistent growth in 2019-2025 to learn from")
    rf = model().fit(Xa, ya)
    t_last = max(target.maps)
    X, _, (rows, cols) = target.sample(t_last, None, rng)
    prob = np.zeros(target.lake.shape, "float32")
    prob[rows, cols] = rf.predict_proba(X)[:, 1]
    imp = sorted(zip(FEATURES, rf.feature_importances_), key=lambda kv: -kv[1])
    out.update(from_year=t_last, to_year=t_last + HORIZON, n_train_final=int(len(ya)),
               importance=[(k, round(float(v), 4)) for k, v in imp[:10]])
    return {"prob": prob, "mask": prob >= thr, "info": out}
