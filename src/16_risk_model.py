"""
16_risk_model.py
================
Layer C: the risk models, and Layer D: the desilting rule.

Why mixed-effects models and not a tree ensemble
------------------------------------------------
There are 42 rows, but they are not 42 independent observations - they are six
lakes measured seven times each. Every OpenStreetMap building and road column in
the table is constant within a lake (Bellandur is 128 / 733 / 2079 in all seven
years), which means those columns *are* a lake-identity label. A gradient
boosting model splits on them, memorises which lake it is looking at, and
returns a confident number that measures nothing.

A linear mixed-effects model gives each lake its own intercept instead. That is
graceful degradation rather than silent failure: the model reports a wide,
honest interval instead of a narrow, false one. Estimating between-lake
variance from six clusters sits well below the usual 15-50 minimum, so the
interval is wide by construction, and that is the finding.

Two models, one per question:

  * **Water model** - `water_frac ~ year + rainfall + (1 | lake)`. How much of
    each footprint is open water, and how much of the year-to-year change is
    the monsoon rather than anything happening at the lake.
  * **Building model** - `log(1 + buildings in the fence) ~ year + (1 | lake)`,
    on GOOGLE/Research/open-buildings-temporal/v1, 2016-2023. The year
    coefficient is the average yearly growth in buildings inside the fence - a
    direct encroachment-pressure measure. This is the model the design asked
    for. The design's first choice of target, the segmentation's built
    fraction, cannot be used: its training labels come from one 2026
    OpenStreetMap snapshot, so it does not change from year to year.

`--show-leakage` fits XGBoost on the same table for the one purpose it is
honestly good for here: demonstrating the memorisation, by showing a near
perfect in-sample fit collapse under leave-one-lake-out.

    python src/16_risk_model.py
    python src/16_risk_model.py --show-leakage

Writes outputs/dl/risk/{mixed_effects_*.txt, predictions_*.csv,
building_growth_by_lake.csv, desilting_rule.csv}
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dl import common as C

OUT = C.DL / "risk"

TARGET = "water_frac"
FIXED = ["year_c", "rain_365d_z"]
WATER_FORMULA = f"{TARGET} ~ " + " + ".join(FIXED)


# ==========================================================================
# Mixed-effects fitting
# ==========================================================================
def mixed_fit(train: pd.DataFrame, formula: str):
    import statsmodels.formula.api as smf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm(formula, train, groups=train["lake"]).fit(
            reml=True, method="lbfgs")


# ==========================================================================
# Conformal prediction
# ==========================================================================
def jackknife_plus_by_lake(d: pd.DataFrame, formula: str, target: str,
                           alpha: float = 0.10) -> pd.DataFrame:
    """
    Jackknife+ intervals (Barber, Candes, Ramdas & Tibshirani, 2021), with
    lakes as the unit that is left out, evaluated on lakes the intervals never
    saw.

    For each lake L in turn:
      1. L is set aside as the test lake.
      2. Among the other five lakes, each lake k is left out once. A model
         fitted on the remaining four gives
           - a leave-one-out residual R_i = |y_i - mu_-k(x_i)| for every row i
             of lake k, and
           - a prediction mu_-k(x) for every row x of the test lake L.
      3. For a test row x, with n training rows, the interval is
           lower = the floor(alpha (n+1))-th smallest of  mu_-k(i)(x) - R_i
           upper = the ceil((1-alpha)(n+1))-th smallest of mu_-k(i)(x) + R_i
    Coverage is then measured on L, which played no part in building its own
    intervals - so the reported coverage is an honest out-of-sample number.

    Split conformal would need a calibration set held out of 42 rows, which is
    unaffordable; jackknife+ reuses every row. Lakes rather than rows are left
    out because rows within one lake are not exchangeable with each other.

    Caveats that travel with every interval produced here:
      * Coverage is over the project's own derived target, not over true
        encroachment - there is no ground truth to calibrate against.
      * The jackknife+ worst-case guarantee is 1 - 2*alpha (80% for a 90%
        target), and it assumes exchangeable units. Six lakes are a thin basis
        for that, so only marginal (pooled) coverage is claimed, never per-lake.
    """
    lakes = list(dict.fromkeys(d["lake"]))
    out = []
    for test in lakes:
        train = d[d.lake != test]
        te = d[d.lake == test]
        R, MU = [], []
        for k in [lk for lk in lakes if lk != test]:
            fit = mixed_fit(train[train.lake != k], formula)
            held = train[train.lake == k]
            r = np.abs(held[target].to_numpy() - fit.predict(held).to_numpy())
            mu = fit.predict(te).to_numpy()      # fixed effects only: lake unseen
            for ri in r:
                R.append(ri)
                MU.append(mu)
        R = np.asarray(R)                        # (n,)
        MU = np.vstack(MU)                       # (n, n_test)
        n = len(R)
        lo_k = int(np.floor(alpha * (n + 1)))
        hi_k = int(np.ceil((1 - alpha) * (n + 1)))
        lower_vals = np.sort(MU - R[:, None], axis=0)
        upper_vals = np.sort(MU + R[:, None], axis=0)
        lo = lower_vals[lo_k - 1] if lo_k >= 1 else np.full(MU.shape[1], -np.inf)
        hi = upper_vals[hi_k - 1] if hi_k <= n else np.full(MU.shape[1], np.inf)
        point = mixed_fit(train, formula).predict(te).to_numpy()
        part = te[["lake", "year", target]].copy()
        part["pred"], part["lo"], part["hi"] = point, lo, hi
        out.append(part)
    res = pd.concat(out, ignore_index=True)
    res["covered"] = (res[target] >= res.lo) & (res[target] <= res.hi)
    return res


def report_intervals(res: pd.DataFrame, target: str, alpha: float, label: str):
    width = res.hi - res.lo
    print(f"  median interval width {width.median():.4f}   (target '{target}')")
    print(f"  out-of-sample marginal coverage {res.covered.mean():.1%} over "
          f"{len(res)} rows, each covered by intervals built without its lake")
    print(f"  nominal {1-alpha:.0%}; jackknife+ worst-case guarantee "
          f"{1-2*alpha:.0%}")
    per = res.groupby("lake").covered.mean()
    print("  per lake (descriptive only, not a claimable guarantee): " +
          ", ".join(f"{k.split()[0]} {v:.0%}" for k, v in per.items()))
    print(f"  These intervals cover the project's OWN {label}, not true "
          "encroachment.")


# ==========================================================================
# Data
# ==========================================================================
def load_table() -> pd.DataFrame:
    """The segmentation-derived table if it exists, else the threshold one."""
    dl = C.PROC / "features_lake_year_dl.csv"
    if dl.exists():
        df = pd.read_csv(dl)
        src = "features_lake_year_dl.csv (segmentation)"
    else:
        df = pd.read_csv(C.PROC / "features_lake_year.csv")
        src = "features_lake_year.csv (MNDWI threshold)"
    print(f"  source: {src}   {df.shape[0]} rows x {df.shape[1]} cols")

    df = df.sort_values(["lake", "year"]).reset_index(drop=True)
    df["year_c"] = df["year"] - df["year"].mean()
    if "rain_365d" in df.columns:
        r = df["rain_365d"].astype("float64")
        df["rain_365d_z"] = (r - r.mean()) / max(r.std(ddof=0), 1e-9)
    else:
        df["rain_365d_z"] = 0.0
    return df


def building_panel():
    """
    Lake-year building counts from Open Buildings Temporal, 2016-2023.

    Kept separate from the 42-row table because it has its own years: eight of
    them, of which only 2019-2023 overlap the Sentinel-2 scenes.
    """
    p = C.DL / "gee" / "open_buildings_intrusion.csv"
    if not p.exists():
        return None
    b = pd.read_csv(p).sort_values(["lake", "year"]).reset_index(drop=True)
    for m in C.FENCES_M:
        col = f"ob_buildings_ring_{m}m"
        if col in b:
            b[f"log_bldg_{m}m"] = np.log1p(b[col].clip(lower=0))
    b["year_c"] = b["year"] - b["year"].mean()
    return b


def building_growth_by_lake(b: pd.DataFrame, m: int) -> pd.DataFrame:
    """Per-lake log-linear trend: the yearly % change in buildings in the fence."""
    rows = []
    col = f"ob_buildings_ring_{m}m"
    for lake, s in b.groupby("lake"):
        x = s.year.to_numpy(float)
        y = np.log1p(s[col].clip(lower=0).to_numpy(float))
        slope = float(np.polyfit(x, y, 1)[0]) if len(s) > 2 else np.nan
        rows.append({"lake": lake, "fence_m": m,
                     "first_year": int(s.year.min()), "last_year": int(s.year.max()),
                     "buildings_first": round(float(s[col].iloc[0]), 1),
                     "buildings_last": round(float(s[col].iloc[-1]), 1),
                     "growth_pct_per_year": round((np.exp(slope) - 1) * 100, 1)})
    return pd.DataFrame(rows)


# ==========================================================================
# Layer D: the desilting rule
# ==========================================================================
def desilting_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reads consecutive years and names what the change looks like.

    This is a rule, not a model, and it is where the four-class segmentation
    earns its keep: water falling while *bare* rises is a drained bed, water
    falling while *built* rises is the encroachment signal, and water falling
    while *vegetation* rises is hyacinth. A three-class water/vegetation/other
    split cannot separate the first two, which is why the old pipeline could
    only ever report "water went down".
    """
    need = {"water_frac", "bare_frac", "built_frac", "vegetation_frac"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    rows = []
    for lake, s in df.groupby("lake"):
        s = s.sort_values("year").reset_index(drop=True)
        for i in range(1, len(s)):
            a, b = s.loc[i - 1], s.loc[i]
            d_water = b.water_frac - a.water_frac
            d_bare = b.bare_frac - a.bare_frac
            d_built = b.built_frac - a.built_frac
            d_veg = b.vegetation_frac - a.vegetation_frac
            later = s.loc[i + 1:, "water_frac"]
            recovers = bool(len(later) and later.max() >= a.water_frac * 0.8)

            # A building does not appear one year and vanish the next. A rise in
            # the built class that reverses is the segmentation confusing a dry,
            # cracked lake bed for an urban surface - spectrally they are close,
            # and the built class scores the weakest IoU of the four. Requiring
            # the rise to persist is what stops a drained bed being reported as
            # encroachment.
            built_later = s.loc[i + 1:, "built_frac"]
            built_persists = bool(len(built_later) == 0
                                  or built_later.max() >= b.built_frac * 0.8)

            if d_water >= -0.05:
                verdict = "no material water loss"
            elif d_built > 0.02 and built_persists:
                verdict = "ENCROACHMENT SIGNAL - check the building count in the fence"
            elif d_veg > 0.05:
                verdict = "hyacinth spread, not lake loss"
            elif d_bare > 0.05 and recovers:
                verdict = "drained (bed bare, water returns) - check desilting records"
            elif d_bare > 0.05:
                verdict = "bed bare, no recovery yet - ambiguous, needs documents"
            elif d_built > 0.02:
                verdict = ("transient built spike, does not persist - most likely "
                           "dry bed misclassified, not construction")
            else:
                verdict = "water loss, cause unresolved"

            rows.append({"lake": lake, "from_year": int(a.year),
                         "to_year": int(b.year),
                         "d_water_frac": round(d_water, 4),
                         "d_veg_frac": round(d_veg, 4),
                         "d_bare_frac": round(d_bare, 4),
                         "d_built_frac": round(d_built, 4),
                         "recovers_later": recovers,
                         "built_persists": built_persists, "reading": verdict})
    return pd.DataFrame(rows)


def show_leakage(df: pd.DataFrame, target: str):
    """XGBoost's one honest use here: demonstrating lake-identity memorisation."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  xgboost not installed - skipping")
        return None

    # Exclude every column computed from the same land-cover map as the target.
    # water_ha is the target multiplied by the footprint area, and the other
    # footprint fractions sum to ~1 with it - a model given those is doing
    # arithmetic, not prediction, and its score says nothing about lake
    # identity. What remains are the pressure features a risk model would
    # actually use: rainfall, buildings, roads, geometry and location.
    derived = ("water_", "vegetation_", "bare_", "built_ha", "valid_")
    num = df.select_dtypes("number")
    feats = [c for c in num.columns
             if c not in (target, "year", "built_frac")
             and not c.startswith(derived)]
    const_cols = [c for c in feats
                  if c != "year_c"
                  and df.groupby("lake")[c].nunique(dropna=False).max() == 1]
    d = df.dropna(subset=[target]).copy()
    X = d[feats].fillna(0).to_numpy()
    y = d[target].to_numpy()

    print(f"  {len(const_cols)} of {len(feats)} numeric features are constant "
          f"within a lake, i.e. they encode lake identity:")
    print("    " + ", ".join(const_cols[:12]) + (" ..." if len(const_cols) > 12 else ""))

    def model():
        return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.1,
                            random_state=42)

    m = model().fit(X, y)
    r2_in = 1 - ((y - m.predict(X)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    preds = np.empty_like(y)
    for lake in d["lake"].unique():
        tr = (d["lake"] != lake).to_numpy()
        preds[~tr] = model().fit(X[tr], y[tr]).predict(X[~tr])
    r2_out = 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    print(f"  XGBoost in-sample R2            {r2_in:+.3f}")
    print(f"  XGBoost leave-one-lake-out R2   {r2_out:+.3f}")
    print("  That gap is the memorisation. This is the figure to put in the "
          "report, not a SHAP plot.")
    return {"r2_in_sample": r2_in, "r2_leave_one_lake_out": r2_out,
            "constant_within_lake": const_cols}


# ==========================================================================
def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/5] loading the lake-year table")
    df = load_table()

    # ------------------------------------------------------------ water model
    print(f"\n[2/5] water model: {WATER_FORMULA}  + (1 | lake)")
    d = df.dropna(subset=[TARGET] + FIXED).copy()
    fit = mixed_fit(d, WATER_FORMULA)
    txt = fit.summary().as_text()
    (OUT / "mixed_effects_water.txt").write_text(txt, encoding="utf-8")
    print(txt)
    gv = float(fit.cov_re.iloc[0, 0])
    rv = float(fit.scale)
    print(f"  between-lake variance {gv:.5f}   residual variance {rv:.5f}   "
          f"ratio {gv / max(rv, 1e-12):.1f}x")
    print("  Estimated from 6 clusters, below the 15-50 usually cited as a "
          "minimum - imprecise\n  by construction, and reported as such.")

    print(f"\n  jackknife+ by lake (alpha={args.alpha})")
    res = jackknife_plus_by_lake(d, WATER_FORMULA, TARGET, args.alpha)
    res.to_csv(OUT / "predictions_water.csv", index=False)
    report_intervals(res, TARGET, args.alpha, "segmentation-derived water fraction")

    # --------------------------------------------------------- building model
    print("\n[3/5] building model on GOOGLE/Research/open-buildings-temporal/v1")
    b = building_panel()
    if b is None:
        print("  no Open Buildings counts - run src/17_fetch_gee.py first")
    else:
        growth = pd.concat([building_growth_by_lake(b, m) for m in C.FENCES_M])
        growth.to_csv(OUT / "building_growth_by_lake.csv", index=False)
        print("  buildings inside the fence, first vs last year, and yearly growth")
        print(growth.to_string(index=False))

        m = args.fence
        tgt = f"log_bldg_{m}m"
        formula = f"{tgt} ~ year_c"
        print(f"\n  mixed model: {formula}  + (1 | lake)   "
              f"({len(b)} lake-years, {b.year.min()}-{b.year.max()})")
        bfit = mixed_fit(b, formula)
        btxt = bfit.summary().as_text()
        (OUT / f"mixed_effects_buildings_{m}m.txt").write_text(btxt, encoding="utf-8")
        print(btxt)
        slope = float(bfit.fe_params["year_c"])
        ci = bfit.conf_int().loc["year_c"].to_numpy()
        print(f"  average growth of buildings inside the {m} m fence: "
              f"{(np.exp(slope) - 1) * 100:+.1f}% per year "
              f"(95% CI {(np.exp(ci[0]) - 1) * 100:+.1f}% to "
              f"{(np.exp(ci[1]) - 1) * 100:+.1f}%)")
        bres = jackknife_plus_by_lake(b, formula, tgt, args.alpha)
        bres.to_csv(OUT / f"predictions_buildings_{m}m.csv", index=False)
        print(f"\n  jackknife+ by lake (alpha={args.alpha}), on log(1 + count)")
        report_intervals(bres, tgt, args.alpha,
                         "Open Buildings count (itself a model output)")

    # ---------------------------------------------------------- desilting rule
    print("\n[4/5] desilting rule")
    rule = desilting_rule(df)
    if rule.empty:
        print("  needs the segmentation table - run src/14_train_unet.py then "
              "src/15_geofence.py")
    else:
        rule.to_csv(OUT / "desilting_rule.csv", index=False)
        for lake in C.lake_order():
            s = rule[rule.lake == lake]
            if s.empty:
                continue
            print(f"  {lake}")
            for r in s.itertuples():
                print(f"    {r.from_year}->{r.to_year}  water {r.d_water_frac:+.3f} "
                      f"veg {r.d_veg_frac:+.3f} bare {r.d_bare_frac:+.3f} "
                      f"built {r.d_built_frac:+.3f}   {r.reading}")

    if args.show_leakage:
        print("\n[5/5] the XGBoost leakage exhibit")
        show_leakage(df, TARGET)

    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--fence", type=int, default=75, choices=list(C.FENCES_M),
                    help="fence distance for the building model")
    ap.add_argument("--show-leakage", action="store_true",
                    help="fit XGBoost to demonstrate lake-identity memorisation")
    main(ap.parse_args())
