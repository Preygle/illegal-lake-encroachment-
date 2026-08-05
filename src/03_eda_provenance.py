"""
03_eda_provenance.py
====================
"Where and what": the dataset landscape behind the 12 candidate projects.

F01  project x dataset usage matrix        (which project draws on which source)
F02  datasets per project, split by modality
F03  access tier - how obtainable each source actually is
F04  spatial resolution vs temporal span   (the gridded-data design space)
F05  study areas on a NASA MODIS basemap   (the literal "where")
F06  live endpoint reachability
F07  who publishes the data (provider origin)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import viz_style as vs

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

ds = pd.read_csv(PROC / "dataset_inventory.csv")
edges = pd.read_csv(PROC / "project_dataset_edges.csv")
proj = pd.read_csv(PROC / "projects.csv")
areas = pd.read_csv(PROC / "study_areas.csv")

SRC_ALL = "Source: dataset inventory compiled from deep-research-report.md (44 sources, 12 projects)"


# --------------------------------------------------------------------------
def order_axes():
    """Row order = modality block then usage; column order = feasibility."""
    use = edges.groupby("dataset_id").size().rename("n_proj")
    d = ds.merge(use, left_on="dataset_id", right_index=True, how="left").fillna({"n_proj": 0})
    d["mod_rank"] = (d.modality == "Non-satellite").astype(int)
    d = d.sort_values(["mod_rank", "n_proj", "dataset"], ascending=[True, False, True])
    p = proj.sort_values(["feasible_score", "uniqueness"], ascending=[False, False])
    return d, p


# ============================== F01 =======================================
def f01_matrix(t: vs.Theme):
    d, p = order_axes()
    rows = d.dataset_id.tolist()
    cols = p.project_key.tolist()
    ri = {k: i for i, k in enumerate(rows)}
    ci = {k: i for i, k in enumerate(cols)}

    fig = plt.figure(figsize=(13.4, 14.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.17], height_ratios=[0.075, 1],
                          wspace=0.03, hspace=0.012)
    ax = fig.add_subplot(gs[1, 0])
    axr = fig.add_subplot(gs[1, 1], sharey=ax)     # row totals
    axt = fig.add_subplot(gs[0, 0], sharex=ax)     # column totals

    # cells - surface gap between fills, colour carries modality
    gx, gy = 0.20, 0.13
    for e in edges.itertuples():
        if e.dataset_id not in ri or e.project_key not in ci:
            continue
        x, y = ci[e.project_key], ri[e.dataset_id]
        ax.add_patch(Rectangle((x - 0.5 + gx, y - 0.5 + gy),
                               1 - 2 * gx, 1 - 2 * gy,
                               facecolor=t.modality(e.modality),
                               edgecolor="none", zorder=3))

    # faint separator between the satellite block and the non-satellite block
    split = int((d.modality == "Satellite").sum())
    ax.axhline(split - 0.5, color=t.baseline, lw=1.2, zorder=4)

    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(p.project.tolist(), rotation=42, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(d.dataset.tolist(), fontsize=8.6)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    # row totals
    n_row = d.set_index("dataset_id").index.map(
        edges.groupby("dataset_id").size()).fillna(0).astype(int)
    axr.barh(range(len(rows)), n_row,
             color=[t.modality(m) for m in d.modality], height=0.62)
    for i, v in enumerate(n_row):
        if v:
            axr.text(v + 0.18, i, str(v), va="center", fontsize=8, color=t.ink2)
    axr.set_xlim(0, max(n_row) * 1.32)
    axr.grid(False)
    axr.set_xticks([])
    plt.setp(axr.get_yticklabels(), visible=False)
    axr.tick_params(length=0)
    for s in axr.spines.values():
        s.set_visible(False)
    axr.set_xlabel("projects\nusing", fontsize=8.5, color=t.muted, labelpad=6)

    # modality block labels, parked in the right margin clear of every mark
    xlab = max(n_row) * 1.29
    axr.text(xlab, (split - 1) / 2, "SATELLITE", rotation=90, ha="center",
             va="center", fontsize=8.6, color=t.modality("Satellite"),
             fontweight="700")
    axr.text(xlab, split + (len(rows) - split) / 2, "NON-SATELLITE", rotation=90,
             ha="center", va="center", fontsize=8.6,
             color=t.modality("Non-satellite"), fontweight="700")

    # column totals, split by modality (stacked)
    cnt = edges.groupby(["project_key", "modality"]).size().unstack(fill_value=0)
    cnt = cnt.reindex(cols).fillna(0)
    sat = cnt.get("Satellite", pd.Series(0, index=cols))
    non = cnt.get("Non-satellite", pd.Series(0, index=cols))
    axt.bar(range(len(cols)), sat, color=t.modality("Satellite"), width=0.62)
    axt.bar(range(len(cols)), non, bottom=sat, color=t.modality("Non-satellite"),
            width=0.62)
    for i, (a, b) in enumerate(zip(sat, non)):
        axt.text(i, a + b + 0.4, str(int(a + b)), ha="center", fontsize=8, color=t.ink2)
    axt.set_ylim(0, (sat + non).max() * 1.30)
    axt.grid(False)
    axt.set_yticks([])
    plt.setp(axt.get_xticklabels(), visible=False)
    axt.tick_params(length=0)
    for s in axt.spines.values():
        s.set_visible(False)
    axt.set_ylabel("sources\nper project", fontsize=8.5, color=t.muted,
                   rotation=0, ha="right", va="center", labelpad=12)

    from matplotlib.lines import Line2D
    axt.legend(handles=[Line2D([], [], marker="s", ls="", ms=9,
                               color=t.modality(m), label=m)
                        for m in ("Satellite", "Non-satellite")],
               loc="lower left", bbox_to_anchor=(0, 1.06), ncol=2,
               handletextpad=0.5)

    fig.suptitle("Which project draws on which data source",
                 x=0.005, y=0.995, ha="left", fontsize=15.5,
                 fontweight="600", color=t.ink)
    fig.text(0.005, 0.9805,
             "44 catalogued sources × 12 candidate projects — 109 usage links. "
             "Columns ordered by one-semester feasibility, rows by how many projects need them.",
             ha="left", va="top", fontsize=9.5, color=t.muted)
    vs.source(fig, SRC_ALL, t)
    return vs.save(fig, "F01_project_dataset_matrix", FIG, t)


# ============================== F02 =======================================
def f02_per_project(t: vs.Theme):
    cnt = (edges.groupby(["project", "modality"]).size().unstack(fill_value=0))
    for c in ("Satellite", "Non-satellite"):
        if c not in cnt:
            cnt[c] = 0
    cnt["total"] = cnt.sum(axis=1)
    cnt = cnt.sort_values("total")

    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    y = np.arange(len(cnt))
    ax.barh(y, cnt["Satellite"], color=t.modality("Satellite"),
            height=0.62, label="Satellite", zorder=2)
    ax.barh(y, cnt["Non-satellite"], left=cnt["Satellite"] + 0.06,
            color=t.modality("Non-satellite"), height=0.62,
            label="Non-satellite", zorder=2)

    for i, (sat, non, tot) in enumerate(zip(cnt["Satellite"], cnt["Non-satellite"],
                                            cnt["total"])):
        if sat:
            ax.text(sat / 2, i, str(int(sat)), ha="center",
                    va="center", fontsize=9, color="white", fontweight="600")
        if non:
            ax.text(sat + non / 2, i, str(int(non)), ha="center",
                    va="center", fontsize=9, color="white", fontweight="600")
        ax.text(tot + 0.25, i, f"{int(tot)}", va="center",
                fontsize=9.5, color=t.ink2, fontweight="600")

    ax.set_yticks(y)
    ax.set_yticklabels(cnt.index, fontsize=10)
    ax.set_xlim(0, cnt["total"].max() * 1.13)
    ax.set_xlabel("Number of distinct data sources required")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=1)
    vs.title(ax, "Data-source load per project",
             "Every project is a multi-source integration - none is a single-file download.", t)
    vs.source(fig, SRC_ALL, t)
    return vs.save(fig, "F02_sources_per_project", FIG, t)


# ============================== F03 =======================================
def f03_access(t: vs.Theme):
    order = ["open_api", "open_download", "scrape", "foi_rti", "restricted", "commercial"]
    pretty = {"open_api": "Open API", "open_download": "Open download",
              "scrape": "Web scrape", "foi_rti": "RTI / FOI request",
              "restricted": "Restricted", "commercial": "Commercial / paid"}
    piv = (ds.groupby(["access_tier", "modality"]).size().unstack(fill_value=0)
             .reindex(order).fillna(0))
    for c in ("Satellite", "Non-satellite"):
        if c not in piv:
            piv[c] = 0

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2 - 0.012, piv["Satellite"], w, color=t.modality("Satellite"),
           label="Satellite", zorder=2)
    ax.bar(x + w / 2 + 0.012, piv["Non-satellite"], w,
           color=t.modality("Non-satellite"), label="Non-satellite", zorder=2)
    for xi, (a, b) in enumerate(zip(piv["Satellite"], piv["Non-satellite"])):
        if a:
            ax.text(xi - w / 2 - 0.012, a + 0.16, int(a), ha="center", fontsize=9.5, color=t.ink2)
        if b:
            ax.text(xi + w / 2 + 0.012, b + 0.16, int(b), ha="center", fontsize=9.5, color=t.ink2)

    # frictionless vs friction divider
    ax.axvline(1.5, color=t.baseline, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.75, ax.get_ylim()[1] * 0.94, "no friction", ha="center",
            fontsize=9, color=t.muted, style="italic")
    ax.text(3.5, ax.get_ylim()[1] * 0.94, "needs scraping, paperwork or money",
            ha="center", fontsize=9, color=t.muted, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([pretty[o] for o in order], fontsize=9.5)
    ax.set_ylabel("Number of sources")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    vs.title(ax, "How obtainable is each source?",
             "Satellite data is almost entirely frictionless; the friction sits in Indian civic records.", t)
    vs.source(fig, SRC_ALL, t)
    return vs.save(fig, "F03_access_tier", FIG, t)


# ============================== F04 =======================================
def f04_design_space(t: vs.Theme):
    use = edges.groupby("dataset_id").size().rename("n_proj")
    d = ds.merge(use, left_on="dataset_id", right_index=True, how="left").fillna({"n_proj": 1})
    d = d.dropna(subset=["res_m"]).copy()
    d["span"] = (d.year_end - d.year_start).clip(lower=0.6)

    fig, ax = plt.subplots(figsize=(10.6, 6.8))
    for mod, sub in d.groupby("modality"):
        ax.scatter(sub.res_m, sub.span, s=38 + sub.n_proj * 30,
                   color=t.modality(mod), alpha=0.82, label=mod,
                   edgecolor=t.surface, linewidth=1.6, zorder=3)

    for r in d.itertuples():
        ax.annotate(r.dataset_id, (r.res_m, r.span), fontsize=8.2,
                    color=t.ink2, xytext=(0, 11 + r.n_proj * 0.7),
                    textcoords="offset points", ha="center")

    ax.set_xscale("log")
    ax.set_xlabel("Nominal spatial resolution (m, log scale)  → coarser")
    ax.set_ylabel("Years of archive available")
    ax.set_xlim(1.6, 1.4e5)
    ax.legend(loc="lower right", title=None)

    # readable resolution bands
    for xv, lab in [(10, "10 m\nSentinel"), (30, "30 m\nLandsat"),
                    (500, "500 m\nMODIS"), (10000, "10 km\nIMERG")]:
        ax.axvline(xv, color=t.grid, lw=0.9, zorder=0)
        ax.text(xv, ax.get_ylim()[0] + 0.6, lab, fontsize=8, color=t.muted,
                ha="center", va="bottom")

    vs.title(ax, "The gridded-data design space",
             "Bubble size = number of projects using the source. "
             f"{len(ds)-len(d)} of {len(ds)} sources are non-gridded (tabular or vector) and are not plotted.", t)
    vs.source(fig, SRC_ALL, t)
    return vs.save(fig, "F04_resolution_vs_archive", FIG, t)


# ============================== F05 =======================================
def f05_study_areas(t: vs.Theme):
    """Study cities over a real NASA MODIS true-colour basemap."""
    import matplotlib.image as mpimg
    bg = ROOT / "data" / "raw" / "gibs" / "modis_truecolor_india.png"
    fig, ax = plt.subplots(figsize=(9.6, 9.9))
    extent = (67.0, 98.0, 6.0, 37.0)                # lon_min, lon_max, lat_min, lat_max
    if bg.exists():
        ax.imshow(mpimg.imread(bg), extent=extent, origin="upper", zorder=0)
        basemap = "NASA GIBS / MODIS Terra true colour, 2024-02-15"
    else:
        ax.set_facecolor(t.surface)
        basemap = "basemap unavailable"

    a = areas.copy()
    a["n"] = a.used_by.str.count(";") + 1
    ax.scatter(a.lon, a.lat, s=52 + a.n * 78, color=t.color(1), alpha=0.92,
               edgecolor="white", linewidth=2.0, zorder=4)
    for r in a.itertuples():
        ax.annotate(f"{r.city}  ({r.n})", (r.lon, r.lat),
                    xytext=(0, 15 + r.n * 2.2), textcoords="offset points",
                    ha="center", fontsize=10, color="white", fontweight="600",
                    path_effects=None, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.28", fc="#0b0b0bcc", ec="none"))

    # the satellite deep-dive footprint
    ax.add_patch(Rectangle((77.555, 12.905), 0.20, 0.15, fill=False,
                           edgecolor=t.color(3), lw=2.0, zorder=5))
    ax.annotate("Sentinel-2 study window\n(Bengaluru lake belt)",
                (77.655, 12.905), xytext=(24, -46), textcoords="offset points",
                fontsize=9, color="white", fontweight="600", ha="left",
                arrowprops=dict(arrowstyle="-", color=t.color(3), lw=1.6),
                bbox=dict(boxstyle="round,pad=0.3", fc=t.color(3), ec="none"), zorder=6)

    ax.set_xlim(67, 98)
    ax.set_ylim(6, 37)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(color="#ffffff", alpha=0.16, lw=0.6)
    vs.title(ax, "Where the data is collected",
             "Eight study cities; the number in brackets is how many of the 12 projects name that city.", t)
    vs.source(fig, f"Basemap: {basemap}. Cities from deep-research-report.md", t)
    return vs.save(fig, "F05_study_areas_map", FIG, t)


# ============================== F06 =======================================
def f06_health(t: vs.Theme):
    h = pd.read_csv(PROC / "endpoint_health.csv")
    h = h.merge(ds[["dataset_id", "dataset", "modality"]], on="dataset_id", how="left")
    h = h.sort_values(["reachable", "latency_s"], ascending=[False, True])

    fig, ax = plt.subplots(figsize=(10.4, 6.6))
    y = np.arange(len(h))
    colors = [t.status["good"] if r else t.status["critical"] for r in h.reachable]
    ax.barh(y, h.latency_s, color=colors, height=0.6, zorder=2)
    for i, r in enumerate(h.itertuples()):
        if r.reachable:
            ax.text(r.latency_s + 0.12, i, f"{r.latency_s:.2f}s   HTTP {int(r.http_status)}",
                    va="center", fontsize=8.8, color=t.ink2)
        else:
            ax.text(r.latency_s + 0.12, i, "unreachable (TLS / timeout)",
                    va="center", fontsize=8.8, color=t.status["critical"],
                    fontweight="600")
    ax.set_yticks(y)
    ax.set_yticklabels(h.dataset.fillna(h.dataset_id), fontsize=9)
    ax.set_xlabel("Response latency (seconds)")
    ax.set_xlim(0, max(h.latency_s.max() * 1.62, 3))
    ax.grid(axis="y", visible=False)
    n_ok = int(h.reachable.sum())
    vs.title(ax, "Live reachability check of every catalogued endpoint",
             f"{n_ok} of {len(h)} responded. Probed automatically - green = HTTP 2xx/3xx, red = no response.", t)
    vs.source(fig, "Source: automated HTTP probe, src/01_fetch_nonsatellite.py", t)
    return vs.save(fig, "F06_endpoint_health", FIG, t)


# ============================== F07 =======================================
def f07_provider(t: vs.Theme):
    use = edges.groupby("dataset_id").size().rename("n_proj")
    d = ds.merge(use, left_on="dataset_id", right_index=True, how="left").fillna({"n_proj": 0})
    g = (d.groupby(["provider_country", "modality"])
           .agg(n_ds=("dataset_id", "size")).reset_index())
    piv = g.pivot(index="provider_country", columns="modality",
                  values="n_ds").fillna(0)
    for c in ("Satellite", "Non-satellite"):
        if c not in piv:
            piv[c] = 0
    piv["tot"] = piv.sum(axis=1)
    piv = piv.sort_values("tot")

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    y = np.arange(len(piv))
    ax.barh(y, piv["Satellite"], color=t.modality("Satellite"), height=0.6,
            label="Satellite", zorder=2)
    ax.barh(y, piv["Non-satellite"], left=piv["Satellite"] + 0.05,
            color=t.modality("Non-satellite"), height=0.6,
            label="Non-satellite", zorder=2)
    for i, r in enumerate(piv.itertuples()):
        ax.text(r.tot + 0.22, i, f"{int(r.tot)}", va="center", fontsize=9.5,
                color=t.ink2, fontweight="600")
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index, fontsize=10)
    ax.set_xlabel("Number of sources published")
    ax.set_xlim(0, piv["tot"].max() * 1.14)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    vs.title(ax, "Who publishes the data",
             "Indian agencies supply most of the ground truth; Earth observation comes almost entirely from abroad.", t)
    vs.source(fig, SRC_ALL, t)
    return vs.save(fig, "F07_provider_origin", FIG, t)


# --------------------------------------------------------------------------
def export_tables():
    d, p = order_axes()
    d.to_csv(TAB / "dataset_inventory_ranked.csv", index=False)
    mat = (edges.assign(v=1)
                .pivot_table(index="dataset", columns="project", values="v",
                             aggfunc="max", fill_value=0))
    mat.to_csv(TAB / "project_dataset_matrix.csv")
    print(f"  tables -> dataset_inventory_ranked.csv, project_dataset_matrix.csv")


if __name__ == "__main__":
    export_tables()
    for mode in ("light", "dark"):
        t = vs.apply(mode)
        for fn in (f01_matrix, f02_per_project, f03_access, f04_design_space,
                   f05_study_areas, f06_health, f07_provider):
            try:
                p = fn(t)
                print(f"  [{mode}] {p.name}")
            except Exception as e:                                # noqa: BLE001
                print(f"  [{mode}] {fn.__name__} FAILED: {type(e).__name__}: {e}")
    print("Done.")
