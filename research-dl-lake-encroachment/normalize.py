"""Flatten any {"categories": {...}} wrapper so every result file has the same
shape, then report field coverage against fields.yaml."""
import json, glob, sys, yaml
from pathlib import Path

CATS = ["identity", "fit_to_project", "implementation",
        "evaluation", "evidence", "risk", "verdict"]

fields = yaml.safe_load(open("fields.yaml", encoding="utf-8"))["categories"]
expected = {f["name"] for c in fields.values() for f in c["fields"]}

for p in sorted(glob.glob("results/*.json")):
    d = json.load(open(p, encoding="utf-8"))
    if "categories" in d:
        inner = d.pop("categories")
        for k in CATS:
            if k in inner:
                d[k] = inner[k]
        # keep ordering: id/group/name, categories, uncertain
        unc = d.pop("uncertain", [])
        d["uncertain"] = unc
        json.dump(d, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        flag = "flattened"
    else:
        flag = "ok"
    present = {k for c in CATS for k in d.get(c, {})}
    missing = expected - present
    extra = present - expected
    status = "PASS" if not missing else f"MISSING {sorted(missing)}"
    print(f"{Path(p).stem[:58]:60s} {flag:10s} {len(present):2d}/{len(expected)} {status}"
          + (f"  EXTRA {sorted(extra)}" if extra else ""))
