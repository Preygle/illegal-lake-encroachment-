"""
20_any_lake.py
==============
Run the whole any-lake pipeline from the command line.

    python src/20_any_lake.py "Hussain Sagar"            # first match
    python src/20_any_lake.py "Ambattur Lake" --pick 2   # second match
    python src/20_any_lake.py "17.4239, 78.4738"         # the lake at a point
    python src/20_any_lake.py "Upper Lake" --list        # just show the matches

Needs the model library first:  python src/19_anylake_library.py
Results go to outputs/anylake/<slug>/ and show up on the dashboard's
"Any lake in India" page.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anylake import pipeline, search  # noqa: E402


def main(args):
    hits = search.search(args.query)
    if not hits:
        print("No lake found in India for that name. Try adding the city or state, "
              "or give the coordinates as 'lat, lon'.")
        return 1
    for i, h in enumerate(hits, 1):
        print(f"  {i}. {h['name']} - {h['district']}, {h['state']}  "
              f"({h['lat']:.4f}, {h['lon']:.4f})  {h['area_ha']:,.1f} ha")
    if args.list:
        return 0
    if not pipeline.ready():
        print("\nThe model library is missing. Run  python src/19_anylake_library.py  first.")
        return 1
    rec = hits[args.pick - 1]
    print(f"\nAnalysing {rec['name']} ...")
    s = pipeline.run(rec, log=lambda *a: print(*a, flush=True))
    print("\n" + json.dumps({k: s[k] for k in ("unet_scores", "growth_ha", "future_zones", "forecast")},
                            indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--pick", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    sys.exit(main(ap.parse_args()))
