"""Pull the verdict fields out of every result file into one ranked view."""
import json, glob, re, textwrap

RANK_ORDER = {"adopt": 0, "adopt-if-time": 1, "consider-with-reasons": 2, "reject": 3}

def rank_key(v):
    v = (v or "").lower()
    for k in ("adopt-if-time", "consider-with-reasons", "reject", "adopt"):
        if v.startswith(k):
            return RANK_ORDER[k]
    for k, n in RANK_ORDER.items():
        if k in v:
            return n
    return 9

rows = []
for p in sorted(glob.glob("results/*.json")):
    d = json.load(open(p, encoding="utf-8"))
    v = d.get("verdict", {})
    rank = v.get("recommended_rank", "?")
    rows.append({
        "id": d.get("id", "?"),
        "name": d.get("name", p),
        "rank": rank,
        "rank_short": re.split(r"[ ,.:;–-]", (rank or "?").strip().lower())[0][:24],
        "days": d.get("implementation", {}).get("student_days", "?"),
        "n_uncertain": len(d.get("uncertain", []) or []),
    })

rows.sort(key=lambda r: (rank_key(r["rank"]), r["id"]))
print(f"{'ID':4s} {'VERDICT':24s} {'DAYS':22s} {'UNC':4s} NAME")
print("-" * 118)
for r in rows:
    days = str(r["days"])[:20].replace("\n", " ")
    print(f"{r['id']:4s} {r['rank_short']:24s} {days:22s} {r['n_uncertain']:<4d} {r['name'][:46]}")
print(f"\n{len(rows)} items summarised")
