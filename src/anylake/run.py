"""
anylake/run.py
==============
Background entry point used by the dashboard:

    python -m anylake.run <slug>          (with src/ on PYTHONPATH)

The lake must already be registered (data/lakes/<slug>/meta.json). Progress
goes to outputs/anylake/<slug>/status.json, see anylake/status.py.
"""

from __future__ import annotations

import json
import sys
import traceback

from . import LAKES
from .pipeline import STEP_NAMES, run
from .status import Status


def main(slug: str) -> int:
    meta = json.loads((LAKES / slug / "meta.json").read_text(encoding="utf-8"))
    st = Status(slug, meta["name"], STEP_NAMES)
    st.log(f"Analysing {meta['name']} ({slug})")
    try:
        run(slug, log=st.log, on_step=st.done, on_start=st.start)
    except Exception as e:                                        # noqa: BLE001
        st.log(traceback.format_exc()[-1500:])
        st.fail(f"{type(e).__name__}: {e}")
        return 1
    st.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
