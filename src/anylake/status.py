"""
anylake/status.py
=================
Live progress of a pipeline run, written to outputs/anylake/<slug>/status.json.

The dashboard starts the pipeline as a separate process (anylake/run.py) and
polls this file once a second, so a run keeps going if the browser tab is
closed or refreshed, and several people can watch the same run.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import OUT

LOG_TAIL = 80


def path_for(slug: str) -> Path:
    return OUT / slug / "status.json"


def read_status(slug: str) -> dict | None:
    p = path_for(slug)
    for _ in range(3):
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except (json.JSONDecodeError, OSError):
            time.sleep(0.05)                      # caught mid-write; try again
    return None


class Status:
    def __init__(self, slug: str, name: str, steps: dict[str, str]):
        self.path = path_for(slug)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        self.s = {"slug": slug, "name": name, "state": "running", "pid": os.getpid(),
                  "started": now, "updated": now, "finished": None, "error": None,
                  "stages": [{"key": k, "label": v, "state": "pending", "started": None,
                              "ended": None, "seconds": None, "detail": ""} for k, v in steps.items()],
                  "log_tail": []}
        self._write()

    def _stage(self, key):
        return next(x for x in self.s["stages"] if x["key"] == key)

    def _write(self):
        self.s["updated"] = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.s), encoding="utf-8")
        for _ in range(20):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:               # Windows: a reader has the file open
                time.sleep(0.05)

    def start(self, key):
        x = self._stage(key)
        x.update(state="running", started=time.time(), detail="")
        self._write()

    def done(self, key, seconds):
        x = self._stage(key)
        x.update(state="done", ended=time.time(), seconds=seconds)
        self._write()

    def log(self, *parts):
        msg = " ".join(str(p) for p in parts).strip()
        if not msg:
            return
        self.s["log_tail"] = (self.s["log_tail"] + [time.strftime("%H:%M:%S ") + msg])[-LOG_TAIL:]
        cur = next((x for x in self.s["stages"] if x["state"] == "running"), None)
        if cur:
            cur["detail"] = msg[:140]
        self._write()

    def finish(self):
        self.s.update(state="done", finished=time.time())
        self._write()

    def fail(self, err: str):
        for x in self.s["stages"]:
            if x["state"] == "running":
                x["state"] = "failed"
        self.s.update(state="failed", finished=time.time(), error=err)
        self._write()
