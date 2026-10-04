"""
12_make_labels.py
=================
Build the four-class seed labels, and move them to and from QGIS for hand
correction.

    python src/12_make_labels.py                  # build seeds from SCL + OSM
    python src/12_make_labels.py --export-qgis    # write GeoTIFFs to correct
    python src/12_make_labels.py --import-corrected   # read corrections back

The seeds are deliberately NOT derived from MNDWI or NDVI - see dl/labels.py for
why that matters. Hand correction is the step that turns them into something a
model can be honestly scored against.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dl import labels as L

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-qgis", action="store_true",
                    help="write GeoTIFFs for hand correction in QGIS")
    ap.add_argument("--import-corrected", action="store_true",
                    help="read hand-corrected GeoTIFFs back over the seeds")
    a = ap.parse_args()

    if a.import_corrected:
        L.import_corrected()
    elif a.export_qgis:
        L.export_for_qgis()
    else:
        df = L.build_all()
        print("\nlabel_qa.csv written to outputs/dl/labels/")
        print("next: python src/12_make_labels.py --export-qgis, correct in QGIS,")
        print("      then python src/12_make_labels.py --import-corrected")
