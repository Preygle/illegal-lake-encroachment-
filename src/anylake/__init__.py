"""
anylake
=======
The encroachment pipeline generalised from the six Bengaluru lakes to any lake
in India that OpenStreetMap knows about.

    search    find a lake by name (or lat,lon) and get its outline
    fetch     Sentinel-2 L2A, all 12 bands, 2019-2025, on the lake's own grid
    lake      lake geometry, fence rings, input channels and seed labels
    enhance   image enhancement (denoise -> CLAHE contrast -> sharpen)
    unet_lib  the U-Net, trained on many lakes that each have their own grid
    vectorize image -> vector: per-pixel feature vectors and raster -> polygons
    geofence  what sits inside the 30 m / 75 m buffers
    forecast  Random Forest that projects encroachment three years ahead
    render    vector -> image: polygons burned back onto the satellite image
    pipeline  all of the above for one lake, timed

Nothing here touches the original Bengaluru grid or its outputs: every lake
gets its own folder under data/lakes/ and outputs/anylake/.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAKES = ROOT / "data" / "lakes"            # per-lake imagery and labels
OUT = ROOT / "outputs" / "anylake"         # results, models, experiment
MODELS = OUT / "models"

YEARS = list(range(2019, 2026))
RES = 20.0                                 # metres, same as the Bengaluru grid
HORIZON = 3                                # forecast years ahead (2025 -> 2028)
