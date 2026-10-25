"""
anylake/enhance.py
==================
Image enhancement for Sentinel-2 bands, and before/after measurements.

Three classic image-processing steps, applied to every band separately, on
surface reflectance:

  1. Denoising - bilateral filter. Averages each pixel with neighbours that
     are both close by and similar in value, so random sensor noise is smoothed
     while edges (a shoreline, a road) are kept.
  2. Contrast enhancement - CLAHE (Contrast Limited Adaptive Histogram
     Equalization). Stretches contrast separately in tiles of 64 x 64 pixels
     (1.28 km), with a clip limit so noise is not amplified. Tiles are a fixed
     size on the ground, so every lake is enhanced the same way.
  3. Sharpening - unsharp mask. Adds back a fraction of the difference
     between the image and a blurred copy, which crisps edges.

The spectral indices (NDWI, MNDWI, NDVI, NDBI) are recomputed from the
enhanced bands. Enhancement changes the reflectance values, so whether it
actually helps the model is not assumed: the U-Net is trained once on raw and
once on enhanced input and the two are compared (19_anylake_library.py).
"""

from __future__ import annotations

import numpy as np

LIMIT = 0.6                 # reflectance mapped to [0, 1] before enhancing
TILE_PX = 64                # CLAHE tile, pixels (1.28 km at 20 m)
CLIP = 2.0                  # CLAHE clip limit
BILATERAL = dict(d=5, sigmaColor=0.04, sigmaSpace=1.5)
SHARPEN_SIGMA, SHARPEN_AMOUNT = 1.0, 0.6


def enhance_band(refl: np.ndarray) -> np.ndarray:
    import cv2

    x = np.clip(refl, 0, LIMIT).astype("float32") / LIMIT
    x = cv2.bilateralFilter(x, **BILATERAL)                                     # 1 denoise
    h, w = x.shape
    clahe = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=(max(1, round(w / TILE_PX)),
                                                          max(1, round(h / TILE_PX))))
    x = clahe.apply((np.clip(x, 0, 1) * 65535).astype("uint16")).astype("float32") / 65535  # 2 contrast
    blur = cv2.GaussianBlur(x, (0, 0), SHARPEN_SIGMA)
    x = np.clip(x + SHARPEN_AMOUNT * (x - blur), 0, 1)                        # 3 sharpen
    return x * LIMIT


def enhance_bands(refl: dict) -> dict:
    """Enhance every band; nodata (reflectance 0) stays 0."""
    out = {}
    for b, a in refl.items():
        e = enhance_band(a)
        e[a <= 0] = 0
        out[b] = e
    return out


# ==========================================================================
# Before / after
# ==========================================================================
def truecolor(refl: dict, hi: float) -> np.ndarray:
    """Plain linear true colour, 0..hi reflectance -> 0..1, no per-image stretch."""
    return np.clip(np.dstack([refl["red"], refl["green"], refl["blue"]]) / hi, 0, 1)


def quality(rgb: np.ndarray) -> dict:
    """
    Standard no-reference image-quality numbers on the grey-level image:

      contrast   RMS contrast - standard deviation of grey level (0-255)
      entropy    information content of the grey-level histogram, bits (max 8)
      sharpness  variance of the Laplacian - higher means crisper edges
      noise      Immerkaer's fast noise estimate, grey levels
    """
    import cv2

    grey = (rgb @ np.array([0.299, 0.587, 0.114])) * 255
    g8 = np.clip(grey, 0, 255).astype("uint8")
    hist = np.bincount(g8.ravel(), minlength=256) / g8.size
    nz = hist[hist > 0]
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], "float32")
    h, w = grey.shape
    noise = (np.sqrt(np.pi / 2) / (6 * (w - 2) * (h - 2))
             * np.abs(cv2.filter2D(grey.astype("float32"), -1, k)[1:-1, 1:-1]).sum())
    return {"contrast": float(grey.std()),
            "entropy": float(-(nz * np.log2(nz)).sum()),
            "sharpness": float(cv2.Laplacian(grey.astype("float32"), cv2.CV_32F).var()),
            "noise": float(noise)}


def before_after(raw_refl: dict, win=None):
    """
    (before_rgb, after_rgb, metrics). "Before" is the raw bands with the usual
    0-0.3 reflectance display stretch; "after" is the enhanced bands, which
    CLAHE has already spread across the 0-0.6 range.
    """
    sub = {b: raw_refl[b] for b in ("red", "green", "blue")}
    enh = enhance_bands(sub)                     # whole grid, exactly as the model sees it
    if win is not None:
        r0, r1, c0, c1 = win
        sub = {b: a[r0:r1, c0:c1] for b, a in sub.items()}
        enh = {b: a[r0:r1, c0:c1] for b, a in enh.items()}
    before = truecolor(sub, 0.30)
    after = truecolor(enh, LIMIT)
    return before, after, {"before": quality(before), "after": quality(after)}
