"""Capture quality checks (glare, blur, exposure) and the camera ghost-outline PNG."""

from __future__ import annotations

import cv2
import numpy as np

from .config import Params


def _gray(img: np.ndarray) -> np.ndarray:
    """Gray float32 0..255 from a BGR or gray image."""
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3].astype(np.uint8, copy=False), cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32, copy=False)


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(h, w)
    if max_side <= 0 or side <= max_side:
        return img
    s = max_side / side
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)


def glare_mask(bgr: np.ndarray, params: Params) -> np.ndarray:
    """Bright, unsaturated (specular) pixels: HSV V >= glare_v and S <= glare_s.

    Small speckles are removed by an opening, then the mask is dilated by
    `params.glare_dilate_px` to cover the halo around each highlight.
    """
    hsv = cv2.cvtColor(bgr[..., :3].astype(np.uint8, copy=False), cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] >= params.glare_v) & (hsv[..., 1] <= params.glare_s)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    r = int(params.glare_dilate_px)
    if r > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return m.astype(bool)


# blur_score rescales to this gray-level std, a typical contrast of a part photo, so its
# values stay comparable with a plain variance of the Laplacian (and params.min_blur).
_REF_CONTRAST = 20.0
_MIN_CONTRAST = 4.0   # floor for nearly uniform images


def blur_score(gray: np.ndarray) -> float:
    """Sharpness: variance of the Laplacian normalised to a reference contrast (higher = sharper).

    var(Laplacian) scales with the square of the image gain, so a dimmer lamp or a lower
    auto-exposure would read as "blurry". Dividing by the gray-level variance (floored)
    and rescaling to `_REF_CONTRAST` makes the score invariant to global gain.
    """
    g = _gray(gray)
    contrast = max(float(g.std()), _MIN_CONTRAST)
    lap_var = float(cv2.Laplacian(g, cv2.CV_32F).var())
    return lap_var * (_REF_CONTRAST / contrast) ** 2


def is_too_dark(gray: np.ndarray, params: Params) -> bool:
    """True if the mean gray level is below `params.dark_mean`."""
    return bool(float(_gray(gray).mean()) < params.dark_mean)


def capture_report(bgr: np.ndarray, params: Params) -> dict:
    """Quality summary of a single capture, evaluated at the working resolution.

    Returns `{blur, too_blurry, glare_frac, too_dark, warnings}` where warnings are
    machine codes among "blurry", "glare", "dark".
    """
    img = _resize_max_side(bgr, params.work_max_side)
    gray = _gray(img)
    blur = blur_score(gray)
    too_blurry = blur < params.min_blur
    glare_frac = float(glare_mask(img, params).mean())
    too_dark = is_too_dark(gray, params)

    warnings = []
    if too_blurry:
        warnings.append("blurry")
    if glare_frac > params.max_glare_frac:
        warnings.append("glare")
    if too_dark:
        warnings.append("dark")
    return {
        "blur": blur,
        "too_blurry": bool(too_blurry),
        "glare_frac": glare_frac,
        "too_dark": too_dark,
        "warnings": warnings,
    }


def outline_png(bgr: np.ndarray, max_side: int = 720) -> bytes:
    """RGBA PNG of the image edges: white opaque edges on a transparent background.

    Used as the semi-transparent "ghost" of the BEFORE shot overlaid on the camera.
    Canny thresholds adapt to the image's gradient distribution.
    """
    img = _resize_max_side(bgr, max_side)
    gray = _gray(img).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Canny compares |gx| + |gy| (3x3 Sobel) against its thresholds; pick the high
    # threshold so roughly the strongest 10 % of gradients seed edges.
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    hi = max(float(np.percentile(np.abs(gx) + np.abs(gy), 90)), 40.0)
    edges = cv2.Canny(blurred, 0.4 * hi, hi)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))

    rgba = np.zeros((*edges.shape, 4), np.uint8)
    on = edges > 0
    rgba[on] = (255, 255, 255, 255)  # white is identical in BGRA and RGBA order
    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
