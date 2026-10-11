"""Procedural synthetic data for the QA station: parts, defects, lighting, motion.

Two generations live here:
- legacy car-panel data (``make_surface``, ``apply_lighting``, ``make_pair``),
  kept for existing tests and experiments;
- QA-station data (``make_part``, ``apply_factory_lighting``,
  ``make_station_pair``, ``make_reference_set``): industrial parts on a
  conveyor belt / workbench under shop-floor lighting drift.

Everything is generated from a ``np.random.Generator`` so results are fully
deterministic for a given seed. No external images are used, and nothing here
draws faces, text or licence plates.

Conventions:
- Images are BGR ``uint8`` arrays (OpenCV order), shape ``H x W x 3``.
- Defect functions never modify their input; they return ``(new_img, mask)``
  where ``mask`` is a ``bool[H, W]`` marking the pixels the defect covers.
  All of them accept an optional ``region`` (bool[H, W]) restricting where the
  defect is centred (e.g. the part's material).
- ``random_homography`` returns a 3x3 matrix mapping BEFORE pixel coordinates to
  AFTER pixel coordinates; ``warp(img, H)`` renders the AFTER view. Station
  pairs use the same convention: ``meta["H"]`` maps REFERENCE -> FRAME.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# Typical car paint colours (BGR).
_CAR_COLOURS_BGR = [
    (236, 236, 233),  # white
    (196, 194, 190),  # silver
    (44, 43, 46),     # black
    (38, 36, 172),    # red
    (150, 78, 28),    # blue
    (98, 98, 102),    # graphite grey
    (150, 176, 194),  # beige / champagne
    (62, 92, 52),     # dark green
    (36, 120, 210),   # orange
    (120, 40, 30),    # navy
]

LIGHTING_KINDS = ("gamma", "exposure", "shadow", "warm", "dusk", "flash")
CHANGE_KINDS = (None, "scratch", "tape", "sticker", "water")


# --------------------------------------------------------------------------- #
# Small drawing helpers
# --------------------------------------------------------------------------- #

def _scale(h: int, w: int) -> float:
    """Size factor relative to a 720 px short side (shapes scale with the image)."""
    return min(h, w) / 720.0


def _rounded_box(h, w, cx, cy, half_w, half_h, radius, angle_deg=0.0, soft=1.0, roi=None):
    """Anti-aliased coverage of a rotated rounded rectangle.

    Uses a signed distance function evaluated only over the shape's bounding
    region (or over ``roi`` if given, so several concentric shapes can share one
    region). Returns ``(slices, alpha)`` where ``alpha`` is float32 in [0, 1] for
    ``img[slices]``; ``slices`` is ``None`` if the shape lies outside the image.
    A circle is ``half_w == half_h == radius``; a stadium is ``radius == half_h``.
    """
    half_w, half_h = max(float(half_w), 0.5), max(float(half_h), 0.5)
    if roi is None:
        ext = math.hypot(half_w, half_h) + 2.0 * soft + 2.0
        x0, x1 = max(0, int(cx - ext)), min(w, int(cx + ext) + 1)
        y0, y1 = max(0, int(cy - ext)), min(h, int(cy + ext) + 1)
        if x0 >= x1 or y0 >= y1:
            return None, None
    else:
        y0, y1, x0, x1 = roi[0].start, roi[0].stop, roi[1].start, roi[1].stop
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    u = np.abs(c * dx + s * dy)
    v = np.abs(-s * dx + c * dy)
    r = float(min(radius, half_w, half_h))
    qx = u - (half_w - r)
    qy = v - (half_h - r)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    d = outside + inside - r
    alpha = np.clip(0.5 - d / soft, 0.0, 1.0).astype(np.float32)
    return (slice(y0, y1), slice(x0, x1)), alpha


def _over(img: np.ndarray, sl, alpha: np.ndarray, colour) -> None:
    """Alpha-composite a flat colour onto float image region ``img[sl]`` in place."""
    if sl is None:
        return
    a = alpha[..., None]
    img[sl] = img[sl] * (1.0 - a) + np.asarray(colour, np.float32) * a


def _mul(img: np.ndarray, sl, alpha: np.ndarray, factor: float) -> None:
    """Multiply ``img[sl]`` by ``factor`` weighted by ``alpha`` (darken/brighten)."""
    if sl is None:
        return
    img[sl] = img[sl] * (1.0 + (factor - 1.0) * alpha[..., None])


def _screen(img: np.ndarray, alpha: np.ndarray, strength: float) -> np.ndarray:
    """Screen-blend white light: brightens without hard clipping."""
    a = (alpha * strength)[..., None]
    return img + (255.0 - img) * a


def _pick_center(rng, h, w, margin_frac=0.15):
    """Random point away from the borders (so defects survive a small warp)."""
    my, mx = int(h * margin_frac), int(w * margin_frac)
    return float(rng.uniform(mx, w - mx)), float(rng.uniform(my, h - my))


def _pick_center_in(rng, h, w, region=None, margin_frac=0.15):
    """Random point inside ``region`` (bool mask); falls back to ``_pick_center``.

    With ``region=None`` the rng is consumed exactly like ``_pick_center`` so
    legacy outputs stay bit-identical.
    """
    if region is None:
        return _pick_center(rng, h, w, margin_frac)
    ys, xs = np.nonzero(region)
    if len(ys) == 0:
        return _pick_center(rng, h, w, margin_frac)
    i = int(rng.integers(len(ys)))
    return float(xs[i]), float(ys[i])


def _bezier(p0, p1, p2, n=64) -> np.ndarray:
    """Points on a quadratic Bezier curve, int32 [n, 2] (x, y)."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = (1 - t) ** 2 * np.asarray(p0) + 2 * (1 - t) * t * np.asarray(p1) + t ** 2 * np.asarray(p2)
    return np.round(pts).astype(np.int32)


def _to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def _smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Large-sigma Gaussian blur computed at reduced resolution (much faster)."""
    h, w = arr.shape[:2]
    f = int(max(1, min(8, sigma // 4)))
    if f == 1:
        return cv2.GaussianBlur(arr, (0, 0), sigma)
    small = cv2.resize(arr, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _add_noise(img: np.ndarray, rng, sigma: float) -> np.ndarray:
    """Additive Gaussian sensor noise."""
    noisy = img.astype(np.float32) + np.float32(sigma) * rng.standard_normal(img.shape, np.float32)
    return _to_u8(noisy)


# --------------------------------------------------------------------------- #
# Surface
# --------------------------------------------------------------------------- #

def make_surface(rng: np.random.Generator, h: int = 720, w: int = 1280) -> np.ndarray:
    """Procedural painted car panel photo (BGR uint8).

    Layers: multi-stop colour gradient in a random car colour, low-frequency
    shading, a body crease, broad soft reflections, 1-3 panel seams, a couple
    of rounded emblem/handle shapes and fine paint texture. Large areas stay
    smooth (flat), while seams/emblems give ORB enough corners to align on.
    """
    k = _scale(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Base colour with a small per-channel jitter.
    base = np.asarray(_CAR_COLOURS_BGR[rng.integers(len(_CAR_COLOURS_BGR))], np.float32)
    base = np.clip(base * rng.uniform(0.92, 1.08, 3), 10, 245).astype(np.float32)

    # Multi-stop brightness gradient along a random direction, smoothed.
    theta = rng.uniform(0, math.pi)
    proj = xx * math.cos(theta) + yy * math.sin(theta)
    proj = (proj - proj.min()) / max(float(np.ptp(proj)), 1.0)
    n_stops = int(rng.integers(3, 6))
    sx = np.concatenate([[0.0], np.sort(rng.uniform(0, 1, n_stops)), [1.0]])
    sv = rng.uniform(0.72, 1.15, n_stops + 2)
    shade = np.interp(proj.ravel(), sx, sv).reshape(h, w).astype(np.float32)
    shade = _smooth(shade, max(2.0, 40 * k))

    # Low-frequency shading (body curvature).
    lf = rng.normal(0, 1, (4, 6)).astype(np.float32)
    shade *= 1.0 + 0.05 * cv2.resize(lf, (w, h), interpolation=cv2.INTER_CUBIC)

    # Body crease: soft brightness step along a gentle curve.
    y_c = rng.uniform(0.3, 0.7) * h
    bend = rng.uniform(-0.08, 0.08) * h
    crease_y = y_c + bend * np.sin(math.pi * xx / w)
    step_ = 0.5 * (1.0 + np.tanh((yy - crease_y) / max(2.0, 6 * k)))  # smooth 0..1
    shade *= 1.0 + rng.uniform(-0.12, 0.12) * (step_ - 0.5)

    img = base[None, None, :] * shade[..., None]

    # Broad soft reflections (elongated light bands).
    # Drawn at 1/4 resolution: the layer is heavily blurred anyway.
    sh, sw = max(1, h // 4), max(1, w // 4)
    refl = np.zeros((sh, sw), np.float32)
    for _ in range(int(rng.integers(2, 5))):
        cx, cy = rng.uniform(0, sw), rng.uniform(0, sh)
        hw = rng.uniform(0.25, 0.6) * sw
        hh = rng.uniform(0.02, 0.08) * sh
        sl, a = _rounded_box(sh, sw, cx, cy, hw, hh, hh, rng.uniform(-25, 25))
        if sl is not None:
            refl[sl] = np.maximum(refl[sl], a * rng.uniform(0.3, 1.0))
    refl = cv2.GaussianBlur(refl, (0, 0), max(0.5, 4.5 * k))
    refl = cv2.resize(refl, (w, h), interpolation=cv2.INTER_LINEAR)
    img = _screen(img, refl, rng.uniform(0.25, 0.5))

    # Panel seams: dark thin curves with a faint highlight next to them.
    seam_dark = np.zeros((h, w), np.uint8)
    seam_light = np.zeros((h, w), np.uint8)
    for _ in range(int(rng.integers(1, 4))):
        if rng.random() < 0.5:  # roughly vertical seam
            x0 = rng.uniform(0.1, 0.9) * w
            p0, p2 = (x0, -5.0), (x0 + rng.uniform(-0.1, 0.1) * w, h + 5.0)
        else:  # roughly horizontal seam
            y0 = rng.uniform(0.1, 0.9) * h
            p0, p2 = (-5.0, y0), (w + 5.0, y0 + rng.uniform(-0.1, 0.1) * h)
        mid = ((p0[0] + p2[0]) / 2 + rng.uniform(-0.06, 0.06) * w,
               (p0[1] + p2[1]) / 2 + rng.uniform(-0.06, 0.06) * h)
        pts = _bezier(p0, mid, p2)
        th = max(1, int(round(rng.uniform(1.5, 3.0) * k)))
        cv2.polylines(seam_dark, [pts], False, 255, th, cv2.LINE_AA)
        cv2.polylines(seam_light, [pts + th], False, 255, 1, cv2.LINE_AA)
    img = img * (1.0 - 0.65 * (seam_dark[..., None] / 255.0))
    img = _screen(img, seam_light / 255.0, 0.3)

    # Emblems / handles: rounded shapes with a rim, a recess and a highlight.
    for _ in range(int(rng.integers(2, 4))):
        cx, cy = _pick_center(rng, h, w, 0.12)
        if rng.random() < 0.5:  # door handle (stadium shape)
            hw, hh = rng.uniform(50, 90) * k, rng.uniform(10, 18) * k
            ang = rng.uniform(-6, 6)
        else:  # round / oval emblem
            hw = rng.uniform(16, 36) * k
            hh = hw * rng.uniform(0.6, 1.0)
            ang = rng.uniform(-20, 20)
        sl, outer = _rounded_box(h, w, cx, cy, hw, hh, hh, ang)
        if sl is None:
            continue
        b = 3 * k + 1
        _, inner = _rounded_box(h, w, cx, cy, hw - b, hh - b, hh - b, ang, roi=sl)
        hr = max(1.5, hh * 0.18)
        _, hi = _rounded_box(h, w, cx, cy - hh * 0.4, hw * 0.7, hr, hr, ang, roi=sl)
        rim = np.clip(outer - inner, 0.0, 1.0)
        chrome = rng.random() < 0.5
        _over(img, sl, rim, (225, 225, 228) if chrome else (25, 25, 28))
        _mul(img, sl, inner, rng.uniform(0.55, 0.8))
        img[sl] = _screen(img[sl], hi * inner, 0.45)

    # A few tiny pre-existing stone chips (present in both frames).
    for _ in range(int(rng.integers(6, 16))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.8, 2.2) * max(k, 0.5)
        sl, a = _rounded_box(h, w, cx, cy, r, r * rng.uniform(0.6, 1.0), r, rng.uniform(0, 180))
        _mul(img, sl, a, rng.choice([0.45, 1.5]))

    # Fine paint texture: metallic flake + orange peel.
    flake = 2.2 * rng.standard_normal((h, w), np.float32)
    peel = cv2.GaussianBlur(6.0 * rng.standard_normal((h, w), np.float32), (0, 0), 1.6)
    img = img + (flake + peel)[..., None]
    return _to_u8(img)


# --------------------------------------------------------------------------- #
# Defects (each returns a new image and a bool mask of covered pixels)
# --------------------------------------------------------------------------- #

def add_scratch(img: np.ndarray, rng: np.random.Generator, region=None):
    """Thin, slightly irregular bright or dark polyline, 40-200 px long, 1-3 px wide."""
    h, w = img.shape[:2]
    length = rng.uniform(40, 200)
    cx, cy = _pick_center_in(rng, h, w, region)
    theta = rng.uniform(0, 2 * math.pi)
    n_seg = max(4, int(length // 10))
    seg = length / n_seg
    # Start half a length behind the centre so the scratch is centred.
    x, y = cx - 0.5 * length * math.cos(theta), cy - 0.5 * length * math.sin(theta)
    pts = [(x, y)]
    for _ in range(n_seg):
        theta += rng.normal(0, 0.07)
        step = seg * rng.uniform(0.8, 1.2)
        x += step * math.cos(theta) + rng.normal(0, 0.4)
        y += step * math.sin(theta) + rng.normal(0, 0.4)
        pts.append((x, y))

    width = int(rng.integers(1, 4))
    layer = np.zeros((h, w), np.uint8)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        th = int(np.clip(width + rng.integers(-1, 2), 1, 3))
        val = int(255 * rng.uniform(0.6, 1.0))
        # shift=4: coordinates in 1/16 px for sub-pixel accurate drawing.
        p0 = (int(round(x0 * 16)), int(round(y0 * 16)))
        p1 = (int(round(x1 * 16)), int(round(y1 * 16)))
        cv2.line(layer, p0, p1, val, th, cv2.LINE_AA, 4)

    alpha = (layer.astype(np.float32) / 255.0)[..., None]
    out = img.astype(np.float32)
    if rng.random() < 0.6:  # clear coat / primer exposed: bright
        out = out + (rng.uniform(225, 255) - out) * alpha * 0.85
    else:  # deep scratch: dark
        out = out * (1.0 - 0.65 * alpha)
    mask = layer >= 64
    if not mask.any():  # degenerate (e.g. tiny image): fall back to a dot
        mask[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))] = True
    return _to_u8(out), mask


def add_tape(img: np.ndarray, rng: np.random.Generator, region=None):
    """Rotated strip of masking tape (opaque-ish paper) or clear tape (glossy haze)."""
    h, w = img.shape[:2]
    k = _scale(h, w)
    cx, cy = _pick_center_in(rng, h, w, region)
    hw = rng.uniform(45, 130) * k + 4
    hh = rng.uniform(14, 30) * k + 3
    ang = rng.uniform(-90, 90)
    sl, a = _rounded_box(h, w, cx, cy, hw, hh, 1.0, ang)
    _, a_in = _rounded_box(h, w, cx, cy, hw - 2, hh - 2, 1.0, ang, roi=sl)
    edge = np.clip(a - a_in, 0.0, 1.0)

    out = img.astype(np.float32)
    roi = out[sl]
    if rng.random() < 0.5:  # masking tape: beige paper with fibre texture
        colour = np.array([150, 200, 222], np.float32) * rng.uniform(0.9, 1.05)
        opacity = rng.uniform(0.78, 0.92)
        tex = cv2.GaussianBlur(rng.normal(0, 7, a.shape).astype(np.float32), (0, 0), 0.8)
        paper = colour[None, None, :] + tex[..., None]
        aa = (a * opacity)[..., None]
        roi = roi * (1 - aa) + paper * aa
        roi = roi * (1.0 - 0.35 * edge[..., None])
    else:  # clear tape: slight haze + glossy streak + visible edges
        aa = a[..., None]
        roi = roi * (1 - 0.18 * aa) + 255.0 * 0.14 * aa
        sr = max(1.5, hh * 0.15)
        _, streak = _rounded_box(h, w, cx, cy - hh * 0.3, hw * 0.9, sr, sr, ang, roi=sl)
        roi = _screen(roi, streak * a, rng.uniform(0.25, 0.45))
        roi = _screen(roi, edge, 0.5) * (1.0 - 0.15 * edge[..., None])
    out[sl] = roi
    mask = np.zeros((h, w), bool)
    mask[sl] = a >= 0.5
    return _to_u8(out), mask


def add_sticker(img: np.ndarray, rng: np.random.Generator, region=None):
    """Opaque rounded sticker with a white border and a second-colour motif."""
    h, w = img.shape[:2]
    k = _scale(h, w)
    cx, cy = _pick_center_in(rng, h, w, region)
    shape = rng.integers(3)
    if shape == 0:  # circle
        hw = hh = rng.uniform(25, 60) * k + 4
        r = hw
    elif shape == 1:  # rounded rectangle
        hw, hh = rng.uniform(30, 80) * k + 4, rng.uniform(20, 50) * k + 4
        r = min(hw, hh) * rng.uniform(0.2, 0.5)
    else:  # stadium
        hw, hh = rng.uniform(35, 80) * k + 4, rng.uniform(15, 30) * k + 3
        r = hh
    ang = rng.uniform(-45, 45)
    border = max(1.5, 3 * k)
    sl, outer = _rounded_box(h, w, cx, cy, hw, hh, r, ang)
    _, body = _rounded_box(h, w, cx, cy, hw - border, hh - border, max(r - border, 0.0),
                           ang, roi=sl)

    c1 = rng.uniform(20, 235, 3)
    c2 = 255.0 - c1  # contrasting second colour
    if rng.random() < 0.5:  # stripe motif
        _, motif = _rounded_box(h, w, cx, cy, hw * 1.5, hh * 0.25, 0.0, ang, roi=sl)
    else:  # inner disc motif
        rr = min(hw, hh) * 0.45
        _, motif = _rounded_box(h, w, cx + hw * 0.2, cy, rr, rr, rr, ang, roi=sl)
    gr = hh * 0.2
    _, gloss = _rounded_box(h, w, cx - hw * 0.2, cy - hh * 0.35, hw * 0.5, gr, gr, ang, roi=sl)

    out = img.astype(np.float32)
    _over(out, sl, outer, (245, 245, 245))
    _over(out, sl, body, c1)
    _over(out, sl, motif * body, c2)
    # Gentle gloss so it reads as a printed sticker.
    out[sl] = _screen(out[sl], cv2.GaussianBlur(gloss, (0, 0), 2.0) * body, 0.25)
    mask = np.zeros((h, w), bool)
    mask[sl] = outer >= 0.5
    return _to_u8(out), mask


def add_water_drops(img: np.ndarray, rng: np.random.Generator, region=None):
    """Cluster of many small round drops: dark rim, slightly darker body, specular dot."""
    h, w = img.shape[:2]
    k = _scale(h, w)
    ccx, ccy = _pick_center_in(rng, h, w, region)
    spread = rng.uniform(50, 150) * k + 10
    n = int(rng.integers(18, 50))
    out = img.astype(np.float32)
    mask = np.zeros((h, w), bool)
    for _ in range(n):
        # Mostly small drops, a few larger ones.
        r = float(np.clip(rng.gamma(2.0, 2.2) * k + 2.5, 2.5, 12.0 * max(k, 0.5)))
        cx = float(np.clip(ccx + rng.normal(0, spread * 0.5), r, w - 1 - r))
        cy = float(np.clip(ccy + rng.normal(0, spread * 0.5), r, h - 1 - r))
        sq = rng.uniform(0.8, 1.0)  # slightly oval drops
        ang = rng.uniform(0, 180)
        sl, disc = _rounded_box(h, w, cx, cy, r, r * sq, r, ang)
        if sl is None:
            continue
        ri = max(r - 1.5, 0.8)
        _, inner = _rounded_box(h, w, cx, cy, ri, ri * sq, ri, ang, roi=sl)
        rim = np.clip(disc - inner, 0.0, 1.0)
        dr = max(0.8, 0.28 * r)
        _, dot = _rounded_box(h, w, cx - 0.35 * r, cy - 0.35 * r, dr, dr, dr, roi=sl)
        _mul(out, sl, inner, rng.uniform(0.82, 0.92))
        _mul(out, sl, rim, 0.55)
        _over(out, sl, dot * disc, (250, 250, 250))
        mask[sl] |= disc >= 0.5
    return _to_u8(out), mask


# --------------------------------------------------------------------------- #
# Lighting
# --------------------------------------------------------------------------- #

def _away_from_one(rng, lo, hi, gap=0.15):
    """Sample uniformly from [lo, hi] but at least ``gap`` away from 1 (visible change)."""
    while True:
        v = rng.uniform(lo, hi)
        if abs(v - 1.0) >= gap:
            return v


def apply_lighting(img: np.ndarray, rng: np.random.Generator, kind: str) -> np.ndarray:
    """Global/local illumination change; returns a new BGR uint8 image.

    kinds: "gamma" (0.6-1.6), "exposure" (x0.6-1.4, clipped), "shadow" (soft
    multiplicative shape, factor 0.5-1.0), "warm" (colour temperature shift),
    "dusk" (darker, lower contrast, more noise), "flash" (central specular blob
    that saturates to 255 + vignetting).
    """
    h, w = img.shape[:2]
    x = img.astype(np.float32)
    if kind == "gamma":
        g = _away_from_one(rng, 0.6, 1.6)
        out = 255.0 * (x / 255.0) ** g
    elif kind == "exposure":
        out = x * _away_from_one(rng, 0.6, 1.4)
    elif kind == "shadow":
        # Soft shadow shapes, drawn at 1/4 resolution and blurred.
        sh, sw = max(1, h // 4), max(1, w // 4)
        layer = np.zeros((sh, sw), np.float32)
        for _ in range(int(rng.integers(1, 3))):
            sl, a = _rounded_box(sh, sw, rng.uniform(0, sw), rng.uniform(0, sh),
                                 rng.uniform(0.2, 0.5) * sw, rng.uniform(0.2, 0.5) * sh,
                                 rng.uniform(0, 0.2) * sh, rng.uniform(-40, 40))
            if sl is not None:
                layer[sl] = np.maximum(layer[sl], a)
        layer = cv2.GaussianBlur(layer, (0, 0), max(1.0, min(sh, sw) / 10))
        layer = cv2.resize(layer, (w, h), interpolation=cv2.INTER_LINEAR)
        # Plus a gentle linear gradient across the frame.
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = rng.uniform(0, 2 * math.pi)
        grad = xx * math.cos(t) + yy * math.sin(t)
        grad = (grad - grad.min()) / max(float(np.ptp(grad)), 1.0)
        darkness = np.clip(0.75 * layer + 0.25 * grad, 0, 1) * rng.uniform(0.3, 0.5)
        out = x * (1.0 - darkness)[..., None]
    elif kind == "warm":
        gains = np.array([rng.uniform(0.78, 0.9), rng.uniform(0.97, 1.03),
                          rng.uniform(1.08, 1.2)], np.float32)  # B, G, R
        out = x * gains
    elif kind == "dusk":
        mean = float(x.mean())
        out = (x - mean) * rng.uniform(0.6, 0.8) + mean
        out = out * rng.uniform(0.45, 0.65) * np.array([1.06, 1.0, 0.92], np.float32)
        out = out + np.float32(rng.uniform(4, 7)) * rng.standard_normal(x.shape, np.float32)
    elif kind == "flash":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        vignette = 1.0 - rng.uniform(0.25, 0.45) * np.clip(r2 / 2.0, 0, 1)
        out = x * rng.uniform(1.0, 1.15) * vignette[..., None]
        cx = w / 2 + rng.uniform(-0.12, 0.12) * w
        cy = h / 2 + rng.uniform(-0.12, 0.12) * h
        s = rng.uniform(0.06, 0.12) * min(h, w)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
        out = out + 400.0 * blob[..., None]  # the core saturates to 255
    else:
        raise ValueError(f"unknown lighting kind: {kind!r}")
    return _to_u8(out)


# --------------------------------------------------------------------------- #
# Motion
# --------------------------------------------------------------------------- #

def random_homography(rng: np.random.Generator, h: int, w: int, max_shift_px: float = 6,
                      max_rot_deg: float = 1.5, max_scale: float = 0.02) -> np.ndarray:
    """Small random homography (BEFORE -> AFTER pixel coords) about the image centre.

    Combines a shift, a rotation, an isotropic scale and a tiny perspective term
    (its corner displacement is about a quarter of ``max_shift_px``).
    """
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    a = math.radians(rng.uniform(-max_rot_deg, max_rot_deg))
    s = 1.0 + rng.uniform(-max_scale, max_scale)
    tx, ty = rng.uniform(-max_shift_px, max_shift_px, 2)
    half = max(h, w) / 2.0
    p = rng.uniform(-1, 1, 2) * 0.25 * max_shift_px / (half * half)

    to_c = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
    from_c = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], np.float64)
    rsp = np.array([[s * math.cos(a), -s * math.sin(a), 0],
                    [s * math.sin(a), s * math.cos(a), 0],
                    [p[0], p[1], 1]], np.float64)
    H = from_c @ rsp @ to_c
    return H / H[2, 2]


def warp(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp ``img`` by ``H`` (source -> destination coords), same size, reflected borders."""
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, np.asarray(H, np.float64), (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------- #
# Pairs
# --------------------------------------------------------------------------- #

_DEFECTS = {
    "scratch": add_scratch,
    "tape": add_tape,
    "sticker": add_sticker,
    "water": add_water_drops,
}


def make_pair(rng: np.random.Generator, change: str | None, lighting: str | None,
              motion: bool, h: int = 720, w: int = 1280) -> dict:
    """Synthetic BEFORE/AFTER pair.

    Steps: surface -> (AFTER only) defect -> warp by H if ``motion`` -> lighting
    -> independent sensor noise (sigma 2-4) on both images.

    Returns ``dict(before, after, change_mask_before_frame, meta)``. The mask is
    in the BEFORE frame (the defect is drawn before the warp). ``meta["H"]`` maps
    BEFORE pixel coords to AFTER pixel coords (identity without motion), so an
    aligner's AFTER -> BEFORE matrix should approximate ``inv(meta["H"])``.
    ``h``/``w`` are optional extras (default 720x1280) for fast tests.
    """
    if change not in CHANGE_KINDS:
        raise ValueError(f"unknown change kind: {change!r}")
    if lighting is not None and lighting not in LIGHTING_KINDS:
        raise ValueError(f"unknown lighting kind: {lighting!r}")

    surface = make_surface(rng, h, w)
    after = surface
    mask = np.zeros((h, w), bool)
    if change is not None:
        after, mask = _DEFECTS[change](surface, rng)

    H = random_homography(rng, h, w) if motion else np.eye(3)
    if motion:
        after = warp(after, H)
    if lighting is not None:
        after = apply_lighting(after, rng, lighting)

    sigma_b, sigma_a = rng.uniform(2.0, 4.0, 2)
    before = _add_noise(surface, rng, sigma_b)
    after = _add_noise(after, rng, sigma_a)
    meta = {
        "change": change,
        "lighting": lighting,
        "motion": bool(motion),
        "H": H,
        "noise_sigma": (float(sigma_b), float(sigma_a)),
        "size": (h, w),
    }
    return {"before": before, "after": after, "change_mask_before_frame": mask, "meta": meta}


# =========================================================================== #
# QA station: industrial parts, defects, factory lighting, station pairs
# =========================================================================== #

PART_KINDS = ("stamped", "painted", "molded", "machined")
BACKGROUND_KINDS = ("belt", "steel_table", "esd_mat")
STATION_DEFECTS = ("scratch", "tape", "sticker", "dent", "missing_hole", "burr", "stain")
FACTORY_LIGHTING = ("drift_gain", "gamma", "window_gradient", "lamp_aging", "auto_exposure",
                    "specular", "hand_shadow")
# Monotone (globally, or locally for window_gradient) brightness changes: PBP should be robust.
MONOTONE_LIGHTING = ("drift_gain", "gamma", "window_gradient", "lamp_aging", "auto_exposure")
# Expected failure cases (the local intensity order changes); reported as such by eval.
NON_MONOTONE_LIGHTING = ("specular", "hand_shadow")
MOTION_MODES = ("none", "jitter", "shift")

# Part kind -> backgrounds with enough contrast (dark parts never go on the dark belt).
_BG_FOR_PART = {
    "stamped": ("belt", "esd_mat"),
    "machined": ("belt", "esd_mat"),
    "molded": ("steel_table", "esd_mat"),
}


def _full(h, w, cx, cy, half_w, half_h, radius, angle_deg=0.0, soft=1.0) -> np.ndarray:
    """Full-frame float32 coverage of a rounded box (see ``_rounded_box``)."""
    out = np.zeros((h, w), np.float32)
    sl, a = _rounded_box(h, w, cx, cy, half_w, half_h, radius, angle_deg, soft)
    if sl is not None:
        out[sl] = a
    return out


def _disk(h, w, cx, cy, r, soft=1.0) -> np.ndarray:
    return _full(h, w, cx, cy, r, r, r, 0.0, soft)


def _lowfreq(rng, h, w, gh=4, gw=6) -> np.ndarray:
    """Smooth random field ~N(0, 1) (bicubic upsampling of a coarse grid)."""
    g = rng.normal(0, 1, (gh, gw)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def _grain(rng, h, w, sigma, amp) -> np.ndarray:
    """Isotropic blurred noise with standard deviation ~``amp``."""
    n = rng.standard_normal((h, w), np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    return n * np.float32(amp / max(float(n.std()), 1e-6))


def _brushed(rng, h, w, length, amp) -> np.ndarray:
    """Anisotropic noise streaked along x (brushed / rolled metal)."""
    n = rng.standard_normal((h, w), np.float32)
    n = cv2.blur(n, (max(1, int(length)), 1))
    n = cv2.GaussianBlur(n, (0, 0), 0.6)
    return n * np.float32(amp / max(float(n.std()), 1e-6))


def _emboss(layer: np.ndarray, sigma: float, light_deg: float = 225.0) -> np.ndarray:
    """Directional shading of a raised ``layer`` (0..1): about +-1 at its edges.

    Positive on edges facing the light (default: from the top-left).
    """
    b = cv2.GaussianBlur(layer, (0, 0), max(sigma, 0.5))
    gy, gx = np.gradient(b)
    a = math.radians(light_deg)
    d = -(gx * math.cos(a) + gy * math.sin(a))
    return d * np.float32(max(sigma, 0.5) * math.sqrt(2 * math.pi))


# --------------------------------------------------------------------------- #
# Backgrounds
# --------------------------------------------------------------------------- #

def _make_background(rng, h, w, kind) -> np.ndarray:
    """Conveyor belt, steel workbench or ESD mat (float32 BGR, no strong corners)."""
    k = _scale(h, w)
    if kind == "belt":
        v = rng.uniform(46, 70)
        img = np.full((h, w, 3), v, np.float32) * np.array([1.0, 1.0, 1.03], np.float32)
        # Longitudinal wear streaks (the belt runs along x) + rubber grain.
        streak = cv2.GaussianBlur(rng.standard_normal((h, 1)).astype(np.float32), (0, 0), 2.0)
        img += (3.0 * streak / max(float(streak.std()), 1e-6))[..., None]
        img += _grain(rng, h, w, 0.8, 3.0)[..., None]
        # Belt splice: a slightly darker transverse band.
        x0 = rng.uniform(0.05, 0.95) * w
        band = np.clip(1.0 - np.abs(np.arange(w, dtype=np.float32) - x0) / (4 * k + 1), 0, 1)
        img *= (1.0 - 0.15 * band)[None, :, None]
        # Steel side rails along the top and bottom edges.
        rail = int(round(rng.uniform(0.025, 0.045) * h))
        if rail > 0:
            for sl in (slice(0, rail), slice(h - rail, h)):
                img[sl] = rng.uniform(105, 135) + _brushed(rng, rail, w, 30 * k + 1, 4.0)[..., None]
    elif kind == "steel_table":
        v = rng.uniform(118, 150)
        img = np.full((h, w, 3), v, np.float32) * np.array([1.02, 1.0, 0.98], np.float32)
        img *= (1.0 + 0.05 * _lowfreq(rng, h, w))[..., None]
        img += _brushed(rng, h, w, 60 * k + 1, 5.0)[..., None]
        # Faint old scratches on the bench.
        layer = np.zeros((h, w), np.uint8)
        for _ in range(int(rng.integers(3, 8))):
            x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
            ang, ln = rng.uniform(0, math.pi), rng.uniform(30, 160) * k
            x1, y1 = x0 + ln * math.cos(ang), y0 + ln * math.sin(ang)
            cv2.line(layer, (int(x0), int(y0)), (int(x1), int(y1)), 255, 1, cv2.LINE_AA)
        img += (layer.astype(np.float32) / 255.0 * rng.uniform(6, 12))[..., None]
    elif kind == "esd_mat":
        base = np.array([88, 112, 78], np.float32) * np.float32(rng.uniform(0.85, 1.1))
        img = np.broadcast_to(base, (h, w, 3)).astype(np.float32)
        img = img * (1.0 + 0.04 * _lowfreq(rng, h, w))[..., None]
        img += _grain(rng, h, w, 1.2, 2.5)[..., None]
    else:
        raise ValueError(f"unknown background kind: {kind!r}")
    return img


# --------------------------------------------------------------------------- #
# Parts (rendered in canvas coords, then placed with a small rotation/offset)
# --------------------------------------------------------------------------- #

def _hole(h, w, cx, cy, r, k):
    """Through hole: returns (hole coverage, bright cut-edge ring around it)."""
    hole = _disk(h, w, cx, cy, r)
    ring = np.clip(_disk(h, w, cx, cy, r + 2.5 * k + 1) - hole, 0, 1)
    return hole, ring


def _render_stamped(rng, h, w, k):
    """Brushed steel plate: bend line, embossed ribs, punched holes and a slot."""
    cx, cy = w / 2.0, h / 2.0
    hw, hh = rng.uniform(0.31, 0.39) * w, rng.uniform(0.33, 0.40) * h
    outline = _full(h, w, cx, cy, hw, hh, rng.uniform(8, 40) * k + 1)
    if rng.random() < 0.6:  # corner notch
        sx, sy = rng.choice([-1.0, 1.0], 2)
        outline *= 1.0 - _disk(h, w, cx + sx * hw, cy + sy * hh, rng.uniform(0.15, 0.25) * hh)

    v = rng.uniform(140, 185)
    shade = 1.0 + 0.05 * _lowfreq(rng, h, w)
    # Bend line: the far facet is tilted, so it reflects less light.
    vertical = rng.random() < 0.5
    pos = rng.uniform(0.4, 0.6) * (hw if vertical else hh) * rng.choice([-1.0, 1.0])
    coord = (np.arange(w, dtype=np.float32) - cx)[None, :] if vertical else \
        (np.arange(h, dtype=np.float32) - cy)[:, None]
    u = coord * np.sign(pos) - abs(pos)
    facet = 0.5 * (1.0 + np.tanh(u / (2.0 * k + 0.5)))
    shade = shade * (1.0 + rng.uniform(-0.18, -0.08) * facet)
    bend_line = np.broadcast_to(np.exp(-(u / (2.0 * k + 0.7)) ** 2), (h, w))

    fg = v * shade[..., None] * np.array([1.03, 1.0, 0.96], np.float32)
    fg = fg + _brushed(rng, h, w, 40 * k + 1, rng.uniform(5, 9))[..., None]
    fg = _screen(fg, bend_line, 0.35)

    # Embossed ribs (stiffeners) in the central band.
    ribs = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(1, 3))):
        rx = cx + rng.uniform(-0.15, 0.15) * hw
        ry = cy + rng.uniform(-0.35, 0.35) * hh
        rh = rng.uniform(8, 16) * k + 1
        ribs = np.maximum(ribs, _full(h, w, rx, ry, rng.uniform(0.35, 0.55) * hw, rh, rh))
    fg = fg * (1.0 + 0.35 * _emboss(ribs, 2.0 * k + 0.5))[..., None]

    # Punched holes near the corners / along the long edges, plus an optional slot.
    holes, hole_cov, rings = [], np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    m = rng.uniform(0.12, 0.18) * min(hw, hh)
    spots = [(cx + sx * (hw - m), cy + sy * (hh - m)) for sx in (-1, 1) for sy in (-1, 1)]
    if rng.random() < 0.6:
        spots += [(cx, cy - (hh - m)), (cx, cy + (hh - m))]
    r = rng.uniform(10, 20) * k + 1.5
    for hx, hy in spots:
        if outline[int(np.clip(hy, 0, h - 1)), int(np.clip(hx, 0, w - 1))] < 0.99:
            continue  # cut away by the notch
        cov, ring = _hole(h, w, hx, hy, r, k)
        hole_cov, rings = np.maximum(hole_cov, cov), np.maximum(rings, ring)
        holes.append((hx, hy, r))
    if rng.random() < 0.7:  # elongated slot
        sx = cx + rng.uniform(0.3, 0.5) * hw * rng.choice([-1.0, 1.0])
        sr = rng.uniform(7, 12) * k + 1
        sh = rng.uniform(2.5, 4) * sr
        e = 2.5 * k + 1
        slot = _full(h, w, sx, cy, sr, sh, sr)
        ring = np.clip(_full(h, w, sx, cy, sr + e, sh + e, sr + e) - slot, 0, 1)
        hole_cov, rings = np.maximum(hole_cov, slot), np.maximum(rings, ring)
    fg = _screen(fg, rings, 0.45)
    return fg, outline, hole_cov, holes, (cx, cy)


def _render_painted(rng, h, w, k):
    """Painted body-panel piece: smooth paint, seam, crease, two mounting holes (low texture)."""
    cx, cy = w / 2.0, h / 2.0
    hw, hh = rng.uniform(0.32, 0.39) * w, rng.uniform(0.33, 0.40) * h
    outline = _full(h, w, cx, cy, hw, hh, rng.uniform(30, 80) * k + 2)
    base = np.asarray(_CAR_COLOURS_BGR[rng.integers(len(_CAR_COLOURS_BGR))], np.float32)
    base = np.clip(base * rng.uniform(0.94, 1.06, 3), 20, 240).astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = rng.uniform(0, math.pi)
    proj = ((xx - cx) * math.cos(t) + (yy - cy) * math.sin(t)) / max(hw, hh)
    shade = 1.0 + rng.uniform(0.06, 0.14) * proj + 0.03 * _lowfreq(rng, h, w)
    # Soft crease: a brightness step along a gentle curve.
    y_c = cy + rng.uniform(-0.4, 0.4) * hh + rng.uniform(-0.05, 0.05) * h * np.sin(math.pi * xx / w)
    crease = 0.5 * (1.0 + np.tanh((yy - y_c) / (5.0 * k + 1)))
    shade = shade * (1.0 + rng.uniform(-0.1, 0.1) * (crease - 0.5))
    fg = base * shade[..., None]
    # Broad clear-coat reflection band.
    band = _full(h, w, cx + rng.uniform(-0.3, 0.3) * hw, cy + rng.uniform(-0.3, 0.3) * hh,
                 rng.uniform(0.5, 0.9) * hw, rng.uniform(0.04, 0.1) * hh, 1.0, rng.uniform(-20, 20))
    fg = _screen(fg, _smooth(band, 12 * k + 1), rng.uniform(0.15, 0.3))
    fg = fg + _grain(rng, h, w, 1.6, rng.uniform(2.0, 3.5))[..., None]  # orange peel

    # One seam (gap between panels): dark line with a faint highlight.
    seam_d, seam_l = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    xs = cx + rng.uniform(-0.5, 0.5) * hw
    pts = _bezier((xs, cy - hh - 5), (xs + rng.uniform(-0.1, 0.1) * hw, cy),
                  (xs + rng.uniform(-0.1, 0.1) * hw, cy + hh + 5))
    th = max(1, int(round(2.5 * k + 0.5)))
    cv2.polylines(seam_d, [pts], False, 255, th, cv2.LINE_AA)
    cv2.polylines(seam_l, [pts + th], False, 255, 1, cv2.LINE_AA)
    fg = fg * (1.0 - 0.6 * seam_d[..., None] / 255.0)
    fg = _screen(fg, seam_l / 255.0, 0.25)

    # Two mounting holes along one edge.
    holes, hole_cov = [], np.zeros((h, w), np.float32)
    m = rng.uniform(0.1, 0.16) * min(hw, hh)
    r = rng.uniform(7, 12) * k + 1.5
    sy = rng.choice([-1.0, 1.0])
    for sx in (-1.0, 1.0):
        hx, hy = cx + sx * (hw - m), cy + sy * (hh - m)
        cov, ring = _hole(h, w, hx, hy, r, k)
        hole_cov = np.maximum(hole_cov, cov)
        fg = fg * (1.0 - 0.3 * ring[..., None])  # painted-over dark rim
        holes.append((hx, hy, r))
    return fg, outline, hole_cov, holes, (cx, cy)


def _render_molded(rng, h, w, k):
    """Dark injection-moulded plastic: grain, ribs, bosses with bores, ejector marks."""
    cx, cy = w / 2.0, h / 2.0
    hw, hh = rng.uniform(0.32, 0.39) * w, rng.uniform(0.33, 0.40) * h
    outline = _full(h, w, cx, cy, hw, hh, rng.uniform(20, 60) * k + 2)
    v = rng.uniform(30, 58)
    tint = np.array([1.06, 1.0, 0.97], np.float32) if rng.random() < 0.4 else np.ones(3, np.float32)
    fg = v * tint * (1.0 + 0.06 * _lowfreq(rng, h, w))[..., None]
    fg = fg + _grain(rng, h, w, 1.0, 2.5)[..., None] + _grain(rng, h, w, 0.6, 1.5)[..., None]

    def relief(layer, amp_hi, amp_lo):
        """Additive highlight/shadow (reads better than a gain on dark plastic)."""
        e = _emboss(layer, 1.5 * k + 0.5)
        return (amp_hi * np.maximum(e, 0) - amp_lo * np.maximum(-e, 0))[..., None]

    # Vertical ribs in the left part.
    n = int(rng.integers(3, 7))
    x0, x1 = cx - 0.8 * hw, cx - 0.1 * hw
    rh = rng.uniform(0.4, 0.7) * hh
    rw = rng.uniform(4, 7) * k + 1
    ribs = np.zeros((h, w), np.float32)
    for i in range(n):
        ribs = np.maximum(ribs, _full(h, w, x0 + (x1 - x0) * (i + 0.5) / n, cy, rw, rh, rw))
    fg = fg + relief(ribs, 45.0, 15.0)

    # Bosses (cylinders) with a blind bore, in the right part.
    holes = []
    nb = int(rng.integers(2, 5))
    bosses, bores = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    for i in range(nb):  # stacked vertically, so they never overlap
        bx = cx + rng.uniform(0.25, 0.75) * hw
        by = cy - 0.7 * hh + 1.4 * hh * (i + 0.5) / nb
        br = rng.uniform(16, 28) * k + 2
        bosses = np.maximum(bosses, _disk(h, w, bx, by, br))
        bores = np.maximum(bores, _disk(h, w, bx, by, 0.42 * br))
        holes.append((bx, by, 0.42 * br))
    fg = fg * (1.0 + 0.15 * bosses[..., None]) + relief(bosses, 50.0, 18.0)
    fg = fg * (1.0 - 0.6 * bores[..., None]) - relief(bores, 20.0, 0.0)
    # Faint ejector pin marks (thin circles).
    for _ in range(int(rng.integers(3, 7))):
        ex, ey = cx + rng.uniform(-0.9, 0.9) * hw, cy + rng.uniform(-0.9, 0.9) * hh
        er = rng.uniform(4, 7) * k + 1.5
        ring = np.clip(_disk(h, w, ex, ey, er) - _disk(h, w, ex, ey, er - 1.0), 0, 1)
        fg = fg * (1.0 - 0.15 * ring[..., None])
    rim = np.clip(outline - cv2.erode(outline, np.ones((3, 3), np.uint8),
                                      iterations=max(1, int(2 * k + 1))), 0, 1)
    fg = _screen(fg, rim, 0.25)
    return fg, outline, np.zeros((h, w), np.float32), holes, (cx, cy)


def _render_machined(rng, h, w, k):
    """Turned flange: turning marks, bow-tie highlight, hub, grooves, bore + keyway, bolt circle."""
    cx, cy = w / 2.0, h / 2.0
    R = rng.uniform(0.43, 0.48) * min(h, w)
    outline = _disk(h, w, cx, cy, R)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.hypot(xx - cx, yy - cy)
    th = np.arctan2(yy - cy, xx - cx)

    v = rng.uniform(150, 192)
    marks = cv2.GaussianBlur(rng.standard_normal((1, int(4 * R) + 16)).astype(np.float32),
                             (0, 0), 0.8).ravel()
    marks = marks / max(float(marks.std()), 1e-6)
    turning = np.interp(rr.ravel() * 4.0, np.arange(len(marks)), marks).reshape(h, w)
    bowtie = 1.0 + rng.uniform(0.08, 0.15) * np.cos(2.0 * (th - rng.uniform(0, math.pi)))
    hub = _disk(h, w, cx, cy, rng.uniform(0.45, 0.55) * R)
    shade = bowtie * (1.0 + rng.uniform(-0.12, -0.05) * hub)
    fg = (v * shade + rng.uniform(3, 5) * turning)[..., None] * np.array([1.02, 1.0, 0.97], np.float32)
    fg = fg * (1.0 + 0.3 * _emboss(hub, 1.5 * k + 0.5))[..., None]

    # Concentric grooves on the outer face.
    for gr in rng.uniform(0.86, 0.95, int(rng.integers(1, 3))) * R:
        groove = np.clip(1.0 - np.abs(rr - gr) / (1.5 * k + 0.6), 0, 1)
        fg = fg * (1.0 - 0.45 * groove[..., None])

    holes, rings = [], np.zeros((h, w), np.float32)
    # Central bore with a keyway (breaks the rotational symmetry).
    rb = rng.uniform(0.14, 0.2) * R
    ka = rng.uniform(0, 2 * math.pi)
    key = _full(h, w, cx + rb * math.cos(ka), cy + rb * math.sin(ka), 0.35 * rb, 0.18 * rb, 1.0,
                math.degrees(ka))
    bore, ring = _hole(h, w, cx, cy, rb, k)
    hole_cov = np.maximum(bore, key)
    rings = np.maximum(rings, ring * (1.0 - key))
    holes.append((cx, cy, rb))
    # Bolt circle.
    n = int(rng.choice([4, 6, 8]))
    pcd = rng.uniform(0.7, 0.76) * R
    rbolt = rng.uniform(0.05, 0.07) * R
    a0 = rng.uniform(0, 2 * math.pi)
    for i in range(n):
        a = a0 + 2 * math.pi * i / n
        hx, hy = cx + pcd * math.cos(a), cy + pcd * math.sin(a)
        cov, ring = _hole(h, w, hx, hy, rbolt, k)
        hole_cov, rings = np.maximum(hole_cov, cov), np.maximum(rings, ring)
        holes.append((hx, hy, rbolt))
    fg = _screen(fg, rings, 0.5)
    chamfer = np.clip(outline - _disk(h, w, cx, cy, R - 3.0 * k - 1), 0, 1)
    fg = _screen(fg, chamfer, 0.5)
    return fg, outline, hole_cov, holes, (cx, cy)


_PART_RENDERERS = {
    "stamped": _render_stamped,
    "painted": _render_painted,
    "molded": _render_molded,
    "machined": _render_machined,
}


def make_part_layers(rng: np.random.Generator, kind: str | None = None,
                     h: int = 720, w: int = 1280) -> dict:
    """Render a part and its scene as separate layers (to move the part over a static background).

    Returns a dict:
    ``bg`` float32[H,W,3] background; ``fg`` float32[H,W,3] part colour (defined
    everywhere); ``alpha`` float32[H,W] part coverage (0 in through holes);
    ``outline`` float32[H,W] coverage with holes filled; ``shadow`` float32[H,W]
    drop-shadow darkness on the background; ``vignette`` float32[H,W] static
    lens/lamp falloff; ``holes`` list of ``(x, y, r)`` (through holes, bores);
    ``center`` (x, y); ``kind``; ``background``. ``compose_layers(layers)`` renders
    the scene and ``make_part`` returns exactly that as uint8.
    """
    if kind is None:
        kind = PART_KINDS[int(rng.integers(len(PART_KINDS)))]
    if kind not in _PART_RENDERERS:
        raise ValueError(f"unknown part kind: {kind!r}")
    k = _scale(h, w)
    fg, outline, hole_cov, holes, (cx, cy) = _PART_RENDERERS[kind](rng, h, w, k)
    alpha = outline * (1.0 - hole_cov)

    if kind == "painted":
        bright = float(np.mean(fg[int(cy), int(cx)]))
        bg_choices = ("belt", "esd_mat") if bright > 110 else ("steel_table", "esd_mat")
    else:
        bg_choices = _BG_FOR_PART[kind]
    bg_kind = bg_choices[int(rng.integers(len(bg_choices)))]
    bg = _make_background(rng, h, w, bg_kind)

    # Placement: small rotation and offset (the operator never lays the part perfectly).
    angle = rng.uniform(-30, 30) if kind == "machined" else rng.uniform(-4, 4)
    tx, ty = rng.uniform(-0.025, 0.025) * w, rng.uniform(-0.02, 0.02) * h
    A = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    A[:, 2] += (tx, ty)
    fg = cv2.warpAffine(fg.astype(np.float32), A, (w, h), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
    alpha = cv2.warpAffine(alpha.astype(np.float32), A, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    outline = cv2.warpAffine(outline.astype(np.float32), A, (w, h), flags=cv2.INTER_LINEAR,
                             borderValue=0)
    placed = [(float(A[0] @ (x, y, 1.0)), float(A[1] @ (x, y, 1.0)), float(r)) for x, y, r in holes]
    holes = [(x, y, r) for x, y, r in placed if r <= x < w - r and r <= y < h - r]
    center = (float(A[0] @ (cx, cy, 1.0)), float(A[1] @ (cx, cy, 1.0)))

    # Soft drop shadow (light from the top-left); it moves with the part.
    M = np.float32([[1, 0, rng.uniform(4, 10) * k], [0, 1, rng.uniform(6, 12) * k]])
    shadow = cv2.warpAffine(alpha, M, (w, h), borderValue=0)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 6.0 * k + 0.5) * np.float32(rng.uniform(0.3, 0.5))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    vignette = (1.0 - rng.uniform(0.04, 0.12) * r2 / 2.0).astype(np.float32)
    return {
        "bg": bg.astype(np.float32), "fg": fg, "alpha": np.clip(alpha, 0, 1),
        "outline": np.clip(outline, 0, 1), "shadow": shadow.astype(np.float32),
        "vignette": vignette, "holes": holes, "center": center,
        "kind": kind, "background": bg_kind,
    }


def compose_layers(layers: dict, fg=None, alpha=None, shadow=None) -> np.ndarray:
    """Composite background, drop shadow and part (float32 BGR, not clipped).

    ``fg``/``alpha``/``shadow`` override the stored layers (used for a moved part).
    """
    fg = layers["fg"] if fg is None else fg
    alpha = layers["alpha"] if alpha is None else alpha
    shadow = layers["shadow"] if shadow is None else shadow
    a = alpha[..., None]
    img = layers["bg"] * ((1.0 - shadow) * (1.0 - alpha))[..., None]
    img += fg * a
    img *= layers["vignette"][..., None]
    return img


def make_part(rng: np.random.Generator, kind: str | None = None,
              h: int = 720, w: int = 1280) -> np.ndarray:
    """Procedural industrial part on a conveyor belt / workbench (BGR uint8).

    kinds: ``"stamped"`` (brushed steel plate, bend, ribs, holes, slot),
    ``"painted"`` (painted panel piece, seam, crease, orange peel; low texture),
    ``"molded"`` (dark plastic, ribs, bosses, grain; low contrast),
    ``"machined"`` (turned flange, grooves, bore with keyway, bolt circle).
    ``None`` picks one at random. The part's bounding box covers ~40-70 % of the
    frame. No sensor noise is added (``make_station_pair`` does that).
    """
    return _to_u8(compose_layers(make_part_layers(rng, kind, h, w)))


# --------------------------------------------------------------------------- #
# Station defects (same contract as add_scratch & co.)
# --------------------------------------------------------------------------- #

def _gray_u8(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.astype(np.uint8)


def _roi(h, w, cx, cy, ext):
    """Slices of the square of half-size ``ext`` around (cx, cy), clipped to the image."""
    x0, x1 = max(0, int(cx - ext)), min(w, int(cx + ext) + 2)
    y0, y1 = max(0, int(cy - ext)), min(h, int(cy + ext) + 2)
    return slice(y0, y1), slice(x0, x1)


def add_dent(img: np.ndarray, rng: np.random.Generator, region=None):
    """Shallow dent: directional shading of a round depression (dark + bright crescent)."""
    h, w = img.shape[:2]
    k = _scale(h, w)
    cx, cy = _pick_center_in(rng, h, w, region)
    r = rng.uniform(12, 32) * k + 4
    sigma = 0.42 * r
    light = rng.uniform(0, 2 * math.pi)
    amp_mul, amp_add = rng.uniform(0.3, 0.5), rng.uniform(22, 36)
    sl = _roi(h, w, cx, cy, r + 1)
    ys, xs = np.mgrid[sl].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    d = np.hypot(dx, dy)
    # Slope of a Gaussian depression along the light direction, faded to 0 at radius r.
    slope = (dx * math.cos(light) + dy * math.sin(light)) / sigma * np.exp(-d ** 2 / (2 * sigma ** 2))
    fade = np.clip((r - d) / (0.3 * r), 0.0, 1.0)
    s = (slope * fade / math.exp(-0.5))[..., None]  # peak about +-1
    out = img.astype(np.float32)
    out[sl] = out[sl] * (1.0 + amp_mul * s) + amp_add * s
    mask = np.zeros((h, w), bool)
    mask[sl] = d < r
    return _to_u8(out), mask


def add_missing_hole(img: np.ndarray, rng: np.random.Generator, holes=None, region=None):
    """Missing feature: a hole (or, without ``holes``, a strong local feature) is filled in.

    ``holes``: list of ``(x, y, r)`` (e.g. ``make_part_layers(...)["holes"]``). Without
    it the strongest corners (inside ``region`` if given, else the central 70 %) are
    candidates. The disk is inpainted from its surroundings plus a little texture.
    """
    h, w = img.shape[:2]
    k = _scale(h, w)
    if holes:
        hx, hy, hr = holes[int(rng.integers(len(holes)))]
        rr = hr * 1.3 + 2.5 * k + 2  # hole + its cut-edge ring
    else:
        cand = np.zeros((h, w), np.uint8)
        if region is not None:
            cand[region] = 255
        else:
            cand[int(0.15 * h):int(0.85 * h), int(0.15 * w):int(0.85 * w)] = 255
        pts = cv2.goodFeaturesToTrack(_gray_u8(img), 30, 0.05, 10, mask=cand)
        if pts is None or len(pts) == 0:
            hx, hy = _pick_center_in(rng, h, w, region)
        else:
            hx, hy = (float(v) for v in pts[int(rng.integers(len(pts))), 0])
        rr = rng.uniform(8, 16) * k + 3
    sl = _roi(h, w, hx, hy, rr + 1)
    ys, xs = np.mgrid[sl].astype(np.float32)
    mask = np.zeros((h, w), bool)
    mask[sl] = np.hypot(xs - hx, ys - hy) <= rr
    if not mask.any():
        mask[int(np.clip(hy, 0, h - 1)), int(np.clip(hx, 0, w - 1))] = True
    filled = cv2.inpaint(np.ascontiguousarray(img), mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    tex = cv2.GaussianBlur(rng.standard_normal((h, w), np.float32), (0, 0), 1.0) * np.float32(4.0)
    out = np.where(mask[..., None], filled.astype(np.float32) + tex[..., None], img.astype(np.float32))
    return _to_u8(out), mask


def add_burr(img: np.ndarray, rng: np.random.Generator, region=None):
    """Burr / flash: a small irregular bright or dark blob sitting on an edge.

    With ``region`` (part material) the blob sits on its boundary (outline or a hole
    edge); otherwise on a Canny edge in the central 70 % of the image.
    """
    h, w = img.shape[:2]
    k = _scale(h, w)
    if region is not None:
        reg = region.astype(np.uint8)
        edge = (reg > 0) & (cv2.erode(reg, np.ones((3, 3), np.uint8), iterations=2) == 0)
    else:
        edge = cv2.Canny(_gray_u8(img), 60, 150) > 0
        central = np.zeros_like(edge)
        central[int(0.15 * h):int(0.85 * h), int(0.15 * w):int(0.85 * w)] = True
        edge &= central
    ys, xs = np.nonzero(edge)
    if len(ys):
        i = int(rng.integers(len(ys)))
        cx, cy = float(xs[i]), float(ys[i])
    else:
        cx, cy = _pick_center(rng, h, w)
    r = rng.uniform(3, 7) * k + 2
    blob = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(2, 5))):
        ox, oy = rng.uniform(-r, r, 2)
        rb = r * rng.uniform(0.5, 1.0)
        sq = rng.uniform(0.5, 1.0)
        sl, a = _rounded_box(h, w, cx + ox, cy + oy, rb, rb * sq, rb * sq, rng.uniform(0, 180))
        if sl is not None:
            blob[sl] = np.maximum(blob[sl], a)
    grey = rng.uniform(205, 245) if rng.random() < 0.5 else rng.uniform(15, 45)
    out = img.astype(np.float32)
    _over(out, (slice(0, h), slice(0, w)), blob * np.float32(rng.uniform(0.85, 1.0)), (grey,) * 3)
    mask = blob >= 0.5
    if not mask.any():
        mask[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))] = True
    return _to_u8(out), mask


def add_stain(img: np.ndarray, rng: np.random.Generator, region=None):
    """Oil stain (darker, brownish, dark rim) or dried water stain (haze, mineral ring)."""
    h, w = img.shape[:2]
    k = _scale(h, w)
    cx, cy = _pick_center_in(rng, h, w, region)
    r0 = rng.uniform(22, 60) * k + 5
    th = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    rad = np.ones_like(th)
    for m in range(2, 6):  # irregular outline from a few harmonics
        rad += rng.uniform(0, 0.12) * np.cos(m * th + rng.uniform(0, 2 * math.pi))
    pts = np.stack([cx + r0 * rad * np.cos(th), cy + r0 * rad * np.sin(th)], -1)
    layer = np.zeros((h, w), np.uint8)
    cv2.fillPoly(layer, [np.round(pts * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    a = cv2.GaussianBlur(layer.astype(np.float32) / 255.0, (0, 0), 0.8)
    inner = cv2.erode(a, np.ones((3, 3), np.uint8), iterations=max(1, int(round(3 * k + 1))))
    rim = np.clip(a - inner, 0, 1)[..., None]
    a3 = a[..., None]
    out = img.astype(np.float32)
    if rng.random() < 0.6:  # oil: darker, slightly brownish, darker rim
        keep = (1.0 - rng.uniform(0.3, 0.45)) * np.array([0.85, 0.95, 1.0], np.float32)
        out = out * (1.0 - a3 * (1.0 - keep)) * (1.0 - 0.2 * rim)
    else:  # dried water: haze towards grey + bright mineral ring
        out = out + (rng.uniform(150, 190) - out) * 0.22 * a3
        out = out + (235.0 - out) * 0.45 * rim
    return _to_u8(out), a >= 0.5


# --------------------------------------------------------------------------- #
# Factory lighting
# --------------------------------------------------------------------------- #

def apply_factory_lighting(img: np.ndarray, rng: np.random.Generator, kind: str,
                           strength: float = 1.0) -> np.ndarray:
    """Shop-floor illumination change (no light box); returns a new BGR uint8 image.

    Monotone kinds: every channel goes through a non-decreasing map, so the local
    intensity order is kept (clipping at 0/255 and rounding only create ties):
    - ``"drift_gain"``: global gain 1 +- 0.15..0.35;
    - ``"gamma"``: gamma 0.7..0.9 or 1.1..1.4;
    - ``"auto_exposure"``: camera AE pulls the mean towards a target (gain >= 10 %),
      black-level offset and a mild S / anti-S tone curve;
    - ``"lamp_aging"``: dimmer lamp (gain 0.65..0.88) + warm or greenish colour cast;
    - ``"window_gradient"``: smooth spatial gain 0.6..1.0 from the window side with a
      slightly bluer daylight tint (monotone within small windows only).
    Non-monotone kinds (expected failure cases, report them as such):
    - ``"specular"``: saturated elongated highlight at a random place;
    - ``"hand_shadow"``: hard-edged shadow of an arm/hand reaching in from a border.
    ``strength`` scales the deviation from "no change" (1.0 = the ranges above).
    """
    h, w = img.shape[:2]
    k = _scale(h, w)
    s = float(strength)
    x = img.astype(np.float32)
    if kind == "drift_gain":
        d = rng.uniform(0.15, 0.35) * (1.0 if rng.random() < 0.5 else -1.0)
        out = x * np.float32(1.0 + s * d)
    elif kind == "gamma":
        g = rng.uniform(0.7, 0.9) if rng.random() < 0.5 else rng.uniform(1.1, 1.4)
        out = 255.0 * (x / 255.0) ** np.float32(1.0 + s * (g - 1.0))
    elif kind == "auto_exposure":
        m = max(float(x.mean()), 1.0)
        gain = float(np.clip((rng.uniform(90, 170) / m) ** rng.uniform(0.6, 0.9), 0.65, 1.6))
        if abs(gain - 1.0) < 0.1:  # the camera did re-expose
            gain = 1.0 + math.copysign(0.1, gain - 1.0)
        gain = gain ** s
        offset = rng.uniform(-10, 10) * s
        c = rng.uniform(-0.25, 0.25) * min(s, 1.0)  # |c| < 1 keeps the curve increasing
        y = np.clip(x * np.float32(gain) + np.float32(offset), 0.0, 255.0)
        out = y + np.float32(c) * y * (1.0 - y / 255.0)
    elif kind == "lamp_aging":
        gain = 1.0 - s * rng.uniform(0.12, 0.35)
        if rng.random() < 0.6:  # warm: B down, R up
            cast = np.array([rng.uniform(0.86, 0.94), rng.uniform(0.97, 1.0), rng.uniform(1.03, 1.08)])
        else:  # greenish fluorescent
            cast = np.array([rng.uniform(0.92, 0.97), rng.uniform(1.02, 1.06), rng.uniform(0.9, 0.96)])
        out = x * (gain * (1.0 + s * (cast - 1.0))).astype(np.float32)
    elif kind == "window_gradient":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = rng.uniform(0, 2 * math.pi)
        proj = xx * math.cos(t) + yy * math.sin(t)
        proj = (proj - proj.min()) / max(float(np.ptp(proj)), 1.0)
        ramp = 0.5 * (1.0 + np.tanh((proj - rng.uniform(0.3, 0.7)) / rng.uniform(0.3, 0.5)))
        ramp = (ramp - ramp.min()) / max(float(np.ptp(ramp)), 1e-6)  # 1 on the window side
        gmin = rng.uniform(0.6, 0.75)
        gain = gmin + (1.0 - gmin) * ramp
        gain = np.clip(gain * (1.0 + 0.02 * _lowfreq(rng, h, w, 3, 5)), 0.55, 1.0)
        gain = 1.0 - s * (1.0 - gain)
        tint = 1.0 + s * (np.array([1.04, 1.0, 0.97], np.float32) - 1.0)
        out = x * gain[..., None] * (1.0 + (tint - 1.0) * ramp[..., None])
    elif kind == "specular":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        a = rng.uniform(0, math.pi)
        u = (xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)
        v = -(xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)
        su, sv = rng.uniform(0.08, 0.2) * w, rng.uniform(0.012, 0.035) * min(h, w) + 1
        core = np.exp(-0.5 * ((u / su) ** 2 + (v / sv) ** 2))
        glow = np.exp(-0.5 * ((u / (1.5 * su)) ** 2 + (v / (4 * sv)) ** 2))
        out = x + (s * (rng.uniform(280, 420) * core + 35.0 * glow))[..., None]
    elif kind == "hand_shadow":
        layer = _hand_layer(rng, h, w)
        layer = cv2.GaussianBlur(layer, (0, 0), rng.uniform(1.0, 2.5) * k + 0.3)
        out = x * (1.0 - s * rng.uniform(0.4, 0.6) * layer)[..., None]
    else:
        raise ValueError(f"unknown factory lighting kind: {kind!r}")
    return _to_u8(out)


def _hand_layer(rng, h, w) -> np.ndarray:
    """Silhouette (0..1) of a forearm, palm and fingers reaching in from a random border."""
    side = int(rng.integers(4))
    along = rng.uniform(0.25, 0.75)
    (sx, sy), (dx, dy) = [((along * w, 0.0), (0.0, 1.0)), ((along * w, float(h)), (0.0, -1.0)),
                          ((0.0, along * h), (1.0, 0.0)), ((float(w), along * h), (-1.0, 0.0))][side]
    tilt = rng.uniform(-0.4, 0.4)
    dx, dy = dx * math.cos(tilt) - dy * math.sin(tilt), dx * math.sin(tilt) + dy * math.cos(tilt)
    nx, ny = -dy, dx
    depth = rng.uniform(0.25, 0.45) * (h if side < 2 else w)
    aw = rng.uniform(0.09, 0.13) * min(h, w)  # forearm half width
    ang = math.degrees(math.atan2(dy, dx))
    layer = np.zeros((h, w), np.float32)

    def blob(cx, cy, half_len, half_wid, radius, a_deg):
        sl, a = _rounded_box(h, w, cx, cy, half_len, half_wid, radius, a_deg)
        if sl is not None:
            layer[sl] = np.maximum(layer[sl], a)

    # Forearm from outside the frame to the wrist.
    wx, wy = sx + dx * depth, sy + dy * depth
    blob((sx + wx) / 2 - dx * aw, (sy + wy) / 2 - dy * aw, depth / 2 + aw, aw, aw, ang)
    # Palm.
    pl = 1.1 * aw
    px, py = wx + dx * pl, wy + dy * pl
    blob(px, py, pl, 1.15 * aw, 0.5 * aw, ang)
    # Four slightly fanned fingers and a thumb.
    fw = 0.2 * aw
    for i, off in enumerate(np.linspace(-0.85, 0.85, 4)):
        fa = ang + rng.uniform(-8, 8) + 6 * off
        fl = pl * rng.uniform(0.7, 1.0) * (1.0 if 0 < i < 3 else 0.85)
        bx, by = px + dx * pl * 0.8 + nx * off * aw, py + dy * pl * 0.8 + ny * off * aw
        ra = math.radians(fa)
        blob(bx + math.cos(ra) * fl, by + math.sin(ra) * fl, fl, fw, fw, fa)
    sgn = 1.0 if rng.random() < 0.5 else -1.0
    ta = ang + sgn * rng.uniform(40, 60)
    ra = math.radians(ta)
    bx, by = wx + dx * 0.6 * pl + nx * sgn * aw, wy + dy * 0.6 * pl + ny * sgn * aw
    blob(bx + math.cos(ra) * 0.6 * pl, by + math.sin(ra) * 0.6 * pl, 0.7 * pl, 1.2 * fw, 1.2 * fw, ta)
    return layer


# --------------------------------------------------------------------------- #
# Station pairs
# --------------------------------------------------------------------------- #

_STATION_DEFECT_FNS = {
    "scratch": add_scratch,
    "tape": add_tape,
    "sticker": add_sticker,
    "dent": add_dent,
    "missing_hole": add_missing_hole,
    "burr": add_burr,
    "stain": add_stain,
}
# Defects that live on the part surface: clipped to the part's material.
_SURFACE_DEFECTS = ("scratch", "dent", "stain")


def _similarity(cx, cy, angle_deg, tx, ty) -> np.ndarray:
    """3x3 rotation about (cx, cy) followed by a translation (tx, ty)."""
    H = np.eye(3)
    H[:2] = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle_deg), 1.0)
    H[0, 2] += tx
    H[1, 2] += ty
    return H


def _station_defect(name, img, rng, layers):
    """Apply a station defect on the part; returns ``(img, mask)``."""
    h, w = img.shape[:2]
    material = layers["alpha"] > 0.5
    er = max(1, int(round(20 * _scale(h, w))))
    inner = cv2.erode(material.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=er) > 0
    if not inner.any():
        inner = material
    fn = _STATION_DEFECT_FNS[name]
    for _ in range(5):  # retry if clipping removed the defect entirely (rare)
        if name == "missing_hole":
            out, mask = fn(img, rng, holes=layers["holes"], region=inner)
        elif name == "burr":
            out, mask = fn(img, rng, region=material)
        else:
            out, mask = fn(img, rng, region=inner)
        if name in _SURFACE_DEFECTS:
            out = np.where(material[..., None], out, img)
            mask = mask & material
        if mask.any():
            break
    return out, mask


def make_station_pair(rng: np.random.Generator, part_kind: str | None = None,
                      defect: str | None = None, lighting: str | None = None,
                      motion: str = "none", h: int = 720, w: int = 1280,
                      strength: float = 1.0) -> dict:
    """Synthetic (reference, frame) pair seen by a fixed camera over a part.

    Steps: part scene -> (frame only) defect on the part -> motion -> factory
    lighting on the frame (scaled by ``strength``) -> independent sensor noise
    (sigma 1.5-3.5) on both images.

    ``motion``:
    - ``"none"``: same pose;
    - ``"jitter"``: camera vibration; the WHOLE image moves by up to 2 px / 0.3 deg
      (pixel amounts scale with the short side relative to 720 px);
    - ``"shift"``: the PART is moved on the belt by up to 40 px (same scaling) and
      3 deg while camera and background stay put: the part, its defect and its drop
      shadow are re-composited at the new pose over the static background.

    Returns ``dict(reference, frame, mask_ref_frame, meta)``; ``mask_ref_frame`` is
    the defect mask in the REFERENCE frame (empty without a defect). ``meta``:
    ``H`` (3x3, REFERENCE -> FRAME pixel coords of the part; identity for "none"),
    ``motion_scope`` ("none" | "camera" | "part"; with "part" the background does
    NOT follow H, so compare inside ``part_mask`` only), ``mask_frame`` (defect mask
    mapped into the frame), ``part_mask`` (bool, part material in the reference),
    ``part_kind``, ``background``, ``defect``, ``lighting``, ``lighting_monotone``
    (None without lighting), ``motion``, ``strength``, ``noise_sigma`` (reference,
    frame), ``size``, ``holes``, ``reference_clean`` (noise-free reference, e.g.
    for ``make_reference_set``).
    """
    if part_kind is not None and part_kind not in PART_KINDS:
        raise ValueError(f"unknown part kind: {part_kind!r}")
    if defect is not None and defect not in STATION_DEFECTS:
        raise ValueError(f"unknown station defect: {defect!r}")
    if lighting is not None and lighting not in FACTORY_LIGHTING:
        raise ValueError(f"unknown factory lighting kind: {lighting!r}")
    if motion not in MOTION_MODES:
        raise ValueError(f"unknown motion mode: {motion!r}")

    k = _scale(h, w)
    layers = make_part_layers(rng, part_kind, h, w)
    ref = _to_u8(compose_layers(layers))
    scene, mask = ref, np.zeros((h, w), bool)
    if defect is not None:
        scene, mask = _station_defect(defect, ref, rng, layers)

    H = np.eye(3)
    scope = "none"
    if motion == "none":
        frame = scene
    elif motion == "jitter":
        scope = "camera"
        H = _similarity((w - 1) / 2.0, (h - 1) / 2.0, rng.uniform(-0.3, 0.3),
                        *(rng.uniform(-2.0, 2.0, 2) * k))
        frame = warp(scene, H)
    else:
        scope = "part"
        pcx, pcy = layers["center"]
        ang = rng.uniform(-3.0, 3.0)
        mag, dirn = rng.uniform(0.3, 1.0) * 40.0 * k, rng.uniform(0, 2 * math.pi)
        t = np.array([mag * math.cos(dirn), mag * math.sin(dirn)])
        # Keep the moved part inside the frame where possible (halve the shift).
        ys, xs = np.nonzero(layers["outline"] > 0.5)
        corners = np.array([[xs.min(), ys.min(), 1], [xs.max(), ys.min(), 1],
                            [xs.min(), ys.max(), 1], [xs.max(), ys.max(), 1]], np.float64).T
        for _ in range(4):
            H = _similarity(pcx, pcy, ang, *t)
            c = H[:2] @ corners
            if c[0].min() >= 0 and c[1].min() >= 0 and c[0].max() <= w - 1 and c[1].max() <= h - 1:
                break
            t = t * 0.5
        # Moving layer = the part + whatever the defect touched (tape / burr may overhang).
        touched = (np.abs(scene.astype(np.int16) - ref.astype(np.int16)).max(-1) > 0) | mask
        touched = cv2.dilate(touched.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        unvig = scene.astype(np.float32) / layers["vignette"][..., None]
        fg = np.where(touched[..., None], unvig, layers["fg"]).astype(np.float32)
        alpha = np.where(touched, 1.0, layers["alpha"]).astype(np.float32)
        A = H[:2]
        fg = cv2.warpAffine(fg, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        alpha = cv2.warpAffine(alpha, A, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        shadow = cv2.warpAffine(layers["shadow"], A, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        frame = _to_u8(compose_layers(layers, fg=fg, alpha=alpha, shadow=shadow))

    if lighting is not None:
        frame = apply_factory_lighting(frame, rng, lighting, strength)
    sigma_r, sigma_f = rng.uniform(1.5, 3.5, 2)
    reference = _add_noise(ref, rng, sigma_r)
    frame = _add_noise(frame, rng, sigma_f)

    mask_frame = cv2.warpPerspective(mask.astype(np.uint8), H, (w, h), flags=cv2.INTER_NEAREST) > 0
    meta = {
        "part_kind": layers["kind"],
        "background": layers["background"],
        "defect": defect,
        "lighting": lighting,
        "lighting_monotone": None if lighting is None else lighting in MONOTONE_LIGHTING,
        "motion": motion,
        "motion_scope": scope,
        "H": H,
        "strength": float(strength),
        "noise_sigma": (float(sigma_r), float(sigma_f)),
        "size": (h, w),
        "holes": list(layers["holes"]),
        "part_mask": layers["alpha"] > 0.5,
        "mask_frame": mask_frame,
        "reference_clean": ref,
    }
    return {"reference": reference, "frame": frame, "mask_ref_frame": mask, "meta": meta}


def make_reference_set(rng: np.random.Generator, base: np.ndarray, n: int,
                       max_shift_px: float = 1.0, max_rot_deg: float = 0.15,
                       noise_sigma: tuple[float, float] = (1.5, 3.0)) -> list:
    """``n`` good shots of the same scene: tiny camera jitter + independent sensor noise.

    ``base`` should be a noise-free scene (``meta["reference_clean"]`` or
    ``make_part(...)``). Shot 0 is not moved (noise only), so it stays registered
    with ``base``; shots 1.. move by up to ``max_shift_px`` (scaled with the short
    side relative to 720 px) and ``max_rot_deg`` about the image centre.
    """
    h, w = base.shape[:2]
    k = _scale(h, w)
    shots = []
    for i in range(int(n)):
        img = base
        if i > 0:
            H = _similarity((w - 1) / 2.0, (h - 1) / 2.0, rng.uniform(-max_rot_deg, max_rot_deg),
                            *(rng.uniform(-max_shift_px, max_shift_px, 2) * k))
            img = warp(base, H)
        shots.append(_add_noise(img, rng, rng.uniform(*noise_sigma)))
    return shots


# --------------------------------------------------------------------------- #
# Mode 1: uniform surfaces filling the whole frame (single-photo surface check)
# --------------------------------------------------------------------------- #

SURFACE_KINDS = ("painted", "plastic", "brushed", "glass_dark")
SURFACE_CHECK_DEFECTS = ("scratch", "chip", "stain", "sticker", "dent", "burr")


def _soft_falloff(rng, h, w, depth) -> np.ndarray:
    """Smooth illumination gain in [1 - depth, 1]: off-centre radial fall-off + faint low-freq."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    r = np.hypot((xx - cx) / w, (yy - cy) / h)
    r = r / max(float(r.max()), 1e-6)
    gain = 1.0 - depth * r ** 2
    gain = gain * (1.0 + 0.015 * _lowfreq(rng, h, w, 3, 4))
    return np.clip(gain, 0.5, 1.05).astype(np.float32)


def make_uniform_surface(rng: np.random.Generator, kind: str = "painted",
                         h: int = 720, w: int = 1280) -> np.ndarray:
    """Close-up photo of a uniform surface filling the WHOLE frame (BGR uint8).

    Kinds (``SURFACE_KINDS``):
    - ``"painted"``: smooth car paint colour + orange peel + fine metallic flake;
    - ``"plastic"``: mid/dark moulded plastic with fine isotropic grain;
    - ``"brushed"``: directional brushed metal (streaks along x or y);
    - ``"glass_dark"``: dark glossy surface with faint broad reflections.
    All include a mild, soft illumination fall-off (no light box) and sensor noise.
    Deterministic for a given rng state.
    """
    if kind not in SURFACE_KINDS:
        raise ValueError(f"unknown surface kind: {kind!r}")
    k = _scale(h, w)
    if kind == "painted":
        base = np.asarray(_CAR_COLOURS_BGR[rng.integers(len(_CAR_COLOURS_BGR))], np.float32)
        # Exposed for the paint: dark colours are lifted (luma >= ~100) and a white panel
        # is shot so it does not clip (channels <= 175).
        base = base * rng.uniform(0.94, 1.06, 3)
        luma = float(base @ np.array([0.114, 0.587, 0.299]))
        base = np.clip(base * max(1.0, 100.0 / max(luma, 1.0)), 20, 175).astype(np.float32)
        img = np.broadcast_to(base, (h, w, 3)).astype(np.float32)
        img = img * _soft_falloff(rng, h, w, rng.uniform(0.08, 0.18))[..., None]
        img = img + _grain(rng, h, w, 1.6 * k + 0.4, rng.uniform(2.0, 3.5))[..., None]  # orange peel
        img = img + _grain(rng, h, w, 0.5, rng.uniform(0.8, 1.5))[..., None]            # flake
    elif kind == "plastic":
        v = rng.uniform(85, 130)
        tint = np.array([1.0, rng.uniform(0.97, 1.03), rng.uniform(0.96, 1.04)], np.float32)
        img = np.full((h, w, 3), v, np.float32) * tint
        img = img * _soft_falloff(rng, h, w, rng.uniform(0.08, 0.2))[..., None]
        img = img + _grain(rng, h, w, 0.9 * k + 0.3, rng.uniform(4.0, 7.0))[..., None]
    elif kind == "brushed":
        v = rng.uniform(120, 175)
        vertical = rng.random() < 0.5
        hh, ww = (w, h) if vertical else (h, w)
        tex = _brushed(rng, hh, ww, rng.uniform(40, 90) * k + 4, rng.uniform(6.0, 10.0))
        tex = tex + _grain(rng, hh, ww, 0.6, 1.5)
        if vertical:
            tex = np.ascontiguousarray(tex.T)
        img = np.full((h, w, 3), v, np.float32) * np.array([1.02, 1.0, 0.98], np.float32)
        img = img * _soft_falloff(rng, h, w, rng.uniform(0.08, 0.2))[..., None] + tex[..., None]
    else:  # glass_dark
        v = rng.uniform(82, 96)
        img = np.full((h, w, 3), v, np.float32) * np.array([1.04, 1.0, 0.97], np.float32)
        img = img * _soft_falloff(rng, h, w, rng.uniform(0.06, 0.14))[..., None]
        # Faint broad reflections (window / ceiling light): very soft bands.
        refl = np.zeros((h, w), np.float32)
        for _ in range(int(rng.integers(1, 3))):
            band = _full(h, w, rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h,
                         rng.uniform(0.3, 0.6) * w, rng.uniform(0.04, 0.1) * h, 1.0,
                         rng.uniform(-30, 30))
            refl += _smooth(band, rng.uniform(40, 70) * k + 2) * np.float32(rng.uniform(8, 16))
        img = img + refl[..., None]
        img = img + _grain(rng, h, w, 0.8, 0.8)[..., None]
    img = img + np.float32(1.2) * rng.standard_normal((h, w, 3), np.float32)  # sensor noise
    return _to_u8(img)


def add_chip(img: np.ndarray, rng: np.random.Generator, region=None):
    """Small paint chip: irregular flake of paint missing, exposing the layer below.

    The exposed layer contrasts with the local surface (as a visible chip does): light
    primer or bare metal on a dark/mid surface, dark e-coat on a light one. A thin
    dark shadow rim sits on one side (the paint layer edge); 3-12 px radius at 720 px.
    """
    h, w = img.shape[:2]
    k = _scale(h, w)
    cx, cy = _pick_center_in(rng, h, w, region)
    r0 = rng.uniform(4, 12) * k + 2
    th = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    rad = np.ones_like(th)
    for m in range(2, 7):
        rad += rng.uniform(0, 0.18) * np.cos(m * th + rng.uniform(0, 2 * math.pi))
    rad = np.clip(rad, 0.45, None)
    pts = np.stack([cx + r0 * rad * np.cos(th), cy + r0 * rad * np.sin(th)], -1)
    layer = np.zeros((h, w), np.uint8)
    cv2.fillPoly(layer, [np.round(pts * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    a = cv2.GaussianBlur(layer.astype(np.float32) / 255.0, (0, 0), 0.6)
    # Shadow rim: the coverage shifted towards a random side, minus the chip itself.
    ang = rng.uniform(0, 2 * math.pi)
    sx, sy = 1.5 * k * math.cos(ang) + 0.5, 1.5 * k * math.sin(ang) + 0.5
    shifted = cv2.warpAffine(a, np.float32([[1, 0, sx], [0, 1, sy]]), (w, h))
    rim = np.clip(shifted - a, 0, 1)
    out = img.astype(np.float32)
    local = float(_gray_u8(img)[_roi(h, w, cx, cy, 2 * r0 + 2)].mean())
    if local < 125:
        if rng.random() < 0.65:  # primer: light grey
            colour = np.full(3, rng.uniform(max(170.0, local + 70), 225), np.float32)
        else:  # bare metal: bright grey, slightly bluish
            colour = rng.uniform(max(160.0, local + 60), 210) * np.array([1.05, 1.0, 0.96], np.float32)
    else:  # dark e-coat / primer under a light paint
        colour = np.full(3, rng.uniform(25, min(70.0, local - 60)), np.float32)
    tex = cv2.GaussianBlur(rng.standard_normal((h, w), np.float32), (0, 0), 0.7) * np.float32(5.0)
    out = out * (1.0 - a[..., None]) + (colour[None, None, :] + tex[..., None]) * a[..., None]
    out = out * (1.0 - 0.45 * rim[..., None])
    mask = (a >= 0.5) | (rim >= 0.5)
    if not mask.any():
        mask[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))] = True
    return _to_u8(out), mask
