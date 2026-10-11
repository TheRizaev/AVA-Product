"""Surface-check (Mode 1) demo examples from real CC0 texture photos (Poly Haven).

Each candidate is a PAIR rendered from the same real texture scan:
  <name>_DEFECT.jpg  real texture photo (CC0, Poly Haven) + one synthetic defect
  <name>_OK.jpg      the same photo, no defect

How an image is made (deterministic, no randomness outside fixed seeds):
  1. crop a 16:9 window of the Poly Haven diffuse (albedo) and OpenGL normal map,
     optionally colour-graded (stated per candidate);
  2. the defect (if any) is painted into the albedo / a height field / the specular
     term at 2x resolution, so it is anti-aliased and lit by the same light as the
     texture relief (the normal map);
  3. "photo" rendering: Lambert + weak Blinn-Phong under one fixed light, a soft
     large-scale illumination fall-off (no hot spot), 2x -> 1x area downsampling,
     slight optical blur, mild sensor noise; 1280x720 JPEG q92.

Every candidate is evaluated with the REAL pipeline (`<pkg>.surface.scan_surface`
with default `Params()`), plus robustness variants (JPEG q80 re-encode; upscale to
1600 px wide and back). Results -> candidates.json + contact_sheet.png.

Assets are downloaded by fetch_assets.py into ./assets/<id>/ (info.json holds the
Poly Haven metadata: name, authors, URL, licence CC0).

Usage:  python make_surface_examples.py [--only name1,name2] [--no-eval]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Users\Asus\Desktop\AVA Pr"
ASSETS = os.path.join(HERE, "assets")
OUT_DIR = os.path.join(HERE, "out")
OUT_W, OUT_H = 1280, 720
SS = 2                       # supersampling factor of the renderer
JPEG_Q = 92
MARGIN_DEFECT = 1.35         # defect max score >= this x threshold
MARGIN_CLEAN = 0.75          # clean max score <= this x threshold

sys.path.insert(0, REPO)
try:
    import ava as pkg        # noqa: E402
except ImportError:          # package not renamed yet
    import dalil as pkg      # noqa: E402
import importlib             # noqa: E402
surface = importlib.import_module(pkg.__name__ + ".surface")
config = importlib.import_module(pkg.__name__ + ".config")


# =========================================================================== assets

def asset_info(aid: str) -> dict:
    with open(os.path.join(ASSETS, aid, "info.json"), encoding="utf-8") as f:
        return json.load(f)


def _find(aid: str, keys) -> str:
    for k in keys:
        hit = sorted(glob.glob(os.path.join(ASSETS, aid, f"*{k}*.jpg")))
        if hit:
            return hit[0]
    raise FileNotFoundError(f"{aid}: none of {keys}")


def load_maps(aid: str):
    """(albedo float32 sRGB 0..1 BGR, normal float32 [-1,1] xyz (x right, y up, z out))."""
    diff = cv2.imread(_find(aid, ("_diff_", "_albedo_", "_coll1_", "_col1_")), cv2.IMREAD_COLOR)
    nor = cv2.imread(_find(aid, ("_nor_gl_",)), cv2.IMREAD_COLOR)
    if nor.shape[:2] != diff.shape[:2]:
        nor = cv2.resize(nor, (diff.shape[1], diff.shape[0]), interpolation=cv2.INTER_AREA)
    return diff.astype(np.float32) / 255.0, nor


def _wrap_crop(img: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """Crop with wrap-around (the Poly Haven maps are tileable)."""
    H, W = img.shape[:2]
    ys = (np.arange(y0, y0 + h) % H)
    xs = (np.arange(x0, x0 + w) % W)
    return img[ys][:, xs]


def crop_maps(diff, nor, x0, y0, cw, rot90=0):
    """16:9 crop of width cw source px, resampled to the supersampled render size."""
    ch = int(round(cw * 9 / 16))
    if rot90:
        diff, nor = np.rot90(diff, rot90), np.rot90(nor, rot90)
    d = _wrap_crop(diff, x0, y0, cw, ch)
    n = _wrap_crop(nor, x0, y0, cw, ch)
    size = (OUT_W * SS, OUT_H * SS)
    interp = cv2.INTER_AREA if cw > size[0] else cv2.INTER_CUBIC
    d = cv2.resize(np.ascontiguousarray(d), size, interpolation=interp)
    n = cv2.resize(np.ascontiguousarray(n), size, interpolation=interp).astype(np.float32)
    # decode the OpenGL normal map (BGR order: B = z, G = y up, R = x right)
    nx = n[..., 2] / 127.5 - 1.0
    ny = n[..., 1] / 127.5 - 1.0
    nz = n[..., 0] / 127.5 - 1.0
    if rot90:  # rotate the tangent-space vectors with the image (k x 90 deg counter-clockwise)
        for _ in range(rot90 % 4):
            nx, ny = -ny, nx
    return np.clip(d, 0, 1), np.stack([nx, ny, np.maximum(nz, 0.05)], -1)


# =========================================================================== colour

def srgb_to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055).astype(np.float32)


def grade(alb_srgb, sat=1.0, gain=(1, 1, 1), gamma=1.0):
    """Colour grade in sRGB: saturation, per-channel gain (B, G, R), gamma."""
    g = alb_srgb.mean(axis=2, keepdims=True)
    out = g + (alb_srgb - g) * sat
    out = out * np.array(gain, np.float32)[None, None, :]
    return np.clip(out, 0, 1) ** gamma


# =========================================================================== defect canvas

class Canvas:
    """Defect layers at render resolution (SS x output). Coordinates given in output px."""

    def __init__(self, shape):
        h, w = shape
        self.h, self.w = h, w
        self.height = np.zeros((h, w), np.float32)      # height field (render px units)
        self.flatten = np.zeros((h, w), np.float32)     # 0..1: remove texture relief
        self.alb_mix = np.zeros((h, w), np.float32)     # 0..1 blend weight towards alb_col
        self.alb_col = np.zeros((h, w, 3), np.float32)  # sRGB target colour (BGR)
        self.alb_mul = np.ones((h, w), np.float32)      # multiplicative albedo factor
        self.spec = np.zeros((h, w), np.float32)        # extra specular (glint), linear
        self.gloss = np.zeros((h, w), np.float32)       # extra glossiness weight 0..1
        self.mask = np.zeros((h, w), np.float32)        # ground-truth defect footprint

    def blend_colour(self, m, col):
        col = np.asarray(col, np.float32)
        if col.ndim == 1:
            col = np.broadcast_to(col, (self.h, self.w, 3))
        new = np.maximum(self.alb_mix, m)
        w = np.divide(m, new, out=np.zeros_like(m), where=new > 1e-6)[..., None]
        self.alb_col = self.alb_col * (1 - w) + col * w
        self.alb_mix = new


def _s(v):
    return float(v) * SS


def poly_mask(shape, pts_out, blur_out=0.6):
    """Anti-aliased filled polygon (points in output px)."""
    m = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(pts_out, np.float64) * SS * 16).astype(np.int32)
    cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], 255, cv2.LINE_AA, shift=4)
    m = m.astype(np.float32) / 255.0
    if blur_out > 0:
        m = cv2.GaussianBlur(m, (0, 0), _s(blur_out))
    return m


def line_mask(shape, pts_out, width_out, blur_out=0.4):
    """Anti-aliased polyline of constant width (points in output px)."""
    m = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(pts_out, np.float64) * SS * 16).astype(np.int32)
    cv2.polylines(m, [pts.reshape(-1, 1, 2)], False, 255, max(1, int(round(_s(width_out)))),
                  cv2.LINE_AA, shift=4)
    m = m.astype(np.float32) / 255.0
    if blur_out > 0:
        m = cv2.GaussianBlur(m, (0, 0), _s(blur_out))
    return m


def blob_outline(rng, cx, cy, r, irregular=0.25, n=48, aspect=1.0, angle=0.0, harmonics=6):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = np.ones_like(th)
    for k in range(2, 2 + harmonics):
        rr += irregular / k ** 0.8 * rng.uniform(-1, 1) * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    rr *= r
    x, y = rr * np.cos(th) * aspect, rr * np.sin(th)
    ca, sa = np.cos(angle), np.sin(angle)
    return np.stack([cx + ca * x - sa * y, cy + sa * x + ca * y], 1)


def curve(rng, p0, p1, n=80, bend=0.08, wiggle=0.6):
    """Gently bent path from p0 to p1 with small hand-made wiggle (output px)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0, 1, n)
    d = p1 - p0
    L = np.hypot(*d)
    nrm = np.array([-d[1], d[0]]) / max(L, 1e-6)
    off = bend * L * np.sin(np.pi * t) * rng.choice([-1, 1])
    wig = np.cumsum(rng.normal(0, wiggle, n))
    wig -= np.linspace(wig[0], wig[-1], n)
    return p0[None] + t[:, None] * d[None] + (off + wig)[:, None] * nrm[None]


def smooth_profile(rng, n, lo, hi, corr=12):
    """Smooth random profile in [lo, hi] along a path of n points."""
    v = np.cumsum(rng.normal(0, 1, n + 4 * corr))
    v = np.convolve(v, np.ones(corr) / corr, mode="same")[2 * corr: 2 * corr + n]
    v = (v - v.min()) / (np.ptp(v) + 1e-9)
    return lo + (hi - lo) * v


def path_mask(shape, path, widths, intens=None, blur_out=0.35):
    """Polyline with per-vertex width and intensity (max-blended short segments)."""
    m = np.zeros(shape, np.float32)
    H, W = shape
    if intens is None:
        intens = np.ones(len(path))
    for i in range(len(path) - 1):
        a = np.round(np.asarray(path[i]) * SS * 16).astype(np.int64)
        b = np.round(np.asarray(path[i + 1]) * SS * 16).astype(np.int64)
        w = max(1, int(round(_s(0.5 * (widths[i] + widths[i + 1])))))
        pad = 2 * w + 4
        x0 = int(max(0, (min(a[0], b[0]) >> 4) - pad))
        y0 = int(max(0, (min(a[1], b[1]) >> 4) - pad))
        x1 = int(min(W, (max(a[0], b[0]) >> 4) + pad))
        y1 = int(min(H, (max(a[1], b[1]) >> 4) + pad))
        if x1 <= x0 or y1 <= y0:
            continue
        sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
        off = np.array([x0 * 16, y0 * 16])
        cv2.line(sub, tuple(int(v) for v in a - off), tuple(int(v) for v in b - off),
                 255, w, cv2.LINE_AA, shift=4)
        val = 0.5 * (intens[i] + intens[i + 1])
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], sub.astype(np.float32) / 255.0 * val)
    if blur_out > 0:
        m = cv2.GaussianBlur(m, (0, 0), _s(blur_out))
    return m


def _end_taper(n, frac=0.12):
    t = np.linspace(0, 1, n)
    return np.clip(np.minimum(t, 1 - t) / frac, 0, 1) ** 0.7


# ---------------------------------------------------------------- defect types

def d_scratch(cv: Canvas, rng, p0, p1, width=1.4, depth=1.0, lighten=0.9, glint=0.25,
              gaps=2, bend=0.05):
    """Hairline scratch: tapered, broken in places, groove + lighter exposed material whose
    strength varies along the length, and one short specular glint."""
    n = 220
    path = curve(rng, p0, p1, n=n, bend=bend, wiggle=0.2)
    inten = smooth_profile(rng, n, 0.35, 1.0, 18) * _end_taper(n, 0.15)
    t = np.linspace(0, 1, n)
    for c in rng.uniform(0.2, 0.8, gaps):            # the tip skipped here
        g = rng.uniform(0.012, 0.03)
        inten *= np.clip(np.abs(t - c) / g - 0.5, 0, 1)
    widths = width * smooth_profile(rng, n, 0.6, 1.1, 25) * (0.4 + 0.6 * _end_taper(n, 0.2))
    m = path_mask((cv.h, cv.w), path, widths, inten, 0.3)
    g0 = rng.uniform(0.3, 0.55)
    gl_int = np.exp(-((t - g0) / 0.05) ** 2) * inten
    gl = path_mask((cv.h, cv.w), path, widths * 0.6, gl_int, 0.4)
    cv.height -= depth * SS * m
    cv.alb_mul *= 1 + lighten * m
    cv.spec += glint * gl
    cv.flatten = np.maximum(cv.flatten, 0.5 * m)
    cv.mask = np.maximum(cv.mask, m)


def d_chip(cv: Canvas, rng, cx, cy, r=12, primer=(0.78, 0.78, 0.76), depth=1.6,
           rim=0.25, irregular=0.35, aspect=1.0, angle=0.0):
    """Paint chip: irregular flake missing, lighter primer inside, soft fractured rim."""
    out = blob_outline(rng, cx, cy, r, irregular, n=72, aspect=aspect, angle=angle)
    m = poly_mask((cv.h, cv.w), out, blur_out=0.45)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (cv.h, cv.w)).astype(np.float32), (0, 0), _s(2.0))
    noise /= noise.std() + 1e-6
    prim = np.asarray(primer, np.float32)[None, None, :] * (1 + 0.04 * noise[..., None])
    cv.blend_colour(m, prim)
    cv.height -= depth * SS * cv2.GaussianBlur(m, (0, 0), _s(0.5))
    cv.flatten = np.maximum(cv.flatten, 0.85 * m)
    ring = np.clip(cv2.GaussianBlur(m, (0, 0), _s(1.6)) - m, 0, 1)
    ring /= ring.max() + 1e-6
    cv.alb_mul *= 1 + rim * ring
    cv.mask = np.maximum(cv.mask, np.maximum(m, ring > 0.2))


def d_cut(cv: Canvas, rng, p0, p1, open_w=3.2, depth=3.0, lip=1.6, dark=0.18, edge=0.45,
          fray=0, fray_gain=0.5):
    """Knife cut: lens-shaped opening (shadowed, darker material inside), raised lips that
    catch the light, and a lighter cut edge (exposed cross-section) on one side."""
    n = 90
    path = curve(rng, p0, p1, n=n, bend=0.07, wiggle=0.12)
    t = np.linspace(0, 1, n)
    half = open_w / 2 * np.sin(np.pi * t) ** 0.8 * smooth_profile(rng, n, 0.7, 1.0, 15) + 0.2
    tang = np.gradient(path, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    left, right = path + nrm * half[:, None], path - nrm * half[:, None]
    m = poly_mask((cv.h, cv.w), np.vstack([left, right[::-1]]), blur_out=0.35)
    lips = path_mask((cv.h, cv.w), path, 2 * half + 3.5, _end_taper(n, 0.2), 1.0)
    lips = np.clip(lips - m, 0, 1)
    side = path + nrm * (half[:, None] + 0.5)
    em = path_mask((cv.h, cv.w), side, np.full(n, 0.9), np.sin(np.pi * t) ** 0.5, 0.3)
    em = np.clip(em - m, 0, 1)
    cv.height += lip * SS * lips
    cv.height -= depth * SS * m
    cv.alb_mul *= (1 - (1 - dark) * m) * (1 + edge * em)
    cv.flatten = np.maximum(cv.flatten, 0.4 * lips)
    cv.mask = np.maximum(cv.mask, np.maximum(m, lips > 0.15))
    # frayed fibres (fabrics): short light strands from both edges, partly across the gap
    fm = np.zeros_like(m)
    for _ in range(int(fray)):
        i = int(rng.integers(n // 8, 7 * n // 8))
        sgn = rng.choice([-1, 1])
        base = path[i] + sgn * nrm[i] * (half[i] + rng.uniform(0.0, 1.0))
        d = -sgn * nrm[i] * rng.uniform(0.6, 1.0) + tang[i] * rng.uniform(-0.7, 0.7)
        L = rng.uniform(2.0, 2.0 + 1.5 * open_w)
        k = 8
        fp = base[None] + np.linspace(0, L, k)[:, None] * d[None] / (np.linalg.norm(d) + 1e-9)
        fp += np.cumsum(rng.normal(0, 0.25, (k, 2)), 0)
        fm = np.maximum(fm, path_mask((cv.h, cv.w), fp, np.linspace(0.8, 0.4, k),
                                      np.linspace(1, 0.5, k) * rng.uniform(0.6, 1.0), 0.25))
    if fray:
        cv.alb_mul *= 1 + fray_gain * fm / (1 - (1 - dark) * m + 1e-3) * m + 0.6 * fray_gain * fm * (1 - m)
        cv.height += 0.6 * SS * fm
        cv.mask = np.maximum(cv.mask, fm)


def d_puncture(cv: Canvas, rng, cx, cy, r=4.5, ring_r=11.0, depth=4.0, dark=0.12, creases=5):
    """Puncture: small dark hole, pressed-in dish around it, short radial creases."""
    out = blob_outline(rng, cx, cy, r, 0.25, n=40)
    m = poly_mask((cv.h, cv.w), out, blur_out=0.35)
    yy, xx = np.mgrid[:cv.h, :cv.w].astype(np.float32)
    rr = np.hypot(xx - _s(cx), yy - _s(cy)) / SS
    dish = np.exp(-(rr / ring_r) ** 2)
    cv.height -= depth * 0.6 * SS * dish
    cv.height -= depth * SS * m
    cv.flatten = np.maximum(cv.flatten, 0.5 * dish)
    for _ in range(creases):
        a = rng.uniform(0, 2 * np.pi)
        L = rng.uniform(1.1, 1.8) * ring_r
        q0 = (cx + np.cos(a) * r, cy + np.sin(a) * r)
        q1 = (cx + np.cos(a) * L, cy + np.sin(a) * L)
        k = 24
        cr = path_mask((cv.h, cv.w), curve(rng, q0, q1, k, 0.1, 0.1),
                       np.linspace(1.4, 0.4, k), np.linspace(1, 0.2, k), 0.5)
        cv.height -= 0.8 * SS * cr
        cv.alb_mul *= 1 - 0.3 * cr
        cv.mask = np.maximum(cv.mask, cr)
    cv.alb_mul *= 1 - (1 - dark) * m
    cv.mask = np.maximum(cv.mask, np.maximum(m, dish > 0.3))


def d_burn(cv: Canvas, rng, cx, cy, r=10.0, hole_dark=0.06, char=(0.05, 0.065, 0.085),
           bead=2.5, bead_gloss=1.0):
    """Cigarette burn on fabric: hole, melted glossy dark rim, brown scorch halo."""
    hole_o = blob_outline(rng, cx, cy, r * 0.5, 0.35, n=60)
    rim_o = blob_outline(rng, cx, cy, r, 0.3, n=60)
    halo_o = blob_outline(rng, cx, cy, r * 1.9, 0.45, n=60)
    mh = poly_mask((cv.h, cv.w), hole_o, 0.4)
    mr = poly_mask((cv.h, cv.w), rim_o, 0.9)
    mz = poly_mask((cv.h, cv.w), halo_o, 3.5)
    cv.alb_mul *= 1 - 0.45 * mz                              # scorch darkening ...
    cv.blend_colour(0.35 * mz, (0.10, 0.19, 0.30))           # ... towards brown (BGR)
    cv.blend_colour(mr, char)                                # melted black-brown
    cv.alb_mul *= 1 - (1 - hole_dark) * mh
    cv.flatten = np.maximum(cv.flatten, 0.9 * mr)            # melted fibres are smooth
    cv.gloss = np.maximum(cv.gloss, bead_gloss * np.clip(mr - mh, 0, 1))  # ... and glossy
    # melted bead: a raised ring around the hole, catches the light
    ring = np.clip(cv2.GaussianBlur(mh, (0, 0), _s(1.8)) * 2.2 - mh * 2.2, 0, 1)
    cv.height += bead * SS * cv2.GaussianBlur(ring, (0, 0), _s(0.6))
    cv.height += 1.0 * SS * (cv2.GaussianBlur(mr, (0, 0), _s(1.2)) - mr * 0.4)
    cv.height -= 4.0 * SS * mh
    cv.mask = np.maximum(cv.mask, np.maximum(mr, mz > 0.3))


def d_scuff(cv: Canvas, rng, cx, cy, rx=45, ry=18, angle=0.2, lighten=0.22, smooth=0.75,
            streaks=18, streak_gain=0.45):
    """Scuff: locally abraded area - relief worn flat, slightly brighter, with fine
    parallel rub streaks."""
    out = blob_outline(rng, cx, cy, ry, 0.45, n=64, aspect=rx / ry, angle=angle)
    m = poly_mask((cv.h, cv.w), out, blur_out=4.0)
    cv.flatten = np.maximum(cv.flatten, smooth * m)
    cv.alb_mul *= 1 + lighten * m
    ca, sa = np.cos(angle), np.sin(angle)
    st = np.zeros_like(m)
    for _ in range(streaks):
        o = rng.uniform(-ry * 0.9, ry * 0.9)
        L = rx * rng.uniform(0.5, 1.2)
        c0 = np.array([cx - sa * o, cy + ca * o]) + rng.uniform(-rx * 0.2, rx * 0.2) * np.array([ca, sa])
        q0, q1 = c0 - L / 2 * np.array([ca, sa]), c0 + L / 2 * np.array([ca, sa])
        k = 30
        st = np.maximum(st, path_mask((cv.h, cv.w), curve(rng, q0, q1, k, 0.01, 0.05),
                                      np.full(k, rng.uniform(0.6, 1.2)),
                                      _end_taper(k, 0.3) * rng.uniform(0.4, 1.0), 0.4))
    st *= np.clip(m * 1.5, 0, 1)
    cv.alb_mul *= 1 + streak_gain * st
    cv.height -= 0.3 * SS * st
    cv.mask = np.maximum(cv.mask, m > 0.25)


def d_crack(cv: Canvas, rng, p0, angle, length=90, width=1.4, branches=2, depth=1.6,
            dark=0.35, edge=0.3):
    """Crack in a coating / grain: jagged, width varying, lighter broken edge, short branches."""
    def walk(p, a, L):
        pts = [np.asarray(p, float)]
        for _ in range(max(4, int(L / 2.5))):
            a += rng.normal(0, 0.3)
            pts.append(pts[-1] + 2.5 * np.array([np.cos(a), np.sin(a)]))
        return np.array(pts)

    main = walk(p0, angle, length)
    n = len(main)
    m = path_mask((cv.h, cv.w), main, width * smooth_profile(rng, n, 0.5, 1.2) *
                  (0.3 + 0.7 * _end_taper(n, 0.25)), None, 0.3)
    for _ in range(branches):
        i = rng.integers(n // 4, 3 * n // 4)
        b = walk(main[i], angle + rng.choice([-1, 1]) * rng.uniform(0.6, 1.1),
                 length * rng.uniform(0.2, 0.35))
        k = len(b)
        m = np.maximum(m, path_mask((cv.h, cv.w), b, width * 0.7 * np.linspace(1, 0.3, k), None, 0.3))
    ring = np.clip(cv2.GaussianBlur(m, (0, 0), _s(1.2)) * 2.0 - m, 0, 1)
    cv.height -= depth * SS * m
    cv.alb_mul *= (1 - (1 - dark) * m) * (1 + edge * ring)
    cv.mask = np.maximum(cv.mask, m)


DEFECTS = {"scratch": d_scratch, "chip": d_chip, "cut": d_cut, "puncture": d_puncture,
           "burn": d_burn, "scuff": d_scuff, "crack": d_crack}



# =========================================================================== renderer

def illumination(h, w, rng_seed, falloff=0.12, tilt=0.06, centre=(0.42, 0.38)):
    """Soft large-scale illumination: off-centre radial fall-off plus a gentle tilt."""
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    x, y = xx / w, yy / h
    cx, cy = centre
    r2 = ((x - cx) * 1.6) ** 2 + (y - cy) ** 2
    f = 1.0 - falloff * r2 / 0.5
    f *= 1.0 - tilt * (x - 0.5) - 0.5 * tilt * (y - 0.5)
    return f.astype(np.float32)


def render(spec: dict, with_defect: bool):
    """Render one candidate -> (BGR uint8 OUT_HxOUT_W, defect mask float32 OUT_HxOUT_W)."""
    rng = np.random.default_rng(spec["seed"])
    diff, nor = load_maps(spec["asset"])
    alb, n = crop_maps(diff, nor, *spec["crop"], rot90=spec.get("rot90", 0))
    alb = grade(alb, **spec.get("grade", {}))
    H, W = alb.shape[:2]
    cv = Canvas((H, W))
    if with_defect:
        d = dict(spec["defect"])
        kind = d.pop("type")
        DEFECTS[kind](cv, np.random.default_rng(spec["seed"] + 1), **d)

    # albedo with defect
    alb = alb * cv.alb_mul[..., None]
    alb = alb * (1 - cv.alb_mix[..., None]) + cv.alb_col * cv.alb_mix[..., None]
    alb_lin = srgb_to_lin(np.clip(alb, 0, 1))

    # normals: texture relief (strength, flattened by the defect) + defect height field
    k = float(spec.get("relief", 1.0))
    tx = n[..., 0] / n[..., 2] * k * (1 - cv.flatten)
    ty = n[..., 1] / n[..., 2] * k * (1 - cv.flatten)
    if np.any(cv.height):
        hgt = cv2.GaussianBlur(cv.height, (0, 0), 0.6 * SS)
        hx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        hy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        tx, ty = tx - hx, ty + hy               # image y points down, normal y points up
    nn = np.stack([tx, ty, np.ones_like(tx)], -1)
    nn /= np.linalg.norm(nn, axis=-1, keepdims=True)

    L = np.asarray(spec.get("light", (-0.45, 0.55, 0.70)), np.float32)
    L /= np.linalg.norm(L)
    V = np.array([0, 0, 1], np.float32)
    Hv = (L + V) / np.linalg.norm(L + V)
    ndl = np.clip(nn @ L, 0, None) / L[2]
    amb = float(spec.get("ambient", 0.4))
    shade = amb + (1 - amb) * ndl
    ndh = np.clip(nn @ Hv, 0, 1)
    ks, p = float(spec.get("ks", 0.03)), float(spec.get("shininess", 30))
    spec_t = ks * ndh ** p + cv.gloss * 0.25 * ndh ** 40 + cv.spec

    illum = illumination(H, W, spec["seed"], **spec.get("illum", {}))
    exposure = float(spec.get("exposure", 1.0))
    lin = (alb_lin * shade[..., None] + spec_t[..., None]) * illum[..., None] * exposure
    srgb = lin_to_srgb(lin)
    img = cv2.resize(srgb, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), float(spec.get("optics_blur", 0.55)))
    nrng = np.random.default_rng(spec["seed"] + 7)       # same noise for both images of a pair
    lum = nrng.normal(0, 1, (OUT_H, OUT_W, 1)).astype(np.float32)
    chroma = nrng.normal(0, 1, (OUT_H, OUT_W, 3)).astype(np.float32)
    sig = float(spec.get("noise", 1.4)) / 255.0
    img = img + sig * (0.85 * lum + 0.45 * chroma) * np.sqrt(np.clip(img, 0.02, 1))[..., :] * 1.6
    out = np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
    mask = cv2.resize(cv.mask, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    return out, mask


def encode_jpeg(img, q=JPEG_Q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    return buf.tobytes()


def decode(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


# =========================================================================== evaluation

def _cell_hits(mask_work: np.ndarray, gh: int, gw: int, win: int, step: int) -> np.ndarray:
    """bool[gh, gw]: the window of the cell touches the mask."""
    k = np.ones((win, win), np.uint8)
    d = cv2.dilate(mask_work.astype(np.uint8), k, anchor=(win - 1, win - 1))
    # window at (i*step, j*step) .. +win: dilate with anchor at bottom-right -> top-left lookup
    d2 = np.zeros_like(d)
    d2[: d.shape[0] - (win - 1), : d.shape[1] - (win - 1)] = d[win - 1:, win - 1:]
    return d2[: gh * step: step, : gw * step: step][:gh, :gw] > 0


def evaluate(img: np.ndarray, mask: np.ndarray | None, P) -> dict:
    """Run the real scan_surface; measure scores inside / outside the defect."""
    res = surface.scan_surface(img, P)
    thr = float(res.threshold)
    out = {"verdict": res.verdict, "reason": res.reason, "threshold": thr,
           "n_zones": len(res.zones), "zones": [list(map(int, z.bbox)) for z in res.zones],
           "timing_ms": res.timing_ms.get("total")}
    if res.score is None:
        out.update(max_in=None, max_out=None, zones_on_defect=None)
        return out, res
    w, h = res.work_size
    score, checked = res.score, res.checked
    gh, gw = score.shape
    sc = np.where(checked, score, 0.0)
    if mask is not None and mask.max() > 0:
        mw = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA) > 0.05
        near = cv2.dilate(mw.astype(np.uint8), np.ones((9, 9), np.uint8))
        far = cv2.dilate(mw.astype(np.uint8), np.ones((41, 41), np.uint8))
        cin = _cell_hits(near, gh, gw, P.win, P.step)
        cfar = _cell_hits(far, gh, gw, P.win, P.step)
        out["max_in"] = float(sc[cin].max()) if cin.any() else 0.0
        out["max_out"] = float(sc[~cfar].max()) if (~cfar).any() else 0.0
        zon = []
        for z in res.zones:
            x, y, bw, bh = z.bbox
            zon.append(bool(far[max(0, y):y + bh, max(0, x):x + bw].any()))
        out["zones_on_defect"] = bool(zon) and all(zon)
    else:
        out["max_in"] = None
        out["max_out"] = float(sc.max())
        out["zones_on_defect"] = None
    out["max_all"] = float(sc.max())
    return out, res


def variants(jpeg_bytes: bytes):
    """Robustness variants of a published JPEG: as is, q80 re-encode, 1600-wide and back."""
    img = decode(jpeg_bytes)
    q80 = decode(encode_jpeg(img, 80))
    up = cv2.resize(img, (1600, int(round(1600 * img.shape[0] / img.shape[1]))),
                    interpolation=cv2.INTER_CUBIC)
    back = cv2.resize(up, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    return {"orig": img, "jpeg_q80": q80, "resize_1600": back}


def judge(ev_def: dict, ev_ok: dict) -> dict:
    thr = ev_def["orig"]["threshold"]
    d, o = ev_def["orig"], ev_ok["orig"]
    ok = (d["verdict"] == "fail" and d["zones_on_defect"] and d["max_in"] >= MARGIN_DEFECT * thr
          and o["verdict"] == "pass" and o["max_all"] <= MARGIN_CLEAN * thr)
    flips = [k for k in ev_def if ev_def[k]["verdict"] != "fail" or not ev_def[k]["zones_on_defect"]]
    flips += [k + "(ok)" for k in ev_ok if ev_ok[k]["verdict"] != "pass"]
    return {"margins_ok": bool(ok), "robust_flips": flips, "robust": bool(ok) and not flips,
            "defect_ratio": d["max_in"] / thr if d["max_in"] is not None else None,
            "clean_ratio": o["max_all"] / thr if o.get("max_all") is not None else None}


# =========================================================================== candidates

CANDIDATES: list[dict] = [
    dict(name="black_leather_scratch", asset="brown_leather", crop=(1024, 0, 1400), seed=19,
         grade=dict(sat=0.15, gain=(0.65, 0.65, 0.65)), ks=0.05, exposure=1.6,
         title="Black seat leather - hairline scratch",
         desc="Black pebble-grain seat leather (colour-graded from a brown scan); a long, thin, lighter hairline scratch upper left of centre.",
         defect=dict(type="scratch", p0=(420, 470), p1=(700, 360), width=2.0, lighten=1.6, glint=0.15)),
    dict(name="leather_cut", asset="brown_leather", crop=(0, 1024, 1400), seed=11,
         title="Seat leather - knife cut",
         desc="Cognac seat leather with fine pebble grain; a short knife cut (gaping slit, raised lips) right of centre.",
         defect=dict(type="cut", p0=(770, 292), p1=(872, 362), open_w=4.5, dark=0.2, edge=0.6)),
    dict(name="suede_scuff", asset="scuba_suede", crop=(0, 1024, 2048), seed=12,
         grade=dict(sat=0.12, gain=(0.55, 0.55, 0.6)),
         title="Dark suede (Alcantara-like) - scuff",
         desc="Charcoal suede-like microfibre (colour-graded); an abraded scuff with rub streaks, lower right.",
         defect=dict(type="scuff", cx=860, cy=470, rx=70, ry=24, angle=-0.25, lighten=0.25)),
    dict(name="melange_burn", asset="jersey_melange", crop=(1024, 0, 2048), seed=13,
         grade=dict(sat=0.35, gain=(0.72, 0.72, 0.72)),
         title="Melange seat fabric - cigarette burn",
         desc="Blue-grey melange knit seat fabric (colour-graded); a cigarette burn hole with a melted rim, upper left.",
         defect=dict(type="burn", cx=430, cy=250, r=12.0, bead=4.0)),
    dict(name="corduroy_cut", asset="ribbed_corduroy", crop=(0, 1024, 1400), seed=14,
         grade=dict(sat=0.25, gain=(0.7, 0.7, 0.7)),
         title="Ribbed seat fabric - cut",
         desc="Grey ribbed (corduroy) seat fabric (colour-graded); a short diagonal cut across the ribs.",
         defect=dict(type="cut", p0=(548, 404), p1=(630, 484), open_w=6.5, dark=0.25, fray=26)),
    dict(name="galvanised_scratch", asset="corrugated_iron_02", crop=(0, 0, 2048), seed=17,
         title="Galvanised ribbed sheet - scratch",
         desc="Galvanised corrugated steel sheet; a long hairline scratch with a glint across the ribs.",
         defect=dict(type="scratch", p0=(380, 470), p1=(700, 330), width=1.5)),
    dict(name="headliner_tear", asset="terlenka", crop=(0, 1024, 1400), seed=18,
         title="Beige headliner fabric - tear",
         desc="Fine beige woven headliner fabric; a short slit tear, centre.",
         defect=dict(type="cut", p0=(630, 326), p1=(712, 356), open_w=5.0, dark=0.35, lip=0.8, fray=20)),
]

# =========================================================================== contact sheet

_TW, _TH, _CAP, _HDR = 320, 180, 40, 34


def _caption(img_w, lines, colour=(30, 30, 30), bg=(250, 250, 250)):
    cap = np.full((_CAP, img_w, 3), bg, np.uint8)
    for k, (txt, col) in enumerate(lines):
        cv2.putText(cap, txt, (6, 16 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col or colour, 1,
                    cv2.LINE_AA)
    return cap


def _verdict_lines(ev):
    t = ev["threshold"]
    col = (0, 0, 200) if ev["verdict"] == "fail" else (0, 140, 0)
    head = f"AVA: {ev['verdict'].upper()}" + (f"  {ev['n_zones']} zone(s)" if ev["n_zones"] else "")
    if ev.get("max_in") is not None:
        sub = f"defect {ev['max_in'] / t:.2f}x thr, elsewhere {ev['max_out'] / t:.2f}x"
    else:
        sub = f"max score {ev['max_all'] / t:.2f}x thr"
    return [(head, col), (sub, None)]


def contact_sheet(results, path):
    rows = []
    W = 5 * _TW
    title = np.full((56, W, 3), 255, np.uint8)
    cv2.putText(title, "AVA Surface check - demo candidates (real CC0 texture photos from Poly Haven + synthetic defects)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(title, "columns: defect photo | zoom x2.5 on the defect | AVA overlay | clean photo | AVA overlay.   "
                "Robust = verdict unchanged after JPEG q80 and 1600px resize round-trip",
                (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    rows.append(title)
    for k, r in enumerate(results, 1):
        d, o = r["eval"]["DEFECT"]["orig"], r["eval"]["OK"]["orig"]
        img_d = cv2.imread(r["files"]["DEFECT"])
        img_o = cv2.imread(r["files"]["OK"])
        ov_d = cv2.imread(os.path.join(OUT_DIR, f"{r['name']}_DEFECT_overlay.jpg"))
        ov_o = cv2.imread(os.path.join(OUT_DIR, f"{r['name']}_OK_overlay.jpg"))
        m = np.load(os.path.join(OUT_DIR, f"{r['name']}_mask.npy"))
        ys, xs = np.nonzero(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        zw, zh = 512, 288
        x0, y0 = int(np.clip(cx - zw // 2, 0, OUT_W - zw)), int(np.clip(cy - zh // 2, 0, OUT_H - zh))
        zoom = img_d[y0:y0 + zh, x0:x0 + zh * 16 // 9]
        tiles = [img_d, zoom, ov_d, img_o, ov_o]
        tiles = [cv2.resize(t, (_TW, _TH), interpolation=cv2.INTER_AREA) for t in tiles]
        # mark the zoom window on the first tile
        s = _TW / OUT_W
        cv2.rectangle(tiles[0], (int(x0 * s), int(y0 * s)), (int((x0 + zw) * s), int((y0 + zh) * s)),
                      (0, 220, 255), 1, cv2.LINE_AA)
        rob = "ROBUST" if r["robust"] else f"NOT robust: {r['robust_flips']}"
        caps = [
            _caption(_TW, [("input: with defect", None), (r["defect"]["type"], (90, 90, 90))]),
            _caption(_TW, [("zoom (same pixels)", None), ("", None)]),
            _caption(_TW, _verdict_lines(d)),
            _caption(_TW, [("input: same photo, no defect", None), ("", None)]),
            _caption(_TW, _verdict_lines(o)),
        ]
        body = np.vstack([np.hstack(tiles), np.hstack(caps)])
        info = r["asset"]
        hdr = np.full((_HDR, W, 3), (235, 235, 235), np.uint8)
        txt = (f"{k}. {r['title']}   [{rob}; defect {r['defect_ratio']:.2f}x, clean {r['clean_ratio']:.2f}x thr]"
               f"   texture: '{info['name']}' by {', '.join(info['authors'])} (Poly Haven, CC0)")
        cv2.putText(hdr, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.vstack([hdr, body, np.full((6, W, 3), 255, np.uint8)]))
    cv2.imwrite(path, np.vstack(rows))


def _row(spec, tag, ev):
    return {"file": f"{spec['name']}_{tag}.jpg", **ev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    P = config.Params()
    only = set(filter(None, args.only.split(",")))
    results = []
    for spec in CANDIDATES:
        if only and spec["name"] not in only:
            continue
        evs, files, mask = {}, {}, None
        for tag, wd in (("DEFECT", True), ("OK", False)):
            img, m = render(spec, wd)
            b = encode_jpeg(img)
            path = os.path.join(OUT_DIR, f"{spec['name']}_{tag}.jpg")
            with open(path, "wb") as f:
                f.write(b)
            files[tag] = path
            if wd:
                mask = m
            if args.no_eval:
                continue
            evs[tag] = {}
            for vname, vimg in variants(b).items():
                ev, res = evaluate(vimg, mask if wd else None, P)
                evs[tag][vname] = ev
                if vname == "orig":
                    with open(os.path.join(OUT_DIR, f"{spec['name']}_{tag}_overlay.jpg"), "wb") as f:
                        f.write(res.images["overlay"])
        if args.no_eval:
            print("wrote", spec["name"])
            continue
        np.save(os.path.join(OUT_DIR, f"{spec['name']}_mask.npy"), (mask > 0.05).astype(np.uint8))
        j = judge(evs["DEFECT"], evs["OK"])
        info = asset_info(spec["asset"])
        rec = {"name": spec["name"], "title": spec["title"], "description": spec["desc"],
               "files": files, "asset": {k: info[k] for k in ("id", "name", "authors", "url", "licence")},
               "colour_grade": spec.get("grade"), "defect": spec["defect"],
               "label": "real texture photo (CC0, Poly Haven) + synthetic defect; synthetic lighting/noise",
               "eval": evs, **j}
        results.append(rec)
        d, o = evs["DEFECT"]["orig"], evs["OK"]["orig"]
        print(f"{spec['name']:20s} DEF {d['verdict']:5s} in={d['max_in'] / d['threshold']:.2f} "
              f"out={d['max_out'] / d['threshold']:.2f} zones={d['n_zones']} on={d['zones_on_defect']} | "
              f"OK {o['verdict']:5s} max={o['max_all'] / o['threshold']:.2f} | robust={j['robust']} "
              f"flips={j['robust_flips']}", flush=True)
        for k in ("jpeg_q80", "resize_1600"):
            dd, oo = evs["DEFECT"][k], evs["OK"][k]
            print(f"   {k:12s} DEF {dd['verdict']} in={dd['max_in'] / dd['threshold']:.2f} on={dd['zones_on_defect']}"
                  f" | OK {oo['verdict']} max={oo['max_all'] / oo['threshold']:.2f}")
    if results and not only:
        with open(os.path.join(HERE, "candidates.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        contact_sheet(results, os.path.join(HERE, "contact_sheet.png"))
        print("wrote candidates.json and contact_sheet.png")
    elif results:
        with open(os.path.join(HERE, "candidates_partial.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
