"""Mode 1: single-photo surface check (self-referenced PBP anomaly), see docs/SPEC.md.

One photo of a uniform surface (painted panel, plastic, sheet metal, glass), no
reference: the surface itself is the reference. On a uniform surface every normal
window has many look-alikes elsewhere in the same photo, while a scratch, chip,
stain edge or sticker border is rare.

    scan_surface(photo)   resize -> capture quality (blur / dark) -> blurred gray
                          -> checked cells (border margin, no glare halo) -> glare /
                             coverage gates -> per-window anomaly score -> smoothing
                          -> zones -> verdict (never PASS if a glare core was masked)
                             + overlay JPEG

Anomaly score of window a (``surface_score_map``, method "pbp"):

    knn(a)  = mean of the ``surface_k`` smallest d'(a, b) over a fixed, seeded random
              sample of ``surface_samples`` windows b of the same image (samples within
              ``surface_exclude_cells`` cells of a are skipped);
    d'(a,b) = max(r(a), r(b)) * d(a, b), where d is ``pbp.pair_distance``'s metric
              (0.5 * L1 of the normalised coefficients averaged over both orientations;
              flat-vs-flat 0, flat-vs-non-flat 1) and r(c) ramps from 0 to 1 with the window
              contrast c (see ``_noise_ramp``: absolute bounds ``surface_noise_floor`` ..
              ``surface_noise_full``, raised to ``surface_rel_floor`` .. ``surface_rel_full``
              times the image's 25th-percentile window contrast). Below the ramp the pixel
              order is set by sensor noise or by a random micro-texture, so two such windows
              count as alike (a soft extension of the flat-window rule, like
              ``contrast_floor`` in the reference pipeline);
    score   = clip(knn + surface_contrast_weight * cterm, 0, 1), where cterm ramps from 0
              to 1 as the robust z-score of log(window contrast) (median / MAD over the
              checked windows) goes from ``surface_contrast_z`` to ``+ surface_contrast_span``.
              Log + median makes it invariant to global exposure / gain.

What each term does (measured on the synthetic set while developing, not a benchmark):
the kNN term carries the PBP structure and works where the surface's windows repeat
(smooth paint, glossy dark surfaces, regular patterns); on random micro-textures
(grain, brushing) ordinary windows have no look-alikes either, so there the relative
noise gate switches the kNN term off and the contrast term does most of the work.
Thin, soft defects in high-variance textures (a dried-water ring on brushed metal) are
a known weak case.

Glare: ``glare_region`` = ``quality.glare_mask`` cores plus their smooth halo (a steep
brightness ramp would otherwise be "rare"); such cells are not checked. Colour alone
cannot tell a highlight from a white defect (label, tape, a bright chip clipped to 255),
so if a glare core of at least ``glare_blob_min_px`` was masked the verdict is never
PASS: without zones it is RETAKE (glare) with the hints change_angle and bright_spot.

Speed: the k nearest samples are first searched with the Hellinger affinity (one
matrix product of square-rooted coefficient vectors), then the ``surface_candidates``
best candidates are re-ranked with the exact distance. Everything is deterministic:
the sample is drawn with ``params.surface_seed``.

Baselines for eval (same grid, glare mask, smoothing and zones):
- "census": the same kNN self-reference (and noise gate / contrast term) on 8-neighbour
  census codes (mean Hamming distance / 8, exact via a matrix product on the bits);
- "gradient": per-window mean Sobel magnitude, robust z-score vs the image median / MAD,
  score = max(z, 0) / 10.

Known limitation: the photo must show only the surface; part edges, holes, logos and
background are "rare" as well and become zones.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import baselines, quality
from .align import resize_max_side
from .config import Params
from .pbp import NBINS, cell_mask_from_pixel_mask, serialize_signatures, window_signatures
from .pipeline import HINT_BRIGHT_SPOT, HINT_CHANGE_ANGLE, InspectResult, render_overlay, smooth_scores
from .zones import extract_zones

SURFACE_METHODS = ("pbp", "census", "gradient")
_RESULT_METHOD = {"pbp": "pbp_surface", "census": "census_surface", "gradient": "gradient_surface"}
_JPEG_QUALITY = 85
_CHUNK = 2048                 # windows per chunk of the kNN search (bounds memory)
_MAX_THREADS = 4
_GRADIENT_SCALE = 10.0        # gradient baseline: score = robust z / this
_GLARE_OPEN_PX = 7            # glare cores thinner than this are not highlights
_BLUR_HP_SIGMA = 8.0          # px; high-pass for the sharpness gate (removes illumination)


# --------------------------------------------------------------------------- helpers

def _as_bgr(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(img, dtype=np.uint8)


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8, copy=False), cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(img, dtype=np.float32)


def _work_gray(img: np.ndarray, params: Params) -> np.ndarray:
    g = _gray(img)
    if params.blur_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), params.blur_sigma)
    return g


def _jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _cells_to_pixels(cells: np.ndarray, shape: tuple[int, int], win: int, step: int) -> np.ndarray:
    """Pixel mask covered by the windows of the True cells."""
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    if cells.any():
        up = np.kron(cells.astype(np.uint8), np.ones((step, step), np.uint8))
        hh, ww = min(h, up.shape[0]), min(w, up.shape[1])
        out[:hh, :ww] = up[:hh, :ww]
        extra = win - step
        if extra > 0:
            k = extra + 1
            out = cv2.dilate(out, np.ones((k, k), np.uint8), anchor=(k - 1, k - 1))
    return out.astype(bool)


def _grid_contrast(gray: np.ndarray, win: int, step: int, gh: int, gw: int) -> np.ndarray:
    """Window max - min on the grid (same as Signatures.contrast)."""
    kernel = np.ones((win, win), np.uint8)
    wmax = cv2.dilate(gray, kernel, anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    wmin = cv2.erode(gray, kernel, anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    return (wmax - wmin)[: gh * step: step, : gw * step: step][:gh, :gw].astype(np.float32)


def _noise_ramp(contrast: np.ndarray, params: Params, checked: np.ndarray | None = None) -> np.ndarray:
    """Weight in [0, 1] of a window's PBP structure: 0 at low contrast, 1 at high contrast.

    Absolute bounds: surface_noise_floor (at least flat_floor) .. surface_noise_full gray
    levels, below which the pixel order is set by sensor noise. With `checked`, the bounds
    are raised to surface_rel_floor / surface_rel_full times the 25th-percentile window
    contrast of the checked cells: on a random micro-texture (grain, brushing) the pixel
    order of ordinary windows is random as well, so no window has look-alikes; only
    windows well above the texture's own contrast count. The relative bounds scale with
    the image gain, so they do not react to exposure.
    """
    lo = max(float(params.surface_noise_floor), float(params.flat_floor))
    hi = float(params.surface_noise_full)
    if checked is not None and checked.any():
        c_ref = float(np.percentile(contrast[checked], 25))
        lo = max(lo, float(params.surface_rel_floor) * c_ref)
        hi = max(hi, float(params.surface_rel_full) * c_ref)
    if hi <= lo:
        return (contrast >= lo).astype(np.float32)
    return np.clip((contrast - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _robust_z(values: np.ndarray, sel: np.ndarray, min_sigma: float) -> np.ndarray:
    """(v - median) / (1.4826 * MAD) using the selected values (sigma floored)."""
    ref = values[sel] if sel.any() else values.ravel()
    if ref.size == 0:
        return np.zeros_like(values, np.float32)
    med = float(np.median(ref))
    sigma = max(1.4826 * float(np.median(np.abs(ref - med))), min_sigma)
    return ((values - med) / sigma).astype(np.float32)


def _contrast_term(contrast: np.ndarray, checked: np.ndarray, params: Params) -> np.ndarray:
    """Robust outlier term of log window contrast (gain-invariant), gated by the noise ramp."""
    lc = np.log(np.maximum(contrast, 0.5)).astype(np.float32)
    z = _robust_z(lc, checked, 0.15)
    span = max(float(params.surface_contrast_span), 1e-6)
    term = np.clip((z - float(params.surface_contrast_z)) / span, 0.0, 1.0)
    return (term * _noise_ramp(contrast, params)).astype(np.float32)


def _sample_cells(pool: np.ndarray, params: Params) -> np.ndarray:
    """Deterministic sample (flat indices, sorted) of up to surface_samples pool cells."""
    idx = np.flatnonzero(pool.ravel())
    n = min(int(params.surface_samples), idx.size)
    if n <= 0:
        return idx[:0]
    rng = np.random.default_rng(int(params.surface_seed))
    return np.sort(rng.choice(idx, size=n, replace=False))


def _knn_scores(n_cells: int, gw: int, targets: np.ndarray, samples: np.ndarray,
                ramp: np.ndarray, approx_fn, exact_fn, params: Params) -> np.ndarray:
    """Mean of the k smallest noise-gated distances from each target cell to the samples.

    approx_fn(t_idx) -> float32[nt, S] distances from the target cells to all samples,
    used to pick the `surface_candidates` best samples; exact_fn(t_idx, cand) ->
    float32[nt, C] exact distances to the samples at positions `cand` (exact_fn None:
    approx is already exact). Gating: d' = max(r_a, r_b) * d. Chunks run on a small
    thread pool (BLAS / OpenCV / large numpy ops release the GIL).
    """
    out = np.zeros(n_cells, np.float32)
    if targets.size == 0 or samples.size == 0:
        return out
    k = max(1, int(params.surface_k))
    e = int(params.surface_exclude_cells)
    s_i, s_j = (samples // gw).astype(np.int32), (samples % gw).astype(np.int32)
    r_s = ramp.ravel()[samples].astype(np.float32)
    n_cand = int(min(max(k, int(params.surface_candidates)), samples.size))
    kk = min(k, n_cand if exact_fn is not None else samples.size)

    def run(c0: int) -> None:
        t = targets[c0:c0 + _CHUNK]
        t_i, t_j = (t // gw).astype(np.int32), (t % gw).astype(np.int32)
        gate = np.maximum(ramp.ravel()[t].astype(np.float32)[:, None], r_s[None, :])
        excl = (np.abs(t_i[:, None] - s_i[None, :]) <= e) & (np.abs(t_j[:, None] - s_j[None, :]) <= e)
        d = approx_fn(t)
        d *= gate
        d[excl] = np.inf
        if exact_fn is None:
            best = np.partition(d, kk - 1, axis=1)[:, :kk]
        else:
            cand = np.argpartition(d, n_cand - 1, axis=1)[:, :n_cand]
            de = exact_fn(t, cand) * np.take_along_axis(gate, cand, axis=1)
            de[np.take_along_axis(excl, cand, axis=1)] = np.inf
            best = np.partition(de, kk - 1, axis=1)[:, :kk]
        best = np.where(np.isfinite(best), best, 1.0)
        out[t] = best.mean(axis=1)

    starts = list(range(0, targets.size, _CHUNK))
    workers = min(_MAX_THREADS, len(starts), os.cpu_count() or 1)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, starts))
    else:
        for c0 in starts:
            run(c0)
    return out


def _knn_targets(checked: np.ndarray, ramp: np.ndarray, samples: np.ndarray,
                 params: Params) -> np.ndarray:
    """Cells to score. Low-contrast cells (r = 0) score exactly 0 when at least k
    low-contrast samples are left after any exclusion box, so they are skipped then."""
    targets = np.flatnonzero(checked.ravel())
    box = (2 * int(params.surface_exclude_cells) + 1) ** 2
    if np.count_nonzero(ramp.ravel()[samples] == 0) >= int(params.surface_k) + box:
        targets = targets[ramp.ravel()[targets] > 0]
    return targets


# --------------------------------------------------------------------------- descriptors

def _pbp_knn(sig, checked: np.ndarray, ramp: np.ndarray, params: Params) -> np.ndarray:
    """kNN self-reference on PBP signatures. Flat windows have all-zero coefficients, so
    the plain L1 already gives flat-vs-non-flat 1; flat-vs-flat is 0 through the noise
    gate (the ramp is 0 below max(surface_noise_floor, flat_floor))."""
    gh, gw = sig.flat.shape
    coeffs = sig.coeffs.reshape(gh * gw, 2 * NBINS)
    samples = _sample_cells(checked, params)
    targets = _knn_targets(checked, ramp, samples, params)
    if targets.size == 0 or samples.size == 0:
        return np.zeros((gh, gw), np.float32)
    s_coeffs = coeffs[samples]
    # Only the bins used by some sample matter: outside them b = 0, so
    # L1(a, b) = sum_U |a_U - b_U| + (|a| - |a_U|). Typically halves the work.
    used = np.flatnonzero(s_coeffs.any(axis=0))
    if used.size < 2:  # all samples flat: keep two (all-zero) constant bins, shapes stay valid
        used = np.array([0, NBINS], np.int64)
    s_used = np.ascontiguousarray(s_coeffs[:, used])         # [S, U]
    # sqrt of the probability vectors: Bhattacharyya affinity per orientation by a dot product.
    root_s = np.sqrt(s_used).T.copy()                        # [U, S]

    def approx(t):
        bc = np.sqrt(np.ascontiguousarray(coeffs[t][:, used])) @ root_s
        return np.float32(1.0) - np.float32(0.5) * bc        # mean over the two orientations

    def exact(t, cand):
        a_full = coeffs[t]
        a = np.ascontiguousarray(a_full[:, used])
        rest = a_full.sum(axis=1) - a.sum(axis=1)            # mass of a outside the used bins
        res = np.empty(cand.shape, np.float32)
        for c in range(cand.shape[1]):
            diff = cv2.absdiff(a, s_used[cand[:, c]])
            res[:, c] = cv2.reduce(diff, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F)[:, 0]
        res += np.maximum(rest, 0.0)[:, None]
        res *= np.float32(0.25)                              # 0.5 * L1, mean of 2 orientations
        return np.clip(res, 0.0, 1.0)

    return _knn_scores(gh * gw, gw, targets, samples, ramp, approx, exact, params).reshape(gh, gw)


def _census_bits(gray: np.ndarray, params: Params) -> tuple[np.ndarray, object]:
    feat = baselines.census_features(gray, params)
    gh, gw = feat.grid
    bits = np.unpackbits(feat.blocks.reshape(gh * gw, -1), axis=1).astype(np.float32)
    return bits, feat


def _census_knn(bits: np.ndarray, shape: tuple[int, int], checked: np.ndarray,
                ramp: np.ndarray, params: Params) -> np.ndarray:
    gh, gw = shape
    nb = np.float32(bits.shape[1])
    samples = _sample_cells(checked, params)
    targets = _knn_targets(checked, ramp, samples, params)
    if targets.size == 0 or samples.size == 0:
        return np.zeros((gh, gw), np.float32)
    pop = bits.sum(axis=1)
    bs_t, pop_s = bits[samples].T.copy(), pop[samples]

    def approx(t):  # exact Hamming / (64 * 8) = mean census distance
        ham = pop[t][:, None] + pop_s[None, :] - 2.0 * (bits[t] @ bs_t)
        return np.clip(ham / nb, 0.0, 1.0).astype(np.float32)

    return _knn_scores(gh * gw, gw, targets, samples, ramp, approx, None, params).reshape(gh, gw)


def _gradient_score(gray: np.ndarray, checked: np.ndarray, params: Params) -> np.ndarray:
    win, step = params.win, params.step
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    box = cv2.boxFilter(mag, -1, (win, win), anchor=(0, 0), normalize=True,
                        borderType=cv2.BORDER_REPLICATE)
    gh, gw = checked.shape
    m = box[: gh * step: step, : gw * step: step][:gh, :gw]
    z = _robust_z(m, checked, 0.5)
    return (np.maximum(z, 0.0) / _GRADIENT_SCALE).astype(np.float32)


def _glare_core(img: np.ndarray, params: Params) -> np.ndarray:
    """Undilated highlight cores (uint8 0/1): ``quality.glare_mask`` without thin structures.

    Highlight cores are blobs: an opening drops thin saturated lines (a bright scratch
    is a defect, not glare).
    """
    core = quality.glare_mask(img, dataclasses.replace(params, glare_dilate_px=0)).astype(np.uint8)
    return cv2.morphologyEx(core, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_GLARE_OPEN_PX,) * 2))


def _largest_blob(mask: np.ndarray) -> int:
    """Pixel count of the largest 8-connected blob of a 0/1 mask."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0


def glare_region(img: np.ndarray, gray: np.ndarray, params: Params) -> np.ndarray:
    """Specular highlight plus its halo (bool[H, W]).

    Core = ``quality.glare_mask`` (bright, unsaturated pixels) without structures thinner
    than ``_GLARE_OPEN_PX``. The halo around a highlight
    is a steep, smooth brightness ramp: on a uniform surface such windows are rare and would
    score as defects. Halo = pixels at least ``surface_glare_elev`` gray levels (or 4 %)
    above the local surface level (median over ~250 px), connected to a core and within
    ``surface_glare_halo_px``; the union is dilated by ``surface_glare_dilate_px``.
    """
    return _glare_region(img, gray, params, _glare_core(img, params))


def _glare_region(img: np.ndarray, gray: np.ndarray, params: Params, core: np.ndarray) -> np.ndarray:
    """`glare_region` from precomputed cores (`_glare_core`)."""
    r0 = int(params.glare_dilate_px)
    if r0 > 0:
        core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r0 + 1,) * 2))
    core = core > 0
    if not core.any():
        return core
    h, w = gray.shape
    f = 4                                     # the halo is smooth: work at 1/4 scale
    hs, ws = max(1, h // f), max(1, w // f)
    small = cv2.resize(np.clip(gray, 0, 255), (ws, hs), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.0)  # suppress surface texture
    core_s = cv2.resize(core.astype(np.uint8), (ws, hs), interpolation=cv2.INTER_AREA) > 0
    # Surface level: median over ~250 px (wider than a highlight band), at 1/16 scale.
    tiny = cv2.resize(small, (max(1, ws // 4), max(1, hs // 4)), interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(np.round(tiny).astype(np.uint8), 15).astype(np.float32)
    bg = cv2.resize(bg, (ws, hs), interpolation=cv2.INTER_LINEAR)
    elev = (small - bg) >= np.maximum(float(params.surface_glare_elev), 0.04 * bg)
    cand = core_s | elev
    r = int(params.surface_glare_halo_px) // f
    if r > 0:
        near = cv2.dilate(core_s.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        cand &= near | core_s
    n, labels = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(labels[core_s])] = True
    keep[0] = False
    halo = cv2.resize(keep[labels].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    region = core | halo
    d = int(params.surface_glare_dilate_px)
    if d > 0:
        region = cv2.dilate(region.astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))) > 0
    return region


# --------------------------------------------------------------------------- public API

def _score(gray: np.ndarray, checked: np.ndarray, params: Params, method: str):
    """(raw score map, sha256 hex of the descriptors)."""
    win, step = params.win, params.step
    if method == "pbp":
        sig = window_signatures(gray, params)
        ramp = _noise_ramp(sig.contrast, params, checked)
        knn = _pbp_knn(sig, checked, ramp, params)
        score = knn + float(params.surface_contrast_weight) * _contrast_term(sig.contrast, checked, params)
        sha = hashlib.sha256(serialize_signatures(sig)).hexdigest()
    elif method == "census":
        bits, feat = _census_bits(gray, params)
        gh, gw = feat.grid
        contrast = _grid_contrast(gray, win, step, gh, gw)
        ramp = _noise_ramp(contrast, params, checked)
        knn = _census_knn(bits, (gh, gw), checked, ramp, params)
        score = knn + float(params.surface_contrast_weight) * _contrast_term(contrast, checked, params)
        sha = hashlib.sha256(np.ascontiguousarray(feat.blocks).tobytes()).hexdigest()
    elif method == "gradient":
        score = _gradient_score(gray, checked, params)
        q = np.clip(np.rint(score * 1000.0), 0, 65535).astype(np.uint16)
        sha = hashlib.sha256(q.tobytes()).hexdigest()
        return score, sha
    else:
        raise ValueError(f"unknown surface method: {method!r} (known: {', '.join(SURFACE_METHODS)})")
    return np.clip(score, 0.0, 1.0).astype(np.float32), sha


def surface_score_map(gray: np.ndarray, params: Params, method: str = "pbp",
                      checked: np.ndarray | None = None) -> np.ndarray:
    """Raw (unsmoothed) per-window anomaly score float32[gh, gw] of a working gray image.

    `gray` is used as given (apply the working blur first, as scan_surface does).
    `checked` (bool[gh, gw]) restricts scoring, sampling and the robust statistics to
    those cells (default: all); unchecked cells score 0 for pbp / census.
    """
    g = _gray(gray) if np.asarray(gray).ndim == 3 else np.asarray(gray, np.float32)
    h, w = g.shape
    gh, gw = (h - params.win) // params.step + 1, (w - params.win) // params.step + 1
    if gh <= 0 or gw <= 0:
        raise ValueError("image smaller than one window")
    if checked is None:
        checked = np.ones((gh, gw), bool)
    return _score(g, np.asarray(checked, bool), params, method)[0]


def method_threshold(method: str, params: Params) -> float:
    if method == "pbp":
        return float(params.surface_threshold)
    return float(getattr(params, f"threshold_surface_{method}"))


def scan_surface(image_bgr: np.ndarray, params: Params, method: str = "pbp") -> InspectResult:
    """Single-photo surface check (see module docstring). Deterministic."""
    if method not in SURFACE_METHODS:
        raise ValueError(f"unknown surface method: {method!r} (known: {', '.join(SURFACE_METHODS)})")
    timing: dict = {}
    t0 = tic = time.perf_counter()

    def lap(name):
        nonlocal tic
        now = time.perf_counter()
        timing[name] = round((now - tic) * 1000.0, 1)
        tic = now

    img, _ = resize_max_side(_as_bgr(image_bgr), params.work_max_side)
    h, w = img.shape[:2]
    lap("resize")
    threshold = method_threshold(method, params)

    def finish(verdict, reason, *, zones=(), cov=0.0, changed=0.0, images=None, sha=None,
               hints=(), score=None, checked=None) -> InspectResult:
        timing["total"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return InspectResult(verdict=verdict, reason=reason, zones=list(zones), coverage=float(cov),
                             changed_frac=float(changed), align={}, timing_ms=dict(timing),
                             method=_RESULT_METHOD[method], threshold=threshold,
                             images=images or {}, signature_sha256=sha, hints=list(hints),
                             work_size=(w, h), score=score, checked=checked)

    # 1. capture quality (no reference: absolute gates only). Sharpness is measured on the
    #    high-pass image: on a uniform surface the gray-level spread is mostly the
    #    illumination fall-off, which would make a sharp photo read as "blurry".
    g = _gray(img)
    reason = None
    if quality.is_too_dark(g, params):  # first: a dark photo also lacks detail
        reason = "dark"
    elif quality.blur_score(g - cv2.GaussianBlur(g, (0, 0), _BLUR_HP_SIGMA)) < params.min_blur:
        reason = "blurry"
    lap("quality")
    frame_jpeg = _jpeg(img)
    if reason is not None:
        return finish("retake", reason, images={"frame": frame_jpeg, "overlay": frame_jpeg})

    # 2. what is checked: border margin, glare highlight + its halo
    win, step = params.win, params.step
    gh, gw = (h - win) // step + 1, (w - win) // step + 1
    if gh <= 0 or gw <= 0:
        raise ValueError("working image smaller than one window")
    area = np.ones((h, w), bool)
    m = int(params.border_margin_px)
    if m > 0:
        area[:m], area[-m:], area[:, :m], area[:, -m:] = False, False, False, False
    core = _glare_core(img, params)
    glare = _glare_region(img, g, params, core) & area
    glare_frac = float(glare.sum()) / max(1, int(area.sum()))
    bright = params.glare_blob_min_px > 0 and _largest_blob(core & area) >= params.glare_blob_min_px
    cells_area = cell_mask_from_pixel_mask(area, win, step, frac=1.0)
    cells_glare = cell_mask_from_pixel_mask(glare, win, step, frac=1.0 / (win * win))
    checked = cells_area & ~cells_glare
    cov = float(checked.mean()) if checked.size else 0.0
    unchecked_px = ~_cells_to_pixels(checked, (h, w), win, step)
    lap("mask")
    gate = "glare" if glare_frac > params.max_glare_frac else (
        "low_coverage" if cov < params.min_coverage else None)
    if gate is not None:
        overlay = _jpeg(render_overlay(img, [], unchecked_px))
        return finish("retake", gate, cov=cov, hints=[HINT_CHANGE_ANGLE] if gate == "glare" else [],
                      images={"frame": frame_jpeg, "overlay": overlay})

    # 3. anomaly score -> smoothing -> zones
    gray_f = _work_gray(img, params)
    score, sha = _score(gray_f, checked, params, method)
    lap("score")
    score = smooth_scores(np.where(checked, score, 0.0).astype(np.float32), checked,
                          params.surface_smooth)
    changed = float(np.count_nonzero(checked & (score >= threshold))) / max(1, int(checked.sum()))
    zones = extract_zones(score, checked, dataclasses.replace(params, threshold=threshold),
                          win=win, step=step)
    lap("zones")
    overlay = _jpeg(render_overlay(img, zones, unchecked_px))
    lap("render")
    if zones:
        verdict, reason, hints = "fail", None, []
    elif bright:  # a masked glare core may hide a white defect: never a silent PASS
        verdict, reason, hints = "retake", "glare", [HINT_CHANGE_ANGLE, HINT_BRIGHT_SPOT]
    else:
        verdict, reason, hints = "pass", None, []
    res = finish(verdict, reason, zones=zones, cov=cov, changed=changed, hints=hints,
                 images={"frame": frame_jpeg, "overlay": overlay}, sha=sha,
                 score=score, checked=checked)
    res.bright_spot = bool(bright)
    return res
