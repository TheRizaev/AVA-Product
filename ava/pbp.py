"""Pseudo-Boolean polynomial (PBP) window signatures.

Method: Chikake & Goldengorin (PBP image representation). This is our own
implementation; see the README for references and the authors' `tmc-pbp` library.

PBP of an m x n patch C (rows are the variables y_1..y_m, columns are "clients"):
for every column j and every distinct value t > min_j in that column, add the term
``(t - prev_j(t)) * prod_{i : C[i, j] < t} y_i`` where ``prev_j(t)`` is the largest
column value below t; the constant is ``sum_j min_j``. Like monomials are merged.

The set of monomials only depends on the per-column ordering of the pixels, so it
is invariant to any strictly monotone brightness change (gamma, exposure, contrast).

Public API (see docs/SPEC.md):
    pbp_reference, format_polynomial, poly_degree,
    Signatures, window_signatures, pair_distance, serialize_signatures,
    grid_shape, cell_mask_from_pixel_mask
"""

from __future__ import annotations

import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import Params

NBINS = 256                 # 2**8 monomials for an 8-variable window
_SUPPORTED_WIN = 8
_CHUNK_WINDOWS = 4096       # windows per vectorised chunk (bounds temporary memory)
_MAX_THREADS = 4
_SERIAL_MAGIC = b"AVA_PBP1"
_SERIAL_VERSION = 1


# --------------------------------------------------------------------------- #
# Window grid helpers
# --------------------------------------------------------------------------- #

def grid_shape(h: int, w: int, win: int, step: int) -> tuple[int, int]:
    """Number of window rows/columns `(gh, gw)` for an `h x w` image."""
    gh = (h - win) // step + 1 if h >= win else 0
    gw = (w - win) // step + 1 if w >= win else 0
    return gh, gw


def cell_mask_from_pixel_mask(mask_hw: np.ndarray, win: int, step: int,
                              frac: float = 0.25) -> np.ndarray:
    """Cell-level mask: cell (i, j) is True if >= `frac` of its window pixels are True."""
    m = np.asarray(mask_hw).astype(bool)
    h, w = m.shape
    gh, gw = grid_shape(h, w, win, step)
    # Integral image -> per-window pixel counts in O(1) each.
    ii = np.zeros((h + 1, w + 1), np.int64)
    ii[1:, 1:] = m.cumsum(0).cumsum(1)
    y0 = (np.arange(gh) * step)[:, None]
    x0 = (np.arange(gw) * step)[None, :]
    y1, x1 = y0 + win, x0 + win
    counts = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    return counts >= frac * win * win


# --------------------------------------------------------------------------- #
# Exact reference implementation
# --------------------------------------------------------------------------- #

def pbp_reference(patch: np.ndarray) -> dict[tuple[int, ...], float]:
    """Exact PBP of a 2-D patch.

    Keys are sorted 0-based row tuples (`()` is the constant term), values are the
    merged coefficients. Monomials with zero coefficient never appear (except `()`).
    """
    c = np.asarray(patch, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError("patch must be 2-D")
    poly: dict[tuple[int, ...], float] = {(): 0.0}
    for j in range(c.shape[1]):
        col = c[:, j]
        values = np.unique(col)                      # sorted distinct values
        poly[()] += float(values[0])
        for prev, t in zip(values[:-1], values[1:]):
            key = tuple(int(i) for i in np.flatnonzero(col < t))
            poly[key] = poly.get(key, 0.0) + float(t - prev)
    return poly


def _format_coeff(c: float) -> str:
    return str(int(round(c))) if float(c).is_integer() else f"{c:g}"


def format_polynomial(poly: dict[tuple[int, ...], float]) -> str:
    """Render like the paper (1-based variables), ordered by degree then lexicographically.

    Example: ``"11 + 11y3 + 3y4 + 2y1y3 + 4y2y3 + 4y1y4 + 12y1y2y3 + 6y1y2y4"``.
    """
    terms = []
    for key in sorted(poly, key=lambda k: (len(k), k[::-1])):
        coeff = poly[key]
        if coeff == 0:
            continue
        mono = "".join(f"y{i + 1}" for i in key)
        terms.append(_format_coeff(coeff) + mono)
    return " + ".join(terms) if terms else "0"


def poly_degree(poly: dict[tuple[int, ...], float]) -> int:
    """Largest number of variables in a monomial with non-zero coefficient."""
    return max((len(k) for k, c in poly.items() if c != 0), default=0)


# --------------------------------------------------------------------------- #
# Vectorised window signatures
# --------------------------------------------------------------------------- #

@dataclass
class Signatures:
    """PBP signatures on the window grid (see docs/SPEC.md).

    coeffs:   float32[gh, gw, 2, 256]; orientation 0 = polynomial over window rows,
              1 = over window columns (transposed window). Index = bitmask of the
              variables in the monomial (bin 0, the constant, is always 0).
              Each [i, j, o, :] sums to 1, or is all-zero (flat window, or a window
              with no non-constant term in that orientation).
    contrast: float32[gh, gw] window max - min.
    degree:   int8[gh, gw, 2] polynomial degree per orientation (0 for flat windows).
    flat:     bool[gh, gw] contrast < params.flat_floor.
    eps, flat_floor: parameters used (kept for the serialisation header).
    """

    coeffs: np.ndarray
    contrast: np.ndarray
    degree: np.ndarray
    flat: np.ndarray
    win: int
    step: int
    shape: tuple[int, int]
    eps: float = 0.0
    flat_floor: float = 0.0


def _as_gray_f32(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3:
        a = cv2.cvtColor(a if a.dtype == np.uint8 else a.astype(np.float32),
                         cv2.COLOR_BGR2GRAY)
    if a.ndim != 2:
        raise ValueError("expected a gray HxW image (or BGR HxWx3)")
    return np.ascontiguousarray(a, dtype=np.float32)


def _quantise(g: np.ndarray) -> tuple[np.ndarray, float]:
    """Map gray levels to non-negative int32 with an exact power-of-two scale `(q, scale)`.

    Integer comparisons/differences are faster than float ones and the scale is
    irrelevant after normalisation. Values stay below 2**27 so that `q << 3` (room
    for a 3-bit variable index) fits in int32. Integer images are represented exactly.
    """
    g64 = g.astype(np.float64)
    lo = float(g64.min())
    span = float(g64.max()) - lo
    scale = 2.0 ** np.floor(np.log2((2 ** 27 - 1) / span)) if span > 0 else 1.0
    return np.rint((g64 - lo) * scale).astype(np.int32), scale


_VAR_INDEX = np.arange(_SUPPORTED_WIN, dtype=np.int32)[:, None, None]

# Optimal 19-comparator sorting network for 8 inputs (depth 6).
_SORT8 = ((0, 2), (1, 3), (4, 6), (5, 7),
          (0, 4), (1, 5), (2, 6), (3, 7),
          (0, 1), (2, 3), (4, 5), (6, 7),
          (2, 4), (3, 5),
          (1, 4), (3, 6),
          (1, 2), (3, 4), (5, 6))


def _sort8(rows: list[np.ndarray]) -> list[np.ndarray]:
    """Sort 8 equally shaped arrays element-wise (rows[0] <= ... <= rows[7])."""
    r = list(rows)
    for i, j in _SORT8:
        lo = np.minimum(r[i], r[j])
        np.maximum(r[i], r[j], out=r[j])
        r[i] = lo
    return r


def _orientation_terms(keys: np.ndarray, thresh: np.ndarray):
    """PBP terms of M columns ("clients") of 8 values each (one orientation).

    keys:   int32[8, M] - ``(q << 3) | r`` for variable r (row r of the column).
    thresh: int32[M] - epsilon-tie threshold in quantised units: pixel r is
            "below" value t iff ``C[r] < t - thresh``.
    Returns ``(masks, weights, degree)``: int32[7, M] monomial bitmasks,
    int64[7, M] (unnormalised) coefficients and int8[M] per-column degree.

    After sorting the column, the pixels below any value form a prefix of the
    sorted order, so every monomial is a prefix mask ``pref[c-1]`` (c = 1..7
    pixels below). With eps = 0 the term of prefix c is the gap
    ``s[c] - s[c-1]`` (zero for ties), which is the exact PBP. With eps > 0 the
    prefix c is produced by the values t with ``s[c-1] < t - thresh <= s[c]``;
    its coefficient is the largest such t minus ``s[c-1]`` (0 if none).
    """
    r = _sort8([keys[i] for i in range(_SUPPORTED_WIN)])
    s = [k >> 3 for k in r]
    masks = np.empty((_SUPPORTED_WIN - 1,) + keys.shape[1:], np.int32)
    np.left_shift(1, r[0] & 7, out=masks[0])
    for c in range(1, _SUPPORTED_WIN - 1):
        np.bitwise_or(masks[c - 1], np.left_shift(1, r[c] & 7), out=masks[c])

    weights = np.empty(masks.shape, np.int64)
    if not thresh.any():
        for c in range(1, _SUPPORTED_WIN):
            np.subtract(s[c], s[c - 1], out=weights[c - 1])
        top = s[-1]
    else:
        for c in range(1, _SUPPORTED_WIN):
            lim = s[c] + thresh
            upper = s[c]                                  # largest t with t - thresh <= s[c]
            for k in range(c + 1, _SUPPORTED_WIN):
                upper = np.where(s[k] <= lim, s[k], upper)
            weights[c - 1] = np.where(upper - thresh > s[c - 1], upper - s[c - 1], 0)
        top = s[-1] - thresh
    # Degree = number of pixels below the column maximum (a prefix => popcount = count).
    degree = np.zeros(keys.shape[1:], np.int8)
    for c in range(_SUPPORTED_WIN - 1):
        degree += s[c] < top
    return masks, weights, degree


def window_signatures(gray: np.ndarray, params: Params) -> Signatures:
    """PBP signatures of all `win x win` windows (stride `step`) of a gray image.

    The image is used as given (the caller applies the optional working blur).
    Fully vectorised over chunks of windows; chunks run on a small thread pool
    (numpy releases the GIL), each writing a disjoint slice of the output.
    """
    win, step = int(params.win), int(params.step)
    if win != _SUPPORTED_WIN:
        raise ValueError(f"only win={_SUPPORTED_WIN} is supported")
    g = _as_gray_f32(gray)
    h, w = g.shape
    gh, gw = grid_shape(h, w, win, step)

    coeffs = np.zeros((gh, gw, 2, NBINS), np.float32)
    degree = np.zeros((gh, gw, 2), np.int8)
    contrast = np.zeros((gh, gw), np.float32)
    flat = np.zeros((gh, gw), bool)
    sig = Signatures(coeffs, contrast, degree, flat, win, step, (h, w),
                     float(params.eps), float(params.flat_floor))
    if gh == 0 or gw == 0:
        return sig

    # Window contrast via max/min filters anchored at the window's top-left corner.
    kernel = np.ones((win, win), np.uint8)
    wmax = cv2.dilate(g, kernel, anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    wmin = cv2.erode(g, kernel, anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    sl = (slice(0, gh * step, step), slice(0, gw * step, step))
    contrast[:] = wmax[sl] - wmin[sl]
    flat[:] = contrast < params.flat_floor

    q, scale = _quantise(g)
    thresh_q = np.floor(float(params.eps) * scale * contrast.astype(np.float64))
    thresh_q = np.clip(thresh_q, 0, 2 ** 30).astype(np.int32)
    windows = sliding_window_view(q << 3, (win, win))[::step, ::step]   # [gh, gw, y, x]
    coeffs_flat = coeffs.reshape(-1)

    def run_chunk(i0: int, i1: int) -> None:
        wy = windows[i0:i1].reshape(-1, win, win)            # strided copy, [N, y, x]
        n = wy.shape[0]
        th = np.repeat(thresh_q[i0:i1].ravel(), win)        # per column, [N * 8]
        base = (np.arange(i0 * gw, i1 * gw, dtype=np.int64) * (2 * NBINS)).repeat(win)
        # Orientation 0: variables = window rows (y), clients = columns (x).
        # Orientation 1: transposed window -> variables = columns, clients = rows.
        for o, perm in enumerate(((1, 0, 2), (2, 0, 1))):
            keys = np.bitwise_or(wy.transpose(perm), _VAR_INDEX).reshape(win, -1)
            masks, weights, deg = _orientation_terms(keys, th)
            degree[i0:i1, :, o] = deg.reshape(n, win).max(axis=1).reshape(i1 - i0, gw)
            total = weights.reshape(win - 1, n, win).sum(axis=(0, 2))
            inv = np.divide(1.0, total, out=np.zeros(n), where=total > 0).repeat(win)
            # Merge like monomials: scatter-add into this chunk's (disjoint) output rows.
            np.add.at(coeffs_flat, (masks + (base + o * NBINS)).ravel(),
                      (weights * inv).astype(np.float32).ravel())

    rows_per_chunk = max(1, _CHUNK_WINDOWS // gw)
    chunks = [(i0, min(gh, i0 + rows_per_chunk)) for i0 in range(0, gh, rows_per_chunk)]
    workers = min(_MAX_THREADS, len(chunks), os.cpu_count() or 1)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda c: run_chunk(*c), chunks))
    else:
        for c in chunks:
            run_chunk(*c)

    coeffs[flat] = 0.0
    degree[flat] = 0
    return sig


# --------------------------------------------------------------------------- #
# Distances and serialisation
# --------------------------------------------------------------------------- #

def pair_distance(a: Signatures, b: Signatures, dy: int = 0, dx: int = 0) -> np.ndarray:
    """Distance between cell (i, j) of `a` and cell (i+dy, j+dx) of `b`, float32[gh, gw].

    Per orientation 0.5 * L1 of the normalised coefficient vectors (in [0, 1]),
    averaged over the two orientations. Flat vs flat -> 0, flat vs non-flat -> 1,
    cells whose partner falls outside `b`'s grid -> +inf. Symmetric:
    ``pair_distance(a, b, dy, dx)[i, j] == pair_distance(b, a, -dy, -dx)[i+dy, j+dx]``.
    """
    gh, gw = a.coeffs.shape[:2]
    bh, bw = b.coeffs.shape[:2]
    out = np.full((gh, gw), np.inf, np.float32)
    i0, i1 = max(0, -dy), min(gh, bh - dy)
    j0, j1 = max(0, -dx), min(gw, bw - dx)
    if i0 >= i1 or j0 >= j1:
        return out

    # One grid row at a time: [cells, 512] blocks with contiguous rows, so OpenCV's
    # SIMD absdiff/reduce can be used (several times faster than numpy here).
    ca = a.coeffs.reshape(gh, gw, 2 * NBINS)
    cb = b.coeffs.reshape(bh, bw, 2 * NBINS)
    d = np.empty((i1 - i0, j1 - j0), np.float32)
    for i in range(i0, i1):
        diff = cv2.absdiff(ca[i, j0:j1], cb[i + dy, j0 + dx:j1 + dx])
        d[i - i0] = cv2.reduce(diff, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F)[:, 0]
    d *= np.float32(0.25)

    fa = a.flat[i0:i1, j0:j1]
    fb = b.flat[i0 + dy:i1 + dy, j0 + dx:j1 + dx]
    d[fa ^ fb] = 1.0
    d[fa & fb] = 0.0
    out[i0:i1, j0:j1] = np.clip(d, 0.0, 1.0)
    return out


def serialize_signatures(sig: Signatures) -> bytes:
    """Deterministic byte representation (used for the SHA-256 fingerprint).

    Layout: magic, version, shapes and parameters (little endian), then coeffs
    quantised to uint8 (round(255 * c)), contrast quantised to uint8, degree int8,
    and the packed flat mask - all in C order.
    """
    gh, gw = sig.coeffs.shape[:2]
    header = _SERIAL_MAGIC + struct.pack(
        "<IIIIIIIdd", _SERIAL_VERSION, int(sig.shape[0]), int(sig.shape[1]),
        int(gh), int(gw), int(sig.win), int(sig.step),
        float(sig.eps), float(sig.flat_floor))
    # round(255 * c) saturated to uint8 (coeffs are >= 0, so the "abs" is a no-op); OpenCV
    # rounds half-to-even like np.rint and is bit-identical to it, but ~6x faster here.
    c = np.ascontiguousarray(sig.coeffs, np.float32)
    q = cv2.convertScaleAbs(c.reshape(-1, NBINS), alpha=255.0) if c.size else c.astype(np.uint8)
    con = np.clip(np.rint(np.asarray(sig.contrast, np.float32)), 0, 255).astype(np.uint8)
    deg = np.asarray(sig.degree, np.int8)
    flat = np.packbits(np.asarray(sig.flat, bool).ravel())
    return b"".join([header, q.tobytes(), con.tobytes(), deg.tobytes(), flat.tobytes()])
