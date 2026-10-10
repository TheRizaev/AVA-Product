"""Baseline change detectors on the shared PBP window grid (SSIM, Census, pixel diff).

Each baseline is a pair ``(feature_fn, dist_fn)``:
- ``feature_fn(gray, params) -> WindowFeatures``
- ``dist_fn(a, b, dy=0, dx=0) -> float32[gh, gw]`` compares cell (i, j) of ``a``
  with cell (i+dy, j+dx) of ``b``; cells whose partner is out of range get +inf.
This is the same contract as ``pbp.window_signatures`` / ``pbp.pair_distance``,
so ``change.tolerant_change_map`` works with any of them.

``gray`` is the working image (float32 0..255, already blurred by the caller if
desired); a BGR image is converted to gray for convenience.

``METHODS`` maps a method name to ``(feature_fn, dist_fn)`` and includes
``"pbp"``. It is built lazily on first access (module ``__getattr__``), so
importing this module never imports ``dalil.pbp``. ``get_methods()`` returns
the same dict explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dalil.config import Params

# Number of set bits for every byte value.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)

# SSIM stabilisers for an 8-bit dynamic range.
_C1 = (0.01 * 255.0) ** 2
_C2 = (0.03 * 255.0) ** 2


@dataclass
class WindowFeatures:
    """Per-window data on the shared grid.

    ``blocks``: window pixels ``[gh, gw, win*win]`` (float32 for SSIM/pixdiff,
    uint8 census codes for Census). ``mean``/``var``: per-window statistics
    ``[gh, gw]`` (SSIM only, otherwise ``None``).
    """

    kind: str
    blocks: np.ndarray
    win: int
    step: int
    shape: tuple[int, int]
    mean: np.ndarray | None = None
    var: np.ndarray | None = None

    @property
    def grid(self) -> tuple[int, int]:
        return self.blocks.shape[0], self.blocks.shape[1]


# --------------------------------------------------------------------------- #
# Grid helpers (local copy so this module does not depend on pbp at import)
# --------------------------------------------------------------------------- #

def _grid_shape(h: int, w: int, win: int, step: int) -> tuple[int, int]:
    """Same formula as ``pbp.grid_shape``."""
    return max(0, (h - win) // step + 1), max(0, (w - win) // step + 1)


def _as_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8) if img.dtype != np.uint8 else img,
                           cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(img, dtype=np.float32)


def _window_blocks(arr: np.ndarray, win: int, step: int) -> np.ndarray:
    """Pixel blocks of every grid window, ``[gh, gw, win*win]`` (contiguous copy)."""
    gh, gw = _grid_shape(arr.shape[0], arr.shape[1], win, step)
    if gh == 0 or gw == 0:
        return np.zeros((gh, gw, win * win), arr.dtype)
    view = np.lib.stride_tricks.sliding_window_view(arr, (win, win))
    view = view[::step, ::step][:gh, :gw]
    return np.ascontiguousarray(view).reshape(gh, gw, win * win)


def _shifted(a: WindowFeatures, b: WindowFeatures, dy: int, dx: int, cell_dist) -> np.ndarray:
    """Evaluate ``cell_dist(a_sel, b_sel)`` on overlapping cells; +inf elsewhere.

    ``cell_dist`` receives two index tuples ``(rows, cols)`` (for a and for b)
    and returns float32 distances for that block of cells.
    """
    gha, gwa = a.grid
    ghb, gwb = b.grid
    out = np.full((gha, gwa), np.inf, np.float32)
    i0, i1 = max(0, -dy), min(gha, ghb - dy)
    j0, j1 = max(0, -dx), min(gwa, gwb - dx)
    if i1 <= i0 or j1 <= j0:
        return out
    sel_a = (slice(i0, i1), slice(j0, j1))
    sel_b = (slice(i0 + dy, i1 + dy), slice(j0 + dx, j1 + dx))
    out[sel_a] = cell_dist(sel_a, sel_b)
    return out


# --------------------------------------------------------------------------- #
# SSIM
# --------------------------------------------------------------------------- #

def ssim_features(gray: np.ndarray, params: Params) -> WindowFeatures:
    """Window means/variances (cv2.boxFilter) plus pixel blocks for covariances."""
    g = _as_gray(gray)
    win, step = params.win, params.step
    gh, gw = _grid_shape(g.shape[0], g.shape[1], win, step)
    # anchor=(0, 0): the output at (y, x) averages g[y:y+win, x:x+win].
    box = dict(ddepth=-1, ksize=(win, win), anchor=(0, 0), normalize=True,
               borderType=cv2.BORDER_REPLICATE)
    mu = cv2.boxFilter(g, **box)[::step, ::step][:gh, :gw]
    mu2 = cv2.boxFilter(g * g, **box)[::step, ::step][:gh, :gw]
    var = np.maximum(mu2 - mu * mu, 0.0)
    return WindowFeatures("ssim", _window_blocks(g, win, step), win, step, g.shape[:2],
                          mean=np.ascontiguousarray(mu), var=np.ascontiguousarray(var))


def ssim_distance(a: WindowFeatures, b: WindowFeatures, dy: int = 0, dx: int = 0) -> np.ndarray:
    """Per-window SSIM dissimilarity ``(1 - ssim) / 2`` in [0, 1]."""

    def cell_dist(sa, sb):
        mu_a, mu_b = a.mean[sa], b.mean[sb]
        cov = np.mean(a.blocks[sa] * b.blocks[sb], axis=-1) - mu_a * mu_b
        num = (2 * mu_a * mu_b + _C1) * (2 * cov + _C2)
        den = (mu_a ** 2 + mu_b ** 2 + _C1) * (a.var[sa] + b.var[sb] + _C2)
        ssim = num / den
        return np.clip((1.0 - ssim) / 2.0, 0.0, 1.0)

    return _shifted(a, b, dy, dx, cell_dist)


# --------------------------------------------------------------------------- #
# Census
# --------------------------------------------------------------------------- #

def _census_codes(g: np.ndarray) -> np.ndarray:
    """8-neighbour census transform: bit k set if neighbour k is darker than the centre."""
    p = np.pad(g, 1, mode="edge")
    h, w = g.shape
    codes = np.zeros((h, w), np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for bit, (oy, ox) in enumerate(offsets):
        neigh = p[1 + oy:1 + oy + h, 1 + ox:1 + ox + w]
        codes |= (neigh < g).astype(np.uint8) << bit
    return codes


def census_features(gray: np.ndarray, params: Params) -> WindowFeatures:
    """Per-pixel census codes grouped into window blocks ``uint8[gh, gw, win*win]``."""
    g = _as_gray(gray)
    codes = _census_codes(g)
    return WindowFeatures("census", _window_blocks(codes, params.win, params.step),
                          params.win, params.step, g.shape[:2])


def census_distance(a: WindowFeatures, b: WindowFeatures, dy: int = 0, dx: int = 0) -> np.ndarray:
    """Mean Hamming distance of census codes over the window, / 8 (in [0, 1])."""

    def cell_dist(sa, sb):
        bits = _POPCOUNT[np.bitwise_xor(a.blocks[sa], b.blocks[sb])]
        return bits.mean(axis=-1, dtype=np.float32) / 8.0

    return _shifted(a, b, dy, dx, cell_dist)


# --------------------------------------------------------------------------- #
# Pixel difference
# --------------------------------------------------------------------------- #

def pixdiff_features(gray: np.ndarray, params: Params) -> WindowFeatures:
    """Raw window pixel blocks ``float32[gh, gw, win*win]``."""
    g = _as_gray(gray)
    return WindowFeatures("pixdiff", _window_blocks(g, params.win, params.step),
                          params.win, params.step, g.shape[:2])


def pixdiff_distance(a: WindowFeatures, b: WindowFeatures, dy: int = 0, dx: int = 0) -> np.ndarray:
    """Mean absolute pixel difference over the window, / 255 (in [0, 1])."""

    def cell_dist(sa, sb):
        return np.abs(a.blocks[sa] - b.blocks[sb]).mean(axis=-1) / 255.0

    return _shifted(a, b, dy, dx, cell_dist)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

BASELINE_METHODS = {
    "ssim": (ssim_features, ssim_distance),
    "census": (census_features, census_distance),
    "pixdiff": (pixdiff_features, pixdiff_distance),
}

_methods_cache: dict | None = None


def get_methods(include_pbp: bool = True) -> dict:
    """Name -> ``(feature_fn, dist_fn)``; ``"pbp"`` is imported lazily from dalil.pbp."""
    global _methods_cache
    if not include_pbp:
        return dict(BASELINE_METHODS)
    if _methods_cache is None:
        from dalil import pbp  # lazy: pbp is developed in parallel

        _methods_cache = {"pbp": (pbp.window_signatures, pbp.pair_distance), **BASELINE_METHODS}
    return _methods_cache


def __getattr__(name: str):
    """Lazy module attribute ``METHODS`` (PEP 562): ``from dalil.baselines import METHODS``."""
    if name == "METHODS":
        return get_methods()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
