"""Tolerant change maps on the shared window grid (see docs/SPEC.md).

A change map compares the reference and the inspected frame window by window.
Small residual misregistration is absorbed by searching +-`tol` cells around each
cell and keeping the best match, in both directions:

    d_ab[i, j] = min_{|dy|,|dx| <= tol} dist_fn(a, b, dy, dx)[i, j]
    d_ba[i, j] = min_{|dy|,|dx| <= tol} dist_fn(b, a, dy, dx)[i, j]
    change     = max(d_ab, d_ba)

The symmetric max makes both "something added" (only in b) and "something
removed" (only in a) visible. `dist_fn` is any distance with the contract of
`pbp.pair_distance` / the `baselines` distances: ``dist_fn(a, b, dy, dx)`` returns
float32[gh, gw] comparing cell (i, j) of `a` with cell (i+dy, j+dx) of `b`, +inf
where the partner is outside the grid. Features are computed once by the caller;
this module only calls `dist_fn` once per shift.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .config import Params

DistFn = Callable[..., np.ndarray]

# Value used for cells where no shift has a partner inside the grid (all distances are in [0, 1]).
OUT_OF_RANGE = 1.0

_known_symmetric: set | None = None


def _is_symmetric(dist_fn: DistFn) -> bool:
    """True for distances known to satisfy D(a, b) == D(b, a) cell by cell.

    That holds for PBP and all baselines; a custom `dist_fn` may declare it with a
    ``symmetric = True`` attribute.
    """
    global _known_symmetric
    if getattr(dist_fn, "symmetric", False):
        return True
    if _known_symmetric is None:
        from . import baselines, pbp  # lazy: keeps this module light to import

        _known_symmetric = {pbp.pair_distance, baselines.ssim_distance,
                            baselines.census_distance, baselines.pixdiff_distance}
    return dist_fn in _known_symmetric


def _grid_of(feat) -> tuple[int, int] | None:
    """Window grid `(gh, gw)` of PBP Signatures / baseline WindowFeatures (None if unknown)."""
    if hasattr(feat, "coeffs"):
        return tuple(feat.coeffs.shape[:2])
    grid = getattr(feat, "grid", None)
    return tuple(grid) if grid is not None else None


def _shifts(tol: int):
    r = range(-int(tol), int(tol) + 1)
    return [(dy, dx) for dy in r for dx in r]


def _shift_into(dst: np.ndarray, src: np.ndarray, dy: int, dx: int) -> None:
    """``dst[i, j] = min(dst[i, j], src[i + dy, j + dx])`` where the source index is valid."""
    h, w = dst.shape
    i0, i1 = max(0, -dy), min(h, h - dy)
    j0, j1 = max(0, -dx), min(w, w - dx)
    if i0 < i1 and j0 < j1:
        np.minimum(dst[i0:i1, j0:j1], src[i0 + dy:i1 + dy, j0 + dx:j1 + dx],
                   out=dst[i0:i1, j0:j1])


def tolerant_change_map(dist_fn: DistFn, a, b, tol: int,
                        symmetric: bool | None = None) -> np.ndarray:
    """Symmetric, shift-tolerant change score per cell of `a`'s grid, float32[gh, gw] in [0, 1].

    `a`, `b`: features of the reference and of the frame (same method).
    `tol`: search radius in cells (0 = compare aligned cells only).
    `symmetric`: if the distance is symmetric, ``d_ba`` is obtained by re-indexing
    the ``d_ab`` maps instead of calling `dist_fn` again (half the work, identical
    result). ``None`` auto-detects PBP and the baselines. Only used when both grids
    have the same shape.
    Cells for which no shift has a partner in range are clamped to 1.0
    (the zero shift is always in range when both grids have the same shape).
    """
    tol = max(0, int(tol))
    maps = {s: np.asarray(dist_fn(a, b, *s), np.float32) for s in _shifts(tol)}
    d_ab = np.minimum.reduce(list(maps.values()))
    gh, gw = d_ab.shape

    if symmetric is None:
        symmetric = _is_symmetric(dist_fn)

    if symmetric and _grid_of(b) == (gh, gw):
        # dist_fn(b, a, dy, dx)[i, j] == dist_fn(a, b, -dy, -dx)[i + dy, j + dx].
        d_ba = np.full((gh, gw), np.inf, np.float32)
        for dy, dx in maps:
            _shift_into(d_ba, maps[(-dy, -dx)], dy, dx)
    else:
        rev = np.minimum.reduce([np.asarray(dist_fn(b, a, *s), np.float32)
                                 for s in _shifts(tol)])
        # Bring b's grid onto a's grid (identical unless the grids differ in size).
        d_ba = np.full((gh, gw), np.inf, np.float32)
        h, w = min(gh, rev.shape[0]), min(gw, rev.shape[1])
        d_ba[:h, :w] = rev[:h, :w]

    out = np.maximum(d_ab, d_ba)
    out[~np.isfinite(out)] = OUT_OF_RANGE
    return out.astype(np.float32, copy=False)


def multi_reference_change_map(dist_fn: DistFn, refs: Sequence, frame, tol: int,
                               symmetric: bool | None = None) -> np.ndarray:
    """Element-wise minimum over references of `tolerant_change_map(dist_fn, ref, frame, tol)`.

    Several good shots absorb natural part-to-part variation: a cell only counts
    as changed if it differs from every reference.
    """
    if len(refs) == 0:
        raise ValueError("at least one reference is required")
    out = None
    for ref in refs:
        m = tolerant_change_map(dist_fn, ref, frame, tol, symmetric=symmetric)
        out = m if out is None else np.minimum(out, m, out=out)
    return out


def pbp_change_map(sig_ref, sig_frame, params: Params) -> np.ndarray:
    """PBP change map with ``params.tolerance``; `sig_ref` may also be a list of references."""
    from .pbp import pair_distance

    refs = list(sig_ref) if isinstance(sig_ref, (list, tuple)) else [sig_ref]
    return multi_reference_change_map(pair_distance, refs, sig_frame, params.tolerance)
