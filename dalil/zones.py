"""Changed-cell grid -> ranked zones, plus shape/content hints and coverage.

Pipeline for `extract_zones`:
1. changed = checked & (score >= params.threshold)
2. 8-connected components on the cell grid ("blobs").
3. Blobs whose pixel bboxes are closer than `params.merge_gap_px` are merged
   (union-find, single linkage) into zones; stats are recomputed per zone.
4. Tiny zones are dropped, the rest ranked by n_cells * mean_score and capped.
5. Each zone gets an optional hint (water/dirt, something removed/added).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from dalil.config import Params

HINT_WATER = "water_or_dirt"
HINT_REMOVED = "something_removed"
HINT_ADDED = "something_added"


@dataclass
class Zone:
    id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h in working-image pixels (BEFORE frame)
    n_cells: int
    area_px: int
    mean_score: float
    max_score: float
    rank_score: float  # n_cells * mean_score
    hint: str | None = None
    contour: list[list[int]] = field(default_factory=list)


@dataclass
class _Blob:
    """One 8-connected component of changed cells (cell-grid coordinates)."""

    label: int
    i0: int  # first row (inclusive)
    j0: int  # first column (inclusive)
    i1: int  # last row (exclusive)
    j1: int  # last column (exclusive)
    n_cells: int

    def pixel_box(self, win: int, step: int) -> tuple[int, int, int, int]:
        """Pixel bbox as (x0, y0, x1, y1), exclusive end."""
        return (self.j0 * step, self.i0 * step,
                (self.j1 - 1) * step + win, (self.i1 - 1) * step + win)

    def roundness(self) -> float:
        """Fill ratio of the cell bbox times its short/long side ratio (1.0 = compact square)."""
        h, w = self.i1 - self.i0, self.j1 - self.j0
        return (self.n_cells / float(h * w)) * (min(h, w) / float(max(h, w)))


# --------------------------------------------------------------------------- helpers

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _cells_to_pixels(cells: np.ndarray, win: int, step: int) -> np.ndarray:
    """Upsample a bool cell mask [gh,gw] to the pixel mask covered by its windows."""
    gh, gw = cells.shape
    h, w = (gh - 1) * step + win, (gw - 1) * step + win
    out = np.zeros((h, w), np.uint8)
    if not cells.any():
        return out
    # Each cell paints its first `min(step, win)` pixels, then a bottom-right dilation
    # extends every block to the full window size.
    base = min(step, win)
    block = np.zeros((step, step), np.uint8)
    block[:base, :base] = 1
    painted = np.kron(cells.astype(np.uint8), block)[:h, :w]
    out[: painted.shape[0], : painted.shape[1]] = painted
    extra = win - base
    if extra > 0:
        k = extra + 1
        kernel = np.ones((k, k), np.uint8)
        out = cv2.dilate(out, kernel, anchor=(k - 1, k - 1))
    return out


def _contour(pix: np.ndarray) -> list[list[int]]:
    """Simplified outer polygon of a pixel mask (convex hull if it has several pieces)."""
    contours, _ = cv2.findContours(pix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    if len(contours) == 1:
        poly = contours[0]
    else:
        poly = cv2.convexHull(np.vstack(contours))
    eps = max(1.0, 0.005 * cv2.arcLength(poly, True))
    poly = cv2.approxPolyDP(poly, eps, True)
    return [[int(p[0][0]), int(p[0][1])] for p in poly]


def _water_hint(blobs: list[_Blob], params: Params) -> bool:
    """Many small, roughly round blobs => looks like water drops or dirt splashes."""
    if len(blobs) < params.blob_min_count:
        return False
    drops = [b for b in blobs
             if b.n_cells <= params.water_blob_max_cells
             and b.roundness() >= params.water_min_roundness]
    return len(drops) >= params.blob_min_count and len(drops) >= 0.7 * len(blobs)


def _activity_hint(cells: np.ndarray, sig_before, sig_after, params: Params) -> str | None:
    """Compare mean PBP degree and contrast in the zone before vs after.

    The activity ratio is the geometric mean of the degree ratio and the contrast
    ratio (small offsets keep flat cells from dividing by zero).
    """
    if sig_before is None or sig_after is None:
        return None
    try:
        deg_b = np.asarray(sig_before.degree, np.float64)[cells].mean()
        deg_a = np.asarray(sig_after.degree, np.float64)[cells].mean()
        con_b = np.asarray(sig_before.contrast, np.float64)[cells].mean()
        con_a = np.asarray(sig_after.contrast, np.float64)[cells].mean()
    except (AttributeError, IndexError, ValueError):
        return None  # missing fields or grid mismatch: no hint rather than a crash
    ratio = np.sqrt(((deg_a + 0.5) / (deg_b + 0.5)) * ((con_a + 1.0) / (con_b + 1.0)))
    if ratio < params.removed_ratio:
        return HINT_REMOVED
    if ratio > params.added_ratio:
        return HINT_ADDED
    return None


# --------------------------------------------------------------------------- public API

def extract_zones(score: np.ndarray, checked: np.ndarray, params: Params,
                  sig_before=None, sig_after=None, win: int = 8, step: int = 4) -> list[Zone]:
    """Turn a per-cell change score into ranked zones (see module docstring).

    `sig_before` / `sig_after` are optional objects with `.degree` int8[gh,gw,2] and
    `.contrast` float32[gh,gw] on the same grid as `score`; they enable the
    something_removed / something_added hints.
    """
    score = np.asarray(score, np.float32)
    checked = np.asarray(checked, bool)
    if score.shape != checked.shape:
        raise ValueError(f"score {score.shape} and checked {checked.shape} differ")
    score = np.where(np.isfinite(score), score, 0.0).astype(np.float32)
    changed = checked & (score >= params.threshold)
    if not changed.any():
        return []

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        changed.astype(np.uint8), connectivity=8)
    blobs = [
        _Blob(label=k,
              i0=int(stats[k, cv2.CC_STAT_TOP]), j0=int(stats[k, cv2.CC_STAT_LEFT]),
              i1=int(stats[k, cv2.CC_STAT_TOP] + stats[k, cv2.CC_STAT_HEIGHT]),
              j1=int(stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH]),
              n_cells=int(stats[k, cv2.CC_STAT_AREA]))
        for k in range(1, n_labels)
    ]

    # Merge nearby fragments. Small blobs take part too, so that a cluster of
    # drops can form one zone; isolated tiny blobs are dropped below.
    boxes = np.array([b.pixel_box(win, step) for b in blobs], np.int64)
    uf = _UnionFind(len(blobs))
    for a in range(len(blobs) - 1):
        rest = boxes[a + 1:]
        gx = np.maximum(0, np.maximum(boxes[a, 0], rest[:, 0]) - np.minimum(boxes[a, 2], rest[:, 2]))
        gy = np.maximum(0, np.maximum(boxes[a, 1], rest[:, 1]) - np.minimum(boxes[a, 3], rest[:, 3]))
        for b in np.nonzero(np.maximum(gx, gy) < params.merge_gap_px)[0]:
            uf.union(a, a + 1 + int(b))
    groups: dict[int, list[_Blob]] = {}
    for idx, blob in enumerate(blobs):
        groups.setdefault(uf.find(idx), []).append(blob)

    # Cheap per-blob score sums, so zones can be ranked before the pixel work.
    score_sum = np.bincount(labels.ravel(), weights=score.ravel(), minlength=n_labels)

    candidates: list[tuple[float, float, int, list[_Blob]]] = []
    for members in groups.values():
        n_cells = sum(b.n_cells for b in members)
        big_enough = any(b.n_cells >= params.min_zone_cells for b in members)
        # Keep a zone if one fragment is big enough, or if it is a dense cluster
        # of many small fragments (typical for water drops) totalling enough cells.
        if not (big_enough or (len(members) >= params.blob_min_count
                               and n_cells >= params.min_zone_cells)):
            continue
        mean_score = float(sum(score_sum[b.label] for b in members) / n_cells)
        candidates.append((n_cells * mean_score, mean_score, n_cells, members))
    candidates.sort(key=lambda c: c[0], reverse=True)

    zones: list[Zone] = []
    for rank, (rank_score, mean_score, n_cells, members) in enumerate(
            candidates[: params.max_zones], start=1):
        cells = np.isin(labels, [b.label for b in members])
        pix = _cells_to_pixels(cells, win, step)
        x, y, w, h = cv2.boundingRect(pix)
        if _water_hint(members, params):
            hint = HINT_WATER  # shape evidence takes precedence
        else:
            hint = _activity_hint(cells, sig_before, sig_after, params)
        zones.append(Zone(
            id=rank,
            bbox=(int(x), int(y), int(w), int(h)),
            n_cells=int(n_cells),
            area_px=int(np.count_nonzero(pix)),
            mean_score=mean_score,
            max_score=float(score[cells].max()),
            rank_score=float(rank_score),
            hint=hint,
            contour=_contour(pix),
        ))
    return zones


def coverage(checked: np.ndarray) -> float:
    """Fraction of grid cells that were actually checked (0.0 for an empty grid)."""
    checked = np.asarray(checked, bool)
    return float(checked.mean()) if checked.size else 0.0
