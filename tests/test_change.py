"""Tests for dalil.change: shift tolerance, lighting robustness, defects, multi-reference."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from dalil import baselines, change, pbp
from dalil.config import Params

P = Params()          # win=8, step=4, tolerance=1
# A clear change in a raw (unsmoothed) PBP change map. The operating thresholds in Params are
# calibrated on whole inspections by eval/benchmark.py; these unit tests check the maps.
CLEAR = 0.45


def _texture(h=96, w=128, seed=0):
    """Smooth random part-like texture, float32 in ~20..235 (no flat windows)."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(20, 235, (h // 4 + 1, w // 4 + 1))
    img = cv2.resize(small, (w + 4, h + 4), interpolation=cv2.INTER_CUBIC)[:h, :w]
    return cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.float32), (0, 0), 1.0)


def _sig(img, params=P):
    return pbp.window_signatures(np.asarray(img, np.float32), params)


def _shift_right(img, px):
    out = np.empty_like(img)
    out[:, px:] = img[:, :-px]
    out[:, :px] = img[:, :1]
    return out


# --------------------------------------------------------------------------- #
# Tolerance to small misregistration
# --------------------------------------------------------------------------- #

def test_one_cell_shift_absorbed_by_tolerance():
    g = _texture()
    a, b = _sig(g), _sig(_shift_right(g, P.step))            # one cell = 4 px
    tol1 = change.tolerant_change_map(pbp.pair_distance, a, b, 1)
    tol0 = change.tolerant_change_map(pbp.pair_distance, a, b, 0)
    inner = (slice(None), slice(2, -2))                       # skip the exposed border columns
    assert tol1[inner].max() < 0.02
    assert tol0[inner].mean() > 0.3
    assert tol1.dtype == np.float32 and tol1.shape == a.flat.shape


def test_identical_is_zero_and_range():
    a = _sig(_texture(seed=1))
    m = change.tolerant_change_map(pbp.pair_distance, a, a, 2)
    assert np.all(m == 0)
    b = _sig(_texture(seed=2))
    m = change.tolerant_change_map(pbp.pair_distance, a, b, 1)
    assert np.all(np.isfinite(m)) and m.min() >= 0 and m.max() <= 1


# --------------------------------------------------------------------------- #
# Lighting: monotone brightness changes
# --------------------------------------------------------------------------- #

def test_gain_change_is_near_zero():
    g = _texture(seed=3)
    m = change.pbp_change_map(_sig(g), _sig(0.6 * g + 5.0), P)
    assert m.max() < 1e-3


@pytest.mark.parametrize("gamma", [0.6, 1.8])
def test_gamma_change_stays_below_threshold(gamma):
    # The monomial support is gamma-invariant; the normalised coefficients move a
    # little, so the score stays well below the decision threshold.
    g = _texture(seed=4)
    m = change.pbp_change_map(_sig(g), _sig(255.0 * (g / 255.0) ** gamma), P)
    assert m.max() < P.threshold
    assert m.mean() < 0.12


def test_gain_change_breaks_plain_subtraction():
    """Contrast for the demo: the gain change that PBP ignores is a large pixdiff score.

    It is of the order of the calibrated pixdiff threshold itself: plain subtraction must
    choose between flagging lighting changes and missing defects (eval/RESULTS.md).
    """
    g = _texture(seed=3)
    feat, dist = baselines.pixdiff_features, baselines.pixdiff_distance
    m = change.tolerant_change_map(dist, feat(g, P), feat(0.6 * g + 5.0, P), 1)
    assert m.mean() > 0.5 * P.threshold_pixdiff
    assert change.pbp_change_map(_sig(g), _sig(0.6 * g + 5.0), P).max() < 1e-3


# --------------------------------------------------------------------------- #
# Defects
# --------------------------------------------------------------------------- #

def test_pasted_line_and_rectangle_detected():
    g = _texture(seed=5)
    frame = g.copy()
    cv2.line(frame, (20, 70), (100, 78), 250.0, 2)            # thin bright scratch
    frame[16:36, 72:104] = _texture(20, 32, seed=99)          # pasted foreign patch
    m = change.pbp_change_map(_sig(g), _sig(frame), P)

    line_mask = np.zeros(g.shape, np.uint8)
    cv2.line(line_mask, (20, 70), (100, 78), 1, 2)
    line_cells = pbp.cell_mask_from_pixel_mask(line_mask > 0, P.win, P.step, frac=0.05)
    rect_cells = np.zeros_like(line_cells)
    rect_cells[5:7, 19:24] = True                             # windows fully inside the patch
    touched = pbp.cell_mask_from_pixel_mask(
        cv2.dilate((np.abs(frame - g) > 0).astype(np.uint8), np.ones((9, 9))) > 0,
        P.win, P.step, frac=0.0001)

    assert np.median(m[line_cells]) > CLEAR
    # Foreign texture vs texture: the patch as a whole is a clear change, while a single
    # window of one random texture can resemble the other one somewhat.
    assert np.median(m[rect_cells]) > CLEAR
    assert m[rect_cells].min() > 0.8 * CLEAR
    assert m[~touched].max() < 1e-3


# --------------------------------------------------------------------------- #
# Multi-reference
# --------------------------------------------------------------------------- #

def test_multi_reference_takes_the_best_reference():
    ref1, ref2 = _texture(seed=6), _texture(seed=7)
    frame = ref2 + np.random.default_rng(0).normal(0, 0.5, ref2.shape).astype(np.float32)
    refs = [_sig(ref1), _sig(ref2)]
    f = _sig(frame)
    single = change.tolerant_change_map(pbp.pair_distance, refs[0], f, 1)
    multi = change.multi_reference_change_map(pbp.pair_distance, refs, f, 1)
    assert single.mean() > CLEAR
    assert multi.mean() < 0.05 and multi.max() < 0.5 * CLEAR   # sensor noise only
    np.testing.assert_array_equal(multi, np.minimum(
        single, change.tolerant_change_map(pbp.pair_distance, refs[1], f, 1)))
    np.testing.assert_array_equal(change.pbp_change_map(refs, f, P), multi)
    with pytest.raises(ValueError):
        change.multi_reference_change_map(pbp.pair_distance, [], f, 1)


# --------------------------------------------------------------------------- #
# Generic dist_fn contract
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method", ["pbp", "pixdiff", "census", "ssim"])
@pytest.mark.parametrize("tol", [0, 1, 2])
def test_symmetric_fast_path_equals_explicit(method, tol):
    feat, dist = baselines.get_methods()[method]
    g = _texture(seed=8)
    frame = _shift_right(g, 3)
    frame[40:60, 50:80] = 30.0
    a, b = feat(g, P), feat(frame, P)
    fast = change.tolerant_change_map(dist, a, b, tol)                  # auto-detected
    slow = change.tolerant_change_map(dist, a, b, tol, symmetric=False)
    np.testing.assert_array_equal(fast, slow)


def test_dist_fn_called_once_per_shift():
    calls = []

    def dist(a, b, dy=0, dx=0):
        calls.append((dy, dx))
        return pbp.pair_distance(a, b, dy, dx)

    a = b = _sig(_texture(seed=9))
    change.tolerant_change_map(dist, a, b, 1)
    assert len(calls) == 2 * 9                                 # unknown fn: both directions
    calls.clear()
    dist.symmetric = True
    change.tolerant_change_map(dist, a, b, 2)
    assert len(calls) == 25                                    # declared symmetric: one direction


def test_different_grids_clamp_out_of_range_cells():
    g = _texture(seed=10)
    a, b = _sig(g), _sig(g[:80, :112])                         # b's grid is smaller
    m = change.tolerant_change_map(pbp.pair_distance, a, b, 1)
    gh_b, gw_b = b.flat.shape
    assert m.shape == a.flat.shape and np.all(np.isfinite(m))
    assert np.all(m[gh_b + 1:, :] == 1.0) and np.all(m[:, gw_b + 1:] == 1.0)
    assert m[:gh_b - 1, :gw_b - 1].max() < 1e-6


def test_pbp_change_map_uses_params_tolerance():
    g = _texture(seed=11)
    a, b = _sig(g), _sig(_shift_right(g, 2 * P.step))         # two cells
    inner = (slice(None), slice(3, -3))
    assert change.pbp_change_map(a, b, replace(P, tolerance=2))[inner].max() < 0.02
    assert change.pbp_change_map(a, b, replace(P, tolerance=1))[inner].mean() > 0.3
