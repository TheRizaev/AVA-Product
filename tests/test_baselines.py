"""Tests for dalil.baselines using small local fixtures (no synth, no pbp)."""

import numpy as np
import pytest

from dalil import baselines
from dalil.config import Params

P = Params()  # win=8, step=4
METHODS = baselines.get_methods(include_pbp=False)


def _textured(h=64, w=96, seed=0):
    """Smooth random texture, float32 0..255."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(30, 220, (h // 4 + 1, w // 4 + 1))
    img = np.kron(small, np.ones((4, 4)))[:h, :w]
    img = img + rng.normal(0, 6, (h, w))
    return np.clip(img, 0, 255).astype(np.float32)


@pytest.mark.parametrize("name", sorted(METHODS))
def test_identical_images_zero(name):
    feat, dist = METHODS[name]
    g = _textured()
    a, b = feat(g, P), feat(g.copy(), P)
    d = dist(a, b)
    gh, gw = (64 - 8) // 4 + 1, (96 - 8) // 4 + 1
    assert d.shape == (gh, gw) and d.dtype == np.float32
    assert np.all(d < 1e-4)


@pytest.mark.parametrize("name", sorted(METHODS))
def test_pasted_rectangle_detected(name):
    feat, dist = METHODS[name]
    g = _textured(seed=1)
    h2 = g.copy()
    # Paste a patch of different texture over pixels [24:40, 40:64].
    h2[24:40, 40:64] = _textured(16, 24, seed=99)
    d = dist(feat(g, P), feat(h2, P))
    # Cells fully inside the rectangle: rows i with 24 <= 4i, 4i+8 <= 40 -> i in 6..8.
    inside = d[6:9, 10:15]
    # Cells far away: untouched.
    outside = np.concatenate([d[:3].ravel(), d[-3:].ravel()])
    assert inside.min() > 0.02
    assert outside.max() < 1e-4
    assert inside.mean() > 10 * (outside.mean() + 1e-3) or outside.max() == 0


@pytest.mark.parametrize("name", sorted(METHODS))
def test_shift_indexing(name):
    feat, dist = METHODS[name]
    g = _textured(seed=2)
    # Shift content by one cell (4 px) down and two cells (8 px) right:
    # a's cell (i, j) equals b's cell (i+1, j+2).
    shifted = np.zeros_like(g)
    shifted[4:, 8:] = g[:-4, :-8]
    a, b = feat(g, P), feat(shifted, P)
    d = dist(a, b, 1, 2)
    gh, gw = d.shape
    assert np.all(np.isinf(d[gh - 1:, :])) and np.all(np.isinf(d[:, gw - 2:]))
    valid = d[:gh - 1, :gw - 2]
    assert np.isfinite(valid).all()
    # Skip border cells: census bits there see zero fill / edge padding.
    assert valid[1:-1, 1:-1].max() < 1e-4
    # The unshifted comparison should differ.
    assert np.nanmean(dist(a, b, 0, 0)) > 0.01
    # Negative shift: b's cell (i, j) equals a's cell (i-1, j-2).
    d2 = dist(b, a, -1, -2)
    assert np.isinf(d2[0]).all() and np.isinf(d2[:, :2]).all()
    assert d2[2:-1, 3:-1].max() < 1e-4


def test_out_of_range_shift_all_inf():
    g = _textured()
    a = baselines.pixdiff_features(g, P)
    assert np.isinf(baselines.pixdiff_distance(a, a, 100, 0)).all()


def test_ssim_stats_match_blocks():
    g = _textured(seed=3)
    f = baselines.ssim_features(g, P)
    assert np.allclose(f.mean, f.blocks.mean(-1), atol=1e-3)
    assert np.allclose(f.var, f.blocks.var(-1), atol=0.05)


def test_census_brightness_invariant():
    g = _textured(seed=4)
    a = baselines.census_features(g, P)
    b = baselines.census_features(g * 0.7 + 10, P)
    assert baselines.census_distance(a, b).max() < 1e-6


def test_bgr_input_accepted():
    g = _textured().astype(np.uint8)
    bgr = np.dstack([g, g, g])
    f = baselines.pixdiff_features(bgr, P)
    assert f.blocks.shape[:2] == ((64 - 8) // 4 + 1, (96 - 8) // 4 + 1)


def test_methods_registry():
    assert {"ssim", "census", "pixdiff"} <= set(METHODS)
    for feat, dist in METHODS.values():
        assert callable(feat) and callable(dist)
