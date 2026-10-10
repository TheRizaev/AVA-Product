"""Tests for dalil.synth (small image sizes keep them fast)."""

import numpy as np
import pytest

from dalil import synth

H, W = 180, 320


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_surface_deterministic_and_shape():
    a = synth.make_surface(_rng(7), H, W)
    b = synth.make_surface(_rng(7), H, W)
    c = synth.make_surface(_rng(8), H, W)
    assert a.shape == (H, W, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # Not a flat image: has some structure.
    assert a.std() > 3


@pytest.mark.parametrize("fn", [synth.add_scratch, synth.add_tape,
                                synth.add_sticker, synth.add_water_drops])
def test_defects(fn):
    base = synth.make_surface(_rng(1), H, W)
    before = base.copy()
    out1, m1 = fn(base, _rng(2))
    out2, m2 = fn(base, _rng(2))
    assert np.array_equal(base, before), "input must not be modified"
    assert np.array_equal(out1, out2) and np.array_equal(m1, m2)
    assert out1.shape == base.shape and out1.dtype == np.uint8
    assert m1.shape == (H, W) and m1.dtype == bool
    assert m1.any()
    # Pixels inside the mask changed; pixels far outside did not.
    diff = np.abs(out1.astype(int) - base.astype(int)).sum(-1)
    assert diff[m1].mean() > 5
    ys, xs = np.nonzero(m1)
    far = np.ones((H, W), bool)
    far[max(ys.min() - 4, 0):ys.max() + 5, max(xs.min() - 4, 0):xs.max() + 5] = False
    assert diff[far].max() == 0


@pytest.mark.parametrize("kind", synth.LIGHTING_KINDS)
def test_lighting(kind):
    img = synth.make_surface(_rng(3), H, W)
    out = synth.apply_lighting(img, _rng(4), kind)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - img.astype(int)).mean() > 1.0
    assert np.array_equal(out, synth.apply_lighting(img, _rng(4), kind))


def test_flash_saturates():
    img = synth.make_surface(_rng(3), H, W)
    out = synth.apply_lighting(img, _rng(5), "flash")
    assert (out == 255).all(axis=-1).any()


def test_unknown_lighting():
    with pytest.raises(ValueError):
        synth.apply_lighting(np.zeros((8, 8, 3), np.uint8), _rng(), "disco")


def test_homography_and_warp():
    Hm = synth.random_homography(_rng(6), H, W)
    assert Hm.shape == (3, 3) and abs(Hm[2, 2] - 1) < 1e-9
    # Corners move by at most a few pixels.
    corners = np.array([[0, 0, 1], [W - 1, 0, 1], [0, H - 1, 1], [W - 1, H - 1, 1]], float).T
    moved = Hm @ corners
    moved = moved[:2] / moved[2]
    assert np.abs(moved - corners[:2]).max() < 6 + 0.02 * W + 0.03 * W + 2
    img = synth.make_surface(_rng(6), H, W)
    out = synth.warp(img, Hm)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(synth.warp(img, np.eye(3)), img)


@pytest.mark.parametrize("change,lighting,motion", [
    (None, None, False),
    ("scratch", "gamma", True),
    ("tape", "shadow", False),
    ("sticker", "flash", True),
    ("water", "dusk", True),
])
def test_make_pair(change, lighting, motion):
    p = synth.make_pair(_rng(9), change, lighting, motion, h=H, w=W)
    q = synth.make_pair(_rng(9), change, lighting, motion, h=H, w=W)
    assert set(p) == {"before", "after", "change_mask_before_frame", "meta"}
    assert p["before"].shape == p["after"].shape == (H, W, 3)
    assert p["before"].dtype == p["after"].dtype == np.uint8
    mask = p["change_mask_before_frame"]
    assert mask.shape == (H, W) and mask.dtype == bool
    assert mask.any() == (change is not None)
    assert np.array_equal(p["after"], q["after"]) and np.array_equal(p["before"], q["before"])
    Hm = p["meta"]["H"]
    assert Hm.shape == (3, 3)
    assert np.allclose(Hm, np.eye(3)) == (not motion)
    assert p["meta"]["change"] == change and p["meta"]["lighting"] == lighting


def test_make_pair_no_change_is_close():
    """Without change/lighting/motion the pair differs only by sensor noise."""
    p = synth.make_pair(_rng(11), None, None, False, h=H, w=W)
    d = p["after"].astype(float) - p["before"].astype(float)
    assert abs(d.mean()) < 1.0 and d.std() < 8.0
