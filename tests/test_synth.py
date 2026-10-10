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


# --------------------------------------------------------------------------- #
# QA station data
# --------------------------------------------------------------------------- #

import cv2  # noqa: E402

SH, SW = 180, 320  # station tests: small frames keep them fast


@pytest.mark.parametrize("kind", synth.PART_KINDS)
def test_make_part_deterministic_and_shape(kind):
    a = synth.make_part(_rng(21), kind, SH, SW)
    b = synth.make_part(_rng(21), kind, SH, SW)
    c = synth.make_part(_rng(22), kind, SH, SW)
    assert a.shape == (SH, SW, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    # make_part is exactly the composite of its layers.
    layers = synth.make_part_layers(_rng(21), kind, SH, SW)
    assert layers["kind"] == kind and layers["background"] in synth.BACKGROUND_KINDS
    assert np.array_equal(synth._to_u8(synth.compose_layers(layers)), a)
    # The part's bounding box covers 40-80 % of the frame and lies inside it.
    ys, xs = np.nonzero(layers["outline"] > 0.5)
    frac = (np.ptp(xs) + 1) * (np.ptp(ys) + 1) / (SH * SW)
    assert 0.4 <= frac <= 0.8
    assert ys.min() > 0 and xs.min() > 0 and ys.max() < SH - 1 and xs.max() < SW - 1
    for x, y, r in layers["holes"]:
        assert 0 <= x < SW and 0 <= y < SH and r > 0


def test_make_part_random_kind_and_unknown():
    img = synth.make_part(_rng(1), None, SH, SW)
    assert img.shape == (SH, SW, 3)
    with pytest.raises(ValueError):
        synth.make_part(_rng(1), "forged", SH, SW)


def _orb_count(img):
    return len(cv2.ORB_create(2000).detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None))


def test_parts_texture_levels():
    """ORB finds plenty of features on metal parts; painted/molded are low-texture."""
    counts = {k: _orb_count(synth.make_part(_rng(3), k, 360, 640)) for k in synth.PART_KINDS}
    assert counts["stamped"] >= 100 and counts["machined"] >= 100, counts
    assert min(counts["painted"], counts["molded"]) < counts["stamped"] / 2, counts


@pytest.mark.parametrize("fn", [synth.add_dent, synth.add_missing_hole,
                                synth.add_burr, synth.add_stain])
def test_station_defects_on_surface(fn):
    """Same contract as the legacy defects, on the car surface used by test_defects."""
    base = synth.make_surface(_rng(1), H, W)
    before = base.copy()
    out1, m1 = fn(base, _rng(2))
    out2, m2 = fn(base, _rng(2))
    assert np.array_equal(base, before), "input must not be modified"
    assert np.array_equal(out1, out2) and np.array_equal(m1, m2)
    assert out1.shape == base.shape and out1.dtype == np.uint8
    assert m1.shape == (H, W) and m1.dtype == bool and m1.any()
    diff = np.abs(out1.astype(int) - base.astype(int)).sum(-1)
    assert diff[m1].mean() > 5
    ys, xs = np.nonzero(m1)
    far = np.ones((H, W), bool)
    far[max(ys.min() - 4, 0):ys.max() + 5, max(xs.min() - 4, 0):xs.max() + 5] = False
    assert diff[far].max() == 0


@pytest.mark.parametrize("kind", synth.PART_KINDS)
def test_station_defects_on_parts(kind):
    """Every station defect lands on/at the part and is visible."""
    part = synth.make_part_layers(_rng(30), kind, SH, SW)["outline"] > 0.5
    near_part = cv2.dilate(part.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    for name in synth.STATION_DEFECTS:
        p = synth.make_station_pair(_rng(30), kind, name, None, "none", h=SH, w=SW)
        m = p["mask_ref_frame"]
        assert m.any(), name
        assert (m & near_part).sum() >= 0.5 * m.sum(), name
        clean = p["meta"]["reference_clean"].astype(int)
        # Frame = defected scene + noise; the change must stand out of the noise.
        assert np.abs(p["frame"].astype(int) - clean).sum(-1)[m].mean() > 10, name


def test_missing_hole_uses_holes():
    layers = synth.make_part_layers(_rng(4), "machined", SH, SW)
    img = synth._to_u8(synth.compose_layers(layers))
    out, m = synth.add_missing_hole(img, _rng(5), holes=layers["holes"])
    assert any(m[int(y), int(x)] for x, y, _ in layers["holes"])
    # The through hole showed background; now it is filled with part-like colour.
    assert np.abs(out.astype(int) - img.astype(int))[m].mean() > 5


@pytest.mark.parametrize("kind", synth.FACTORY_LIGHTING)
def test_factory_lighting_changes_pixels(kind):
    img = synth.make_part(_rng(6), "stamped", SH, SW)
    out = synth.apply_factory_lighting(img, _rng(7), kind)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - img.astype(int)).mean() > 1.0
    assert np.array_equal(out, synth.apply_factory_lighting(img, _rng(7), kind))
    # strength=0 is (nearly) the identity.
    weak = synth.apply_factory_lighting(img, _rng(7), kind, strength=0.0)
    assert np.abs(weak.astype(int) - img.astype(int)).max() <= 1


def test_factory_lighting_kinds_partition():
    assert set(synth.MONOTONE_LIGHTING) | set(synth.NON_MONOTONE_LIGHTING) == set(synth.FACTORY_LIGHTING)
    assert not set(synth.MONOTONE_LIGHTING) & set(synth.NON_MONOTONE_LIGHTING)
    with pytest.raises(ValueError):
        synth.apply_factory_lighting(np.zeros((8, 8, 3), np.uint8), _rng(), "disco")


def _order_reversals(img, out, min_gap, win=6, step=3):
    """Count pixel pairs inside win x win windows (per channel) whose order flips.

    A pair counts when the original values differ by more than ``min_gap`` and the
    new values are strictly in the opposite order.
    """
    a = np.lib.stride_tricks.sliding_window_view(img.astype(np.int16), (win, win), (0, 1))
    b = np.lib.stride_tricks.sliding_window_view(out.astype(np.int16), (win, win), (0, 1))
    a = a[::step, ::step].reshape(-1, 3, win * win)
    b = b[::step, ::step].reshape(-1, 3, win * win)
    da = a[..., :, None] - a[..., None, :]
    db = b[..., :, None] - b[..., None, :]
    return int(((da > min_gap) & (db < 0)).sum())


@pytest.mark.parametrize("kind", synth.MONOTONE_LIGHTING)
@pytest.mark.parametrize("part", ["stamped", "molded"])
def test_monotone_lighting_preserves_local_order(kind, part):
    """Spot check on a noise-free part: monotone kinds never flip the local order.

    Global kinds are exact per-channel monotone maps (gap 0). window_gradient is a
    smooth spatial gain, so it may only flip near-ties (gap 3 grey levels).
    """
    img = synth.make_part(_rng(8), part, SH, SW)
    gap = 3 if kind == "window_gradient" else 0
    for seed in range(3):
        out = synth.apply_factory_lighting(img, _rng(100 + seed), kind)
        assert _order_reversals(img, out, gap) == 0


@pytest.mark.parametrize("kind", synth.NON_MONOTONE_LIGHTING)
def test_non_monotone_lighting_flips_order(kind):
    """The expected failure cases do break the local order somewhere."""
    img = synth.make_part(_rng(8), "stamped", SH, SW)
    flips = sum(_order_reversals(img, synth.apply_factory_lighting(img, _rng(s), kind), 3)
                for s in range(3))
    assert flips > 0


@pytest.mark.parametrize("motion", synth.MOTION_MODES)
@pytest.mark.parametrize("defect,lighting", [(None, None), ("scratch", "gamma"),
                                             ("tape", "window_gradient"), ("burr", "hand_shadow")])
def test_make_station_pair(motion, defect, lighting):
    p = synth.make_station_pair(_rng(40), "stamped", defect, lighting, motion, h=SH, w=SW)
    q = synth.make_station_pair(_rng(40), "stamped", defect, lighting, motion, h=SH, w=SW)
    assert set(p) == {"reference", "frame", "mask_ref_frame", "meta"}
    ref, frame, mask, meta = p["reference"], p["frame"], p["mask_ref_frame"], p["meta"]
    assert ref.shape == frame.shape == (SH, SW, 3) and ref.dtype == frame.dtype == np.uint8
    assert mask.shape == (SH, SW) and mask.dtype == bool
    assert mask.any() == (defect is not None)
    assert np.array_equal(ref, q["reference"]) and np.array_equal(frame, q["frame"])
    assert np.array_equal(mask, q["mask_ref_frame"])
    Hm = meta["H"]
    assert Hm.shape == (3, 3)
    assert np.allclose(Hm, np.eye(3)) == (motion == "none")
    assert meta["motion_scope"] == {"none": "none", "jitter": "camera", "shift": "part"}[motion]
    assert meta["part_kind"] == "stamped" and meta["defect"] == defect
    assert meta["lighting"] == lighting
    assert meta["lighting_monotone"] == (None if lighting is None else lighting in synth.MONOTONE_LIGHTING)
    assert meta["part_mask"].shape == (SH, SW) and meta["part_mask"].any()
    assert meta["mask_frame"].shape == (SH, SW)
    assert meta["mask_frame"].any() == (defect is not None)
    assert meta["reference_clean"].shape == (SH, SW, 3)


def _moved(Hm, x, y):
    v = Hm @ np.array([x, y, 1.0])
    return v[:2] / v[2]


def test_station_pair_motion_magnitudes():
    k = min(SH, SW) / 720.0
    corners = [(0, 0), (SW - 1, 0), (0, SH - 1), (SW - 1, SH - 1)]
    for seed in range(5):
        j = synth.make_station_pair(_rng(seed), "machined", motion="jitter", h=SH, w=SW)["meta"]["H"]
        sm = synth.make_station_pair(_rng(seed), "machined", motion="shift", h=SH, w=SW)["meta"]
        s = sm["H"]
        jd = max(np.hypot(*(_moved(j, x, y) - (x, y))) for x, y in corners)
        assert jd <= 2 * k * 1.5 + np.radians(0.3) * SW  # +-2 px shift, +-0.3 deg
        assert abs(np.degrees(np.arctan2(s[1, 0], s[0, 0]))) <= 3.0 + 1e-6
        assert np.isclose(np.linalg.det(s[:2, :2]), 1.0)  # rigid: no scale
        ys, xs = np.nonzero(sm["part_mask"])
        c = np.array([xs.mean(), ys.mean()])
        assert np.hypot(*(_moved(s, *c) - c)) <= 40 * k + 1.0  # part centre moves <= 40 px


def test_station_pair_shift_moves_part_not_background():
    """With motion='shift' the background is static and the part follows H."""
    p = synth.make_station_pair(_rng(50), "stamped", None, None, "shift", h=SH, w=SW)
    meta = p["meta"]
    ref, frame = meta["reference_clean"].astype(float), p["frame"].astype(float)
    part_ref, Hm = meta["part_mask"], meta["H"]
    assert not np.allclose(Hm, np.eye(3))
    part_frame = cv2.warpPerspective(part_ref.astype(np.uint8), Hm, (SW, SH),
                                     flags=cv2.INTER_NEAREST) > 0
    # Background far from both part poses (and the drop shadow) is unchanged up to noise.
    near = cv2.dilate((part_ref | part_frame).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    bg = ~near
    assert bg.sum() > 1000
    assert np.abs(frame - ref)[bg].mean() < 4.0
    # Part interior: the frame matches the reference warped by H (not the unwarped one).
    warped = cv2.warpPerspective(meta["reference_clean"], Hm, (SW, SH)).astype(float)
    inner = cv2.erode(part_frame.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    err_h = np.abs(frame - warped)[inner].mean()
    err_id = np.abs(frame - ref)[inner].mean()
    assert err_h < 5.0 and err_h < err_id


def test_station_pair_mask_maps_with_h():
    """The defect seen in the frame is where meta['mask_frame'] says it is."""
    p = synth.make_station_pair(_rng(60), "machined", "sticker", None, "shift", h=SH, w=SW)
    meta = p["meta"]
    mf = cv2.erode(meta["mask_frame"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    clean_warped = cv2.warpPerspective(meta["reference_clean"], meta["H"], (SW, SH)).astype(float)
    d = np.abs(p["frame"].astype(float) - clean_warped).sum(-1)
    assert mf.any() and d[mf].mean() > 30


def test_station_pair_validation():
    with pytest.raises(ValueError):
        synth.make_station_pair(_rng(), "stamped", "crack", h=SH, w=SW)
    with pytest.raises(ValueError):
        synth.make_station_pair(_rng(), "stamped", lighting="disco", h=SH, w=SW)
    with pytest.raises(ValueError):
        synth.make_station_pair(_rng(), "stamped", motion="fly", h=SH, w=SW)
    with pytest.raises(ValueError):
        synth.make_station_pair(_rng(), "forged", h=SH, w=SW)


def test_make_reference_set():
    base = synth.make_part(_rng(70), "painted", SH, SW)
    shots = synth.make_reference_set(_rng(71), base, 3)
    again = synth.make_reference_set(_rng(71), base, 3)
    assert len(shots) == 3
    for s, t in zip(shots, again):
        assert s.shape == base.shape and s.dtype == np.uint8 and np.array_equal(s, t)
    # Shot 0 is only noisy; all shots stay close to the base but differ from each other.
    d0 = shots[0].astype(float) - base
    assert abs(d0.mean()) < 1.0 and 0.5 < d0.std() < 5.0
    for s in shots[1:]:
        assert not np.array_equal(s, shots[0])
        assert np.abs(s.astype(float) - base).mean() < 8.0


# --------------------------------------------------------------------------- Mode 1 surfaces

@pytest.mark.parametrize("kind", synth.SURFACE_KINDS)
def test_uniform_surface_fills_frame_and_is_deterministic(kind):
    a = synth.make_uniform_surface(_rng(40), kind, SH, SW)
    b = synth.make_uniform_surface(_rng(40), kind, SH, SW)
    c = synth.make_uniform_surface(_rng(41), kind, SH, SW)
    assert a.shape == (SH, SW, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    g = a.astype(np.float32).mean(axis=2)
    # The whole frame is the surface: no background, no part edges (every 1/8 tile has a
    # similar level), not clipped, exposed above the "dark" gate.
    th, tw = SH // 4, SW // 8
    tiles = np.array([g[i:i + th, j:j + tw].mean() for i in range(0, SH - th + 1, th)
                      for j in range(0, SW - tw + 1, tw)])
    assert tiles.min() > 0.7 * tiles.max()
    assert 45 < g.mean() < 200 and (a == 255).mean() < 0.001 and (a == 0).mean() < 0.001
    # Mild soft illumination fall-off: smooth large-scale variation present but bounded.
    low = cv2.resize(g, (16, 9), interpolation=cv2.INTER_AREA)
    assert 0.02 < (low.max() - low.min()) / low.mean() < 0.35
    # Texture: fine detail exists (not a flat fill), amount depends on the kind.
    hp = g - cv2.GaussianBlur(g, (0, 0), 4.0)
    assert hp.std() > 0.8


def test_uniform_surface_kinds_differ_and_validate():
    stds = {}
    for kind in synth.SURFACE_KINDS:
        g = synth.make_uniform_surface(_rng(3), kind, SH, SW).astype(np.float32).mean(axis=2)
        stds[kind] = float((g - cv2.GaussianBlur(g, (0, 0), 4.0)).std())
    assert stds["glass_dark"] < stds["painted"] < stds["brushed"]
    g = synth.make_uniform_surface(_rng(3), "brushed", SH, SW).astype(np.float32).mean(axis=2)
    # Brushed metal is directional: much more variation across the streaks than along them.
    dx = np.abs(np.diff(g, axis=1)).mean()
    dy = np.abs(np.diff(g, axis=0)).mean()
    assert max(dx, dy) > 1.5 * min(dx, dy)
    with pytest.raises(ValueError):
        synth.make_uniform_surface(_rng(0), "wood", SH, SW)


@pytest.mark.parametrize("kind", synth.SURFACE_KINDS)
def test_add_chip(kind):
    base = synth.make_uniform_surface(_rng(12), kind, SH, SW)
    before = base.copy()
    out, mask = synth.add_chip(base, _rng(13))
    again, mask2 = synth.add_chip(base, _rng(13))
    assert np.array_equal(base, before)                      # input untouched
    assert np.array_equal(out, again) and np.array_equal(mask, mask2)
    assert out.shape == base.shape and out.dtype == np.uint8 and mask.dtype == bool
    k = SH / 720.0
    assert 3 <= mask.sum() <= np.pi * (12 * 1.5 * k + 5) ** 2 * 1.3   # small
    d = np.abs(out.astype(np.float32) - base).sum(-1)
    assert d[mask].mean() > 60                                # visible against the surface
    far = ~cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert d[far].max() == 0                                  # local change only


def test_add_chip_region():
    base = synth.make_uniform_surface(_rng(14), "painted", SH, SW)
    region = np.zeros((SH, SW), bool)
    region[20:50, 30:70] = True
    _, mask = synth.add_chip(base, _rng(15), region=region)
    ys, xs = np.nonzero(mask)
    assert region[int(round(ys.mean())), int(round(xs.mean()))] or \
        np.hypot(ys.mean() - 35, xs.mean() - 50) < 30
