"""Tests for ava.surface (Mode 1: single-photo surface check) on synthetic surfaces."""

import functools
import time

import cv2
import numpy as np
import pytest

from ava import pbp, synth, surface
from ava.config import Params
from ava.pipeline import result_to_json

P = Params()
SEEDS = (0, 1, 2)
MONOTONE = (None, "gamma", "drift_gain", "window_gradient", "auto_exposure")
DEFECTS = {
    "scratch": synth.add_scratch,
    "chip": synth.add_chip,
    "stain": synth.add_stain,
    "sticker": synth.add_sticker,
}
# Mild lighting on defect photos too (no saturation of the defect itself).
DEFECT_LIGHTING = (None, "gamma", "window_gradient")


@functools.lru_cache(maxsize=None)
def _surface_cached(kind, seed):
    img = synth.make_uniform_surface(np.random.default_rng(1000 + 17 * seed), kind)
    img.setflags(write=False)  # shared between tests: never modified in place
    return img


def _surface(kind, seed):
    return _surface_cached(kind, seed)


def _mask_at_work(mask, res, dilate_px=8):
    m = cv2.resize(mask.astype(np.uint8), tuple(res.work_size), interpolation=cv2.INTER_NEAREST)
    if dilate_px:
        m = cv2.dilate(m, np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8))
    return m > 0


def _bbox_hits(bbox, mask):
    x, y, w, h = bbox
    return bool(mask[y:y + h, x:x + w].any())


@pytest.mark.parametrize("kind", synth.SURFACE_KINDS)
def test_clean_surface_passes_under_monotone_lighting(kind):
    for seed in SEEDS[:2]:
        base = _surface(kind, seed)
        for light in MONOTONE:
            img = base if light is None else synth.apply_factory_lighting(
                base, np.random.default_rng(seed + 50), light)
            res = surface.scan_surface(img, P)
            assert res.verdict == "pass", (kind, seed, light, res.reason,
                                           [(z.bbox, z.mean_score) for z in res.zones])
            assert res.zones == [] and res.reason is None
            assert res.method == "pbp_surface" and res.align == {}


# Known weak case (kept visible instead of tuned away): a dried-water stain is a thin, soft
# bright ring; on brushed metal the window contrast of the streaks varies so much that the
# ring is neither a contrast outlier nor structurally rare. Oil stains are found.
KNOWN_WEAK = {("brushed", "stain"): 1}


@pytest.mark.parametrize("kind", synth.SURFACE_KINDS)
@pytest.mark.parametrize("defect", list(DEFECTS))
def test_defect_is_found(kind, defect):
    found = []
    for seed in SEEDS:
        rng = np.random.default_rng(2000 + 31 * seed + len(defect))
        base = _surface(kind, seed)
        img, mask = DEFECTS[defect](base, rng)
        light = DEFECT_LIGHTING[seed % len(DEFECT_LIGHTING)]
        if light is not None:
            img = synth.apply_factory_lighting(img, np.random.default_rng(seed + 70), light)
        res = surface.scan_surface(img, P)
        assert res.verdict in ("pass", "fail"), (kind, defect, seed, res.reason)
        hit = res.verdict == "fail" and _bbox_hits(res.zones[0].bbox, _mask_at_work(mask, res))
        if res.verdict == "fail":  # whatever was flagged must be the defect, not the surface
            assert hit, (kind, defect, seed, light, [z.bbox for z in res.zones])
        found.append(hit)
    need = KNOWN_WEAK.get((kind, defect), len(SEEDS))
    assert sum(found) >= need, (kind, defect, found)


@pytest.mark.parametrize("kind", synth.SURFACE_KINDS)
def test_specular_highlight_is_not_a_defect(kind):
    for seed in SEEDS[:2]:
        img = synth.apply_factory_lighting(_surface(kind, seed), np.random.default_rng(seed + 90),
                                           "specular")
        res = surface.scan_surface(img, P)
        # Either the highlight (and its halo) is left unchecked, or the photo is rejected.
        assert (res.verdict == "pass") or (res.verdict == "retake" and res.reason == "glare"), \
            (kind, seed, res.verdict, res.reason, [z.bbox for z in res.zones])
        if res.verdict == "pass":
            assert res.coverage < 0.95  # the highlight was excluded, not "checked and fine"


def test_defect_next_to_highlight_still_found():
    base = _surface("painted", 3)
    img, mask = synth.add_sticker(base, np.random.default_rng(5))
    img = synth.apply_factory_lighting(img, np.random.default_rng(91), "specular")
    res = surface.scan_surface(img, P)
    m = _mask_at_work(mask, res)
    if res.verdict == "fail":
        assert any(_bbox_hits(z.bbox, m) for z in res.zones)
    else:  # the sticker can only be missed if the glare region covers it
        assert res.verdict == "pass" and not (res.checked & pbp.cell_mask_from_pixel_mask(
            _mask_at_work(mask, res, 0), P.win, P.step)).any()


def test_glare_region_flags_highlight_and_halo():
    img = synth.apply_factory_lighting(_surface("glass_dark", 0), np.random.default_rng(90),
                                       "specular")
    img, _ = surface.resize_max_side(img, P.work_max_side)
    g = surface._gray(img)
    region = surface.glare_region(img, g, P)
    core = surface.quality.glare_mask(img, P)
    assert core.any() and region[core].all()
    assert region.sum() > 1.5 * core.sum()          # the halo is included
    clean, _ = surface.resize_max_side(_surface("glass_dark", 0), P.work_max_side)
    assert not surface.glare_region(clean, surface._gray(clean), P).any()


def test_large_glare_is_retake():
    img = _surface("plastic", 0).copy()
    h, w = img.shape[:2]
    img[:, : int(0.6 * w)] = 255                    # blown-out left 60 %
    res = surface.scan_surface(img, P)
    assert res.verdict == "retake" and res.reason == "glare"
    assert "change_angle" in res.hints


def test_dark_and_blurry_are_retake():
    dark = (_surface("plastic", 1).astype(np.float32) * 0.25).astype(np.uint8)
    assert surface.scan_surface(dark, P).reason == "dark"
    blurry = cv2.GaussianBlur(_surface("painted", 1), (0, 0), 4.0)
    res = surface.scan_surface(blurry, P)
    assert res.verdict == "retake" and res.reason == "blurry"


def test_deterministic_and_json():
    img, _ = synth.add_scratch(_surface("brushed", 0), np.random.default_rng(3))
    a = surface.scan_surface(img, P)
    b = surface.scan_surface(img.copy(), P)
    assert a.signature_sha256 == b.signature_sha256 and len(a.signature_sha256) == 64
    assert [(z.bbox, z.n_cells, z.mean_score) for z in a.zones] == \
           [(z.bbox, z.n_cells, z.mean_score) for z in b.zones]
    assert np.array_equal(a.score, b.score)
    j = result_to_json(a)
    assert j["method"] == "pbp_surface" and j["verdict"] == a.verdict
    assert j["work_size"] == [1024, 576] and j["align"] == {}
    for img_bytes in a.images.values():
        assert img_bytes[:2] == b"\xff\xd8"          # JPEG


def test_score_map_methods_and_flat_handling():
    g = np.full((96, 160), 120.0, np.float32)       # perfectly flat (glossy paint)
    for m in surface.SURFACE_METHODS:
        s = surface.surface_score_map(g, P, m)
        assert s.shape == pbp.grid_shape(96, 160, P.win, P.step)
        assert s.dtype == np.float32 and np.all(s == 0), m
    # A dark line on the flat surface scores high (non-flat windows among flat ones).
    g2 = g.copy()
    cv2.line(g2, (20, 48), (140, 52), 40.0, 2)
    g2 = cv2.GaussianBlur(g2, (0, 0), P.blur_sigma)
    for m in surface.SURFACE_METHODS:
        s = surface.surface_score_map(g2, P, m)
        on_line = s[10:14, 8:30].max()
        assert on_line >= surface.method_threshold(m, P), (m, on_line)
        assert s[:4].max() == 0 or m == "gradient"
    with pytest.raises(ValueError):
        surface.surface_score_map(g, P, "ssim")


def test_baselines_run_through_scan():
    img, mask = synth.add_sticker(_surface("plastic", 2), np.random.default_rng(9))
    for m in ("census", "gradient"):
        res = surface.scan_surface(img, P, method=m)
        assert res.method == f"{m}_surface"
        assert res.verdict == "fail" and _bbox_hits(res.zones[0].bbox, _mask_at_work(mask, res))


def test_runtime_1280x720():
    img, _ = synth.add_scratch(_surface("plastic", 0), np.random.default_rng(4))
    assert img.shape[:2] == (720, 1280)
    surface.scan_surface(img, P)                    # warm-up
    times = []
    for _ in range(3):
        t = time.perf_counter()
        surface.scan_surface(img, P)
        times.append((time.perf_counter() - t) * 1000.0)
    print(f"\nscan_surface 1280x720 (work 1024x576): median {np.median(times):.0f} ms, "
          f"max {max(times):.0f} ms")
    assert max(times) <= 3000.0
