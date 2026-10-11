"""Tests for ava.quality (procedural fixtures only)."""

import cv2
import numpy as np
import pytest

from ava.config import Params
from ava.quality import blur_score, capture_report, glare_mask, is_too_dark, outline_png


def textured(seed: int = 0, h: int = 360, w: int = 480) -> np.ndarray:
    """Mid-gray gradient with random filled/outlined shapes (sharp edges)."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(90, 160, w, dtype=np.float32)
    img = np.repeat(np.repeat(ramp[None, :, None], h, 0), 3, 2).astype(np.uint8)
    for _ in range(30):
        color = tuple(int(v) for v in rng.integers(20, 230, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 60)), y + int(rng.integers(10, 60))),
                          color, int(rng.choice([-1, 2])))
        else:
            cv2.circle(img, (x, y), int(rng.integers(5, 30)), color, int(rng.choice([-1, 2])))
    return img


def test_glare_mask_white_blob_not_red_patch():
    img = textured()
    cv2.circle(img, (120, 120), 25, (255, 255, 255), -1)       # specular highlight
    cv2.rectangle(img, (300, 200), (380, 280), (0, 0, 255), -1)  # saturated red paint
    m = glare_mask(img, Params())
    assert m.shape == img.shape[:2] and m.dtype == bool
    assert m[120, 120] and m[100:141, 100:141].mean() > 0.9
    assert not m[215:265, 315:365].any()


def test_glare_mask_ignores_single_pixel_speckles():
    img = np.full((100, 100, 3), 100, np.uint8)
    img[50, 50] = 255
    assert not glare_mask(img, Params()).any()


def test_blur_and_dark_helpers():
    img = textured()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), 6)
    assert blur_score(gray) > blur_score(blurred)
    assert not is_too_dark(gray, Params())
    assert is_too_dark(gray * 0.1, Params())


def test_capture_report_flags_blur():
    img = cv2.GaussianBlur(textured(), (0, 0), 8)
    rep = capture_report(img, Params())
    assert set(rep) == {"blur", "too_blurry", "glare_frac", "too_dark", "warnings"}
    assert rep["too_blurry"] and "blurry" in rep["warnings"]
    assert not rep["too_dark"]


def test_capture_report_flags_dark():
    img = (textured().astype(np.float32) * 0.12).astype(np.uint8)
    rep = capture_report(img, Params())
    assert rep["too_dark"] and "dark" in rep["warnings"]


def test_capture_report_clean_and_glare():
    img = textured()
    rep = capture_report(img, Params())
    assert rep["warnings"] == []
    assert isinstance(rep["blur"], float) and rep["glare_frac"] == 0.0
    img[:, : img.shape[1] // 2] = 255
    rep = capture_report(img, Params())
    assert rep["glare_frac"] > Params().max_glare_frac and "glare" in rep["warnings"]


def test_outline_png_is_rgba_with_white_edges():
    img = np.ascontiguousarray(np.concatenate([textured(1, 900, 800), textured(2, 900, 800)], 1))
    data = outline_png(img, max_side=720)
    assert isinstance(data, bytes) and data[:8] == b"\x89PNG\r\n\x1a\n"
    dec = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert dec.ndim == 3 and dec.shape[2] == 4
    assert max(dec.shape[:2]) == 720
    alpha = dec[..., 3]
    assert set(np.unique(alpha)) <= {0, 255}
    assert 0.005 < (alpha == 255).mean() < 0.5
    assert (dec[alpha == 255][:, :3] == 255).all()


def test_blur_score_ignores_lighting_gain():
    """A dimmer lamp / lower exposure must not make a sharp frame read as blurry."""
    gray = cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY).astype(np.float32)
    sharp = blur_score(gray)
    for gain in (0.4, 0.6, 1.3):
        assert blur_score(gray * gain) == pytest.approx(sharp, rel=1e-3)
    dim = (textured().astype(np.float32) * 0.45).astype(np.uint8)  # mean ~55: not "dark"
    rep = capture_report(dim, Params())
    assert not rep["too_blurry"] and not rep["too_dark"]
    # Real defocus still lowers the score by an order of magnitude.
    assert blur_score(cv2.GaussianBlur(gray * 0.6, (0, 0), 2)) < 0.1 * sharp


def test_blur_score_flat_image_is_finite():
    flat = np.full((64, 64), 128, np.float32)
    assert blur_score(flat) == 0.0
