"""Tests for dalil.align (procedural fixtures only)."""

import cv2
import numpy as np
import pytest

from dalil.align import AlignResult, align, resize_max_side
from dalil.config import Params


def textured(seed: int, h: int = 480, w: int = 640) -> np.ndarray:
    """Gradient background with random rectangles, circles and stroke polylines."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        a, b, k = rng.uniform(-0.15, 0.15, 3)
        img[..., c] = 110 + a * xx + b * yy + k * 40
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(40):
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        kind = rng.integers(0, 3)
        if kind == 0:
            x2, y2 = x + int(rng.integers(10, 80)), y + int(rng.integers(10, 80))
            cv2.rectangle(img, (x, y), (x2, y2), color, int(rng.choice([-1, 2])))
        elif kind == 1:
            cv2.circle(img, (x, y), int(rng.integers(5, 40)), color, int(rng.choice([-1, 2])))
        else:
            pts = (np.array([x, y]) + rng.integers(-30, 30, (5, 2)).cumsum(0)).astype(np.int32)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
    return img


def shift_rot(h: int, w: int, dx: float, dy: float, deg: float) -> np.ndarray:
    """3x3 homography: rotation by `deg` around the image centre plus a shift."""
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    m[:, 2] += (dx, dy)
    return np.vstack([m, [0, 0, 1]])


@pytest.fixture(scope="module")
def before():
    return textured(1)


def make_after(before, H_true):
    h, w = before.shape[:2]
    return cv2.warpPerspective(before, H_true, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)


def corner_error(H_est, H_true, h, w) -> float:
    """Max distance between corners mapped by H_est and by inv(H_true)."""
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(pts, H_est)
    b = cv2.perspectiveTransform(pts, np.linalg.inv(H_true))
    return float(np.abs(a - b).max())


def test_recovers_small_homography(before):
    h, w = before.shape[:2]
    H_true = shift_rot(h, w, 8, -5, 1.5)
    res = align(before, make_after(before, H_true), Params())
    assert isinstance(res, AlignResult)
    assert res.ok and res.reason is None and res.method == "orb"
    assert res.n_inliers >= Params().min_inliers
    assert 0 < res.inlier_ratio <= 1 and res.n_matches >= res.n_inliers
    assert corner_error(res.H, H_true, h, w) < 1.5
    assert res.after_warped.shape == before.shape and res.after_warped.dtype == np.uint8
    assert res.valid.shape == (h, w) and res.valid.dtype == bool
    assert 0.8 < res.valid.mean() < 1.0
    diff = np.abs(res.after_warped.astype(np.float32) - before.astype(np.float32))
    assert diff[res.valid].mean() < 6.0


def test_aligns_under_gamma_and_exposure_change(before):
    h, w = before.shape[:2]
    H_true = shift_rot(h, w, -6, 7, -1.0)
    after = make_after(before, H_true).astype(np.float32) / 255.0
    after = np.clip(255.0 * 0.8 * after ** 0.6, 0, 255).astype(np.uint8)
    res = align(before, after, Params())
    assert res.ok, res.reason
    assert res.method == "orb"
    assert corner_error(res.H, H_true, h, w) < 2.0


def test_unrelated_image_rejected(before):
    res = align(before, textured(99), Params())
    assert not res.ok
    assert res.reason in ("too_few_inliers", "degenerate_homography")
    assert res.H is None and res.after_warped is None and res.valid is None
    assert res.method == "none"


def test_flat_image_too_few_features(before):
    flat = np.full_like(before, 128)
    res = align(before, flat, Params())
    assert not res.ok and res.reason == "too_few_features"
    assert res.n_matches == 0 and res.n_inliers == 0


def test_flipped_image_rejected(before):
    res = align(before, np.ascontiguousarray(before[:, ::-1]), Params())
    assert not res.ok


def test_resize_max_side():
    img = np.zeros((300, 1200, 3), np.uint8)
    small, scale = resize_max_side(img, 600)
    assert small.shape == (150, 600, 3) and scale == pytest.approx(0.5)
    same, scale = resize_max_side(img, 2000)
    assert same is img and scale == 1.0


# --- identity fast path / ECC fallback ------------------------------------------------

def noisy(img: np.ndarray, seed: int, sigma: float = 2.0) -> np.ndarray:
    """Independent sensor noise, as between two real captures."""
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def relight(img: np.ndarray, gamma: float = 0.6, gain: float = 0.8) -> np.ndarray:
    """Monotone lighting change: gain and gamma."""
    return np.clip(255.0 * gain * (img.astype(np.float32) / 255.0) ** gamma, 0, 255)


SMOOTH_HW = (576, 1024)
SMOOTH_CENTRE = (0.55 * 1024, 0.5 * 576)


def smooth_part(h: int = SMOOTH_HW[0], w: int = SMOOTH_HW[1]) -> np.ndarray:
    """Low-texture part: a lighting gradient plus one big soft-edged circle (float32 BGR)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    circ = np.zeros((h, w), np.float32)
    cv2.circle(circ, (round(SMOOTH_CENTRE[0]), round(SMOOTH_CENTRE[1])), int(0.3 * h), 1.0, -1)
    g = 70 + 90 * xx / w + 30 * yy / h + 60 * cv2.GaussianBlur(circ, (0, 0), 12)
    return np.repeat(g[..., None], 3, 2)


@pytest.fixture(scope="module")
def smooth():
    return smooth_part()


@pytest.mark.parametrize("kind", ["textured", "smooth"])
def test_fixed_camera_lighting_change_takes_identity_path(kind, smooth):
    base = textured(3, *SMOOTH_HW).astype(np.float32) if kind == "textured" else smooth
    ref = noisy(base, 1)
    frame = noisy(relight(base), 2)
    res = align(ref, frame, Params())
    assert res.ok and res.method == "identity", (res.method, res.score)
    assert np.array_equal(res.H, np.eye(3))
    assert res.score >= Params().fixed_cam_ncc
    assert np.array_equal(res.after_warped, frame)
    assert res.valid.shape == SMOOTH_HW and res.valid.all()


def test_small_defect_keeps_registration():
    # A piece of tape on a static scene must not break registration. The identity
    # fast path may hand over to ORB (its quick motion check is deliberately strict,
    # fixed_cam_max_shift), but either way the result is (almost) no motion.
    base = textured(4, *SMOOTH_HW)
    frame = base.copy()
    cv2.rectangle(frame, (400, 200), (480, 240), (60, 180, 200), -1)  # a piece of tape
    res = align(noisy(base, 1), noisy(frame, 2), Params())
    assert res.ok and res.method in ("identity", "orb")
    assert corner_error(res.H, np.eye(3), *SMOOTH_HW) < 0.5


def test_identity_can_be_disabled():
    base = textured(5, 480, 640)
    res = align(base, base.copy(), Params(fixed_cam_ncc=1.01))
    assert res.ok and res.method == "orb"


def circle_error(H_est: np.ndarray, H_true: np.ndarray) -> float:
    """Round-trip error (px) of the circle centre: reference -> frame (true) -> reference (estimate)."""
    c = np.float32([SMOOTH_CENTRE]).reshape(-1, 1, 2)
    back = cv2.perspectiveTransform(cv2.perspectiveTransform(c, H_true), H_est)
    return float(np.linalg.norm(back - c))


@pytest.mark.parametrize("dx,dy,deg", [(10, -4, 0.0), (3, 0, 0.0), (25, 12, 2.0)])
def test_low_texture_part_falls_back_to_ecc(smooth, dx, dy, deg):
    h, w = SMOOTH_HW
    H_true = shift_rot(h, w, dx, dy, deg)
    ref = noisy(smooth, 1)
    frame = noisy(relight(make_after(smooth, H_true)), 2)
    res = align(ref, frame, Params())
    assert res.ok, res.reason
    assert res.method == "ecc"                       # identity rejected (moved), ORB starved
    assert res.score >= Params().ecc_min_corr
    assert circle_error(res.H, H_true) < 1.5
    # Photometric residual on the circle's soft edge (where motion is visible), using the
    # noise-free relit frame warped by the estimate.
    target = relight(smooth)[..., 0]
    clean = relight(make_after(smooth, H_true))[..., 0]
    warped = cv2.warpPerspective(clean, res.H, (w, h), flags=cv2.INTER_LINEAR)
    grad = cv2.magnitude(cv2.Sobel(target, cv2.CV_32F, 1, 0), cv2.Sobel(target, cv2.CV_32F, 0, 1))
    edge = res.valid & (grad > np.percentile(grad, 90))
    aligned = np.abs(warped - target)[edge].mean()
    unaligned = np.abs(clean - target)[edge].mean()
    assert aligned < 1.5 and aligned < 0.4 * unaligned


@pytest.mark.parametrize("frame_kind", ["no_part", "part_removed"])
def test_ecc_does_not_accept_absent_part(smooth, frame_kind):
    h, w = SMOOTH_HW
    if frame_kind == "no_part":
        empty = np.full_like(smooth, 120.0)
    else:  # the background gradient alone
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        empty = np.repeat((70 + 90 * xx / w + 30 * yy / h)[..., None], 3, 2)
    res = align(noisy(smooth, 1), noisy(empty, 2), Params())
    assert not res.ok and res.method == "none"
    assert res.reason in ("too_few_features", "too_few_inliers")


def test_unrelated_textured_images_rejected_by_all_paths():
    res = align(noisy(textured(11, *SMOOTH_HW), 1), noisy(textured(12, *SMOOTH_HW), 2), Params())
    assert not res.ok and res.method == "none" and res.H is None


def test_different_sizes_skip_identity():
    big = textured(6, 480, 640)
    crop = np.ascontiguousarray(big[20:460, 10:630])
    res = align(big, crop, Params())
    assert res.ok and res.method == "orb"
    assert res.after_warped.shape == big.shape


def test_small_rotation_is_not_identity():
    """A pure rotation keeps the global shift ~0; the ECC check must still catch it."""
    base = textured(7, *SMOOTH_HW)
    H_true = shift_rot(*SMOOTH_HW, 0, 0, 0.5)
    res = align(noisy(base, 1), noisy(make_after(base, H_true), 2), Params())
    assert res.ok and res.method == "orb"
    assert corner_error(res.H, H_true, *SMOOTH_HW) < 1.5
