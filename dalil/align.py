"""Reference/frame registration with three strategies, tried in order.

1. ``identity`` - fixed-camera fast path: the frame is already registered when the
   normalised cross-correlation (NCC, worst 10 % of pixel residuals trimmed) of
   downscaled, lighting-normalised gray images is high and a quick low-res ECC
   refinement finds (almost) no motion. The normalisation makes the test robust to
   monotone lighting changes; the trimming keeps a small defect from vetoing it.
2. ``orb`` - ORB features + Lowe ratio test + RANSAC homography (textured parts).
3. ``ecc`` - fallback for low-texture parts when ORB finds too few features/inliers:
   coarse-to-fine ``cv2.findTransformECC`` (Euclidean) on lighting-normalised gray
   (see `_photonorm`), initialised by phase correlation, accepted only if the final
   correlation >= ``params.ecc_min_corr``.

The homography maps AFTER (frame) pixel coordinates into the BEFORE (reference)
frame, so that the frame can be warped onto the reference and compared cell by cell.
A failed quality gate yields `ok=False` with a machine-readable `reason`, which
the pipeline turns into a "retake" request.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Params

# Detection internals (not user tunables: they only shape the keypoint search).
_FAST_THRESHOLDS = (20, 10, 5)  # retried in order while too few keypoints are found
_GRID = 8                       # keypoints are bucketed on a _GRID x _GRID grid
_MIN_KEYPOINTS = 16             # below this an image is "featureless"
_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

# Degeneracy limits for the affine part of H (scale in [0.6, 1.6] => det in [0.36, 2.56]).
_DET_MIN, _DET_MAX = 0.6 ** 2, 1.6 ** 2
_MAX_ANISOTROPY = 2.0           # ratio of singular values of the 2x2 part (rejects strong shear)

# Identity fast path / ECC internals.
_NCC_SIDE = 256                 # max side of the images compared by the fixed-camera test
_NCC_TRIM = 0.10                # fraction of largest residuals ignored (local defects)
_ECC_COARSE_SIDE = 128          # coarsest ECC pyramid level (max side)
_ECC_FINE_SIDE = 512            # finest ECC level (sub-pixel there is ~0.2 px at 1024)
_ECC_ITERS = (80, 40, 25)       # iterations per level, coarse -> fine (last value repeats)
_ECC_EPS = 1e-5
_FIXED_ECC_ITERS = 30           # ECC iterations of the identity check (256 px, from a close start)
_LOCAL_STD_FLOOR = 4.0          # gray levels; keeps sensor noise in flat areas from blowing up
_ECC_MAX_ANGLE_DEG = 15.0       # a part on a belt does not rotate more than this
_ECC_MAX_SHIFT_FRAC = 0.25      # max translation as a fraction of the image's max side
_ECC_FALLBACK_REASONS = ("too_few_features", "too_few_inliers")


@dataclass
class AlignResult:
    ok: bool
    reason: str | None
    H: np.ndarray | None          # 3x3, AFTER -> BEFORE pixel coordinates
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    after_warped: np.ndarray | None  # BGR, same size as before
    valid: np.ndarray | None         # bool HxW, before pixels covered by warped after
    method: str = "orb"   # "identity" | "orb" | "ecc"; "none" when every strategy failed
    score: float = 0.0    # identity: robust NCC; orb: inlier ratio; ecc: final ECC correlation


def resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale so that max(H, W) <= max_side. Returns (image, scale), scale = new / old <= 1."""
    h, w = img.shape[:2]
    side = max(h, w)
    if max_side <= 0 or side <= max_side:
        return img, 1.0
    scale = max_side / side
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), scale


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3].astype(np.uint8, copy=False), cv2.COLOR_BGR2GRAY)
    return np.clip(img, 0, 255).astype(np.uint8, copy=False)


# --------------------------------------------------------------------------- ORB

def _bucket(kps: list, shape: tuple[int, int], n_total: int) -> list:
    """Keep the strongest keypoints per grid cell so matches cover the whole frame."""
    h, w = shape
    per_cell = max(4, n_total // (_GRID * _GRID))
    cells: dict[tuple[int, int], list] = {}
    for kp in kps:
        x, y = kp.pt
        key = (min(int(y * _GRID / h), _GRID - 1), min(int(x * _GRID / w), _GRID - 1))
        cells.setdefault(key, []).append(kp)
    kept = []
    for bucket in cells.values():
        bucket.sort(key=lambda k: k.response, reverse=True)
        kept.extend(bucket[:per_cell])
    if len(kept) > n_total:
        kept.sort(key=lambda k: k.response, reverse=True)
        kept = kept[:n_total]
    return kept


def _detect(gray: np.ndarray, n_features: int):
    """ORB keypoints + descriptors on a CLAHE-equalised image (robust to lighting)."""
    eq = _CLAHE.apply(gray)
    kps: list = []
    for fast in _FAST_THRESHOLDS:
        orb = cv2.ORB_create(nfeatures=n_features * 2, fastThreshold=fast)
        kps = list(orb.detect(eq, None))
        if len(kps) >= n_features // 2:
            break
    kps = _bucket(kps, gray.shape, n_features)
    if not kps:
        return [], None
    kps, desc = orb.compute(eq, kps)
    return list(kps), desc


def _match(desc_after: np.ndarray, desc_before: np.ndarray, ratio: float) -> list:
    """Brute-force Hamming kNN (k=2) with Lowe's ratio test; query = after, train = before."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pair in bf.knnMatch(desc_after, desc_before, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    return good


def _degenerate(H: np.ndarray, shape: tuple[int, int], params: Params) -> bool:
    """Reject flips, extreme scale/shear and strong perspective."""
    if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return True
    Hn = H / H[2, 2]
    A = Hn[:2, :2]
    det = float(np.linalg.det(A))
    if not (_DET_MIN <= det <= _DET_MAX):  # also rejects det <= 0, i.e. a flip
        return True
    s = np.linalg.svd(A, compute_uv=False)
    if s[1] <= 0 or s[0] / s[1] > _MAX_ANISOTROPY:
        return True
    h, w = shape
    persp = abs(Hn[2, 0]) * w + abs(Hn[2, 1]) * h
    return persp > params.max_persp_distortion


def _align_orb(g_before: np.ndarray, g_after: np.ndarray, params: Params):
    """ORB + RANSAC. Returns (H or None, reason or None, n_matches, n_inliers)."""
    kp_b, d_b = _detect(g_before, params.orb_features)
    kp_a, d_a = _detect(g_after, params.orb_features)
    if d_b is None or d_a is None or len(kp_b) < _MIN_KEYPOINTS or len(kp_a) < _MIN_KEYPOINTS:
        return None, "too_few_features", 0, 0

    matches = _match(d_a, d_b, params.ratio_test)
    n_matches = len(matches)
    if n_matches < max(4, params.min_inliers):
        return None, "too_few_inliers", n_matches, 0

    src = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, params.ransac_thresh,
                                maxIters=5000, confidence=0.999)
    if H is None or inl is None:
        return None, "too_few_inliers", n_matches, 0
    n_inliers = int(inl.sum())
    if n_inliers < params.min_inliers or n_inliers / n_matches < params.min_inlier_ratio:
        return None, "too_few_inliers", n_matches, n_inliers
    # Plain least-squares refit on all RANSAC inliers: measurably more accurate
    # at the image corners than the RANSAC estimate itself.
    mask = inl.ravel().astype(bool)
    H_ls, _ = cv2.findHomography(src[mask], dst[mask], 0)
    if H_ls is not None and np.all(np.isfinite(H_ls)):
        H = H_ls
    if _degenerate(H, g_before.shape, params):
        return None, "degenerate_homography", n_matches, n_inliers
    return H / H[2, 2], None, n_matches, n_inliers


# ------------------------------------------------------- identity fast path / ECC

def _photonorm(gray: np.ndarray, side: int) -> np.ndarray:
    """Lighting-normalised float32 image at max side `side`, used by the identity test and ECC.

    Global histogram equalisation is invariant to any global monotone intensity change
    (gain, gamma, exposure); the following local mean/std normalisation (Gaussian
    window ~1/16 of the image) also absorbs smooth spatial gain such as window light.
    Unlike CLAHE's fixed tile grid, both steps are shift-equivariant, so content
    motion stays visible even on smooth, low-texture parts.
    """
    small, _ = resize_max_side(gray, side)
    f = cv2.GaussianBlur(cv2.equalizeHist(small).astype(np.float32), (0, 0), 1.0)
    sigma = max(small.shape) / 16.0
    mu = cv2.GaussianBlur(f, (0, 0), sigma)
    d = f - mu
    sd = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), sigma) + _LOCAL_STD_FLOOR ** 2)
    return d / sd


def _robust_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """NCC with the worst `_NCC_TRIM` fraction of pixel residuals discarded.

    For standardised images NCC = 1 - mean((a - b)^2) / 2; trimming the largest
    residuals keeps a local defect (tape, a mark) from vetoing the fast path, while
    misregistration, a wrong part or non-monotone lighting affect most pixels.
    """
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    r = ((a - b) ** 2).ravel()
    k = max(1, int(round(r.size * (1.0 - _NCC_TRIM))))
    return float(1.0 - np.partition(r, k - 1)[:k].mean() / 2.0)


def _visible_motion(W: np.ndarray, img: np.ndarray) -> float:
    """RMS displacement of the 2x3 warp W, weighted by the image's squared gradient.

    Motion is measured where there is content: on a rotation-symmetric part a spurious
    rotation about its centre (which ECC cannot pin down) moves nothing visible.
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    wgt = gx * gx + gy * gy
    total = float(wgt.sum())
    if total <= 1e-9:
        return float("inf")
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (W[0, 0] - 1) * xx + W[0, 1] * yy + W[0, 2]
    dy = W[1, 0] * xx + (W[1, 1] - 1) * yy + W[1, 2]
    return float(np.sqrt(((dx * dx + dy * dy) * wgt).sum() / total))


def _fixed_camera(g_before: np.ndarray, g_after: np.ndarray, params: Params) -> float | None:
    """Robust NCC if the frame is already registered to the reference, else None.

    Two checks at `_NCC_SIDE` resolution: robust NCC >= params.fixed_cam_ncc (same scene),
    and a quick Euclidean ECC refinement (initialised by phase correlation) finds a visible
    motion (`_visible_motion`) of at most params.fixed_cam_max_shift full-resolution
    pixels. NCC alone is not enough: a smooth part keeps a high NCC under a few pixels
    of shift or a small rotation, which the refinement exposes.
    """
    if g_before.shape != g_after.shape:
        return None
    a = _photonorm(g_before, _NCC_SIDE)
    b = _photonorm(g_after, _NCC_SIDE)
    ncc = _robust_ncc(a, b)
    if not np.isfinite(ncc) or ncc < params.fixed_cam_ncc:
        return None
    scale = a.shape[1] / g_before.shape[1]
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    W = np.float32([[1, 0, dx], [0, 1, dy]])
    try:
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, _FIXED_ECC_ITERS, _ECC_EPS)
        _, W = cv2.findTransformECC(a, b, W, cv2.MOTION_EUCLIDEAN, crit, None, 5)
    except cv2.error:
        return None
    return ncc if _visible_motion(W, a) / scale <= params.fixed_cam_max_shift else None


def _ecc(g_before: np.ndarray, g_after: np.ndarray,
         params: Params) -> tuple[np.ndarray, float] | None:
    """Coarse-to-fine Euclidean ECC. Returns (H frame->reference at full res, corr) or None.

    ECC's warp W maps reference (template) coordinates into frame (input) coordinates;
    it is initialised by phase correlation at the coarsest level (then by identity if
    that does not reach params.ecc_min_corr) and refined level by level.
    """
    if g_before.shape != g_after.shape:
        return None
    h, w = g_before.shape
    full = max(h, w)
    sides = [min(_ECC_COARSE_SIDE, full)]
    while sides[-1] < min(_ECC_FINE_SIDE, full):
        sides.append(min(sides[-1] * 2, _ECC_FINE_SIDE, full))
    levels = [(_photonorm(g_before, s), _photonorm(g_after, s)) for s in sides]

    tb, ta = levels[0]
    win = cv2.createHanningWindow(tb.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(tb, ta, win)
    inits = (np.float32([[1, 0, dx], [0, 1, dy]]), np.eye(2, 3, dtype=np.float32))

    best: tuple[np.ndarray, float] | None = None
    for W0 in inits:
        W, corr = W0.copy(), -1.0
        try:
            for k, (tb, ta) in enumerate(levels):
                if k:
                    W[:, 2] *= tb.shape[1] / levels[k - 1][0].shape[1]
                crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                        _ECC_ITERS[min(k, len(_ECC_ITERS) - 1)], _ECC_EPS)
                corr, W = cv2.findTransformECC(tb, ta, W, cv2.MOTION_EUCLIDEAN, crit, None, 5)
        except cv2.error:  # non-convergence or a constant image
            continue
        if np.isfinite(corr) and (best is None or corr > best[1]):
            best = (W.copy(), float(corr))
        if best is not None and best[1] >= params.ecc_min_corr:
            break
    if best is None:
        return None

    W = best[0].astype(np.float64)
    W[:, 2] *= w / levels[-1][0].shape[1]  # finest level -> full resolution
    angle = np.degrees(np.arctan2(W[1, 0], W[0, 0]))
    if abs(angle) > _ECC_MAX_ANGLE_DEG or np.hypot(W[0, 2], W[1, 2]) > _ECC_MAX_SHIFT_FRAC * full:
        return None
    H = np.linalg.inv(np.vstack([W, [0.0, 0.0, 1.0]]))
    return H / H[2, 2], best[1]


# --------------------------------------------------------------------- public API

def _warp(after_bgr: np.ndarray, H: np.ndarray, shape: tuple[int, int],
          params: Params) -> tuple[np.ndarray, np.ndarray]:
    """Warp the frame into the reference frame; returns (warped, valid mask)."""
    h, w = shape
    warped = cv2.warpPerspective(after_bgr, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ones = np.ones(after_bgr.shape[:2], np.uint8)
    cover = cv2.warpPerspective(ones, H, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    r = int(params.align_erode_px)
    if r > 0:
        # Erode away interpolation seams along the after-image border; the default
        # border handling leaves the before-frame edges themselves untouched.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
        cover = cv2.erode(cover, kernel)
    return warped, cover.astype(bool)


def _fail(reason: str, n_matches: int = 0, n_inliers: int = 0) -> AlignResult:
    ratio = n_inliers / n_matches if n_matches else 0.0
    return AlignResult(False, reason, None, n_matches, n_inliers, ratio, None, None,
                       method="none", score=0.0)


def align(before_bgr: np.ndarray, after_bgr: np.ndarray, params: Params) -> AlignResult:
    """Register `after_bgr` (frame) onto `before_bgr` (reference) and warp it there.

    Strategies in order: identity fast path, ORB+RANSAC, ECC fallback (see module
    docstring). When everything fails, the reason of the ORB stage is reported.
    """
    g_before = _to_gray_u8(before_bgr)
    g_after = _to_gray_u8(after_bgr)
    shape = g_before.shape

    ncc = _fixed_camera(g_before, g_after, params)
    if ncc is not None:
        H = np.eye(3)
        warped, valid = _warp(after_bgr, H, shape, params)
        return AlignResult(True, None, H, 0, 0, 0.0, warped, valid, method="identity", score=ncc)

    H, reason, n_matches, n_inliers = _align_orb(g_before, g_after, params)
    if H is not None:
        ratio = n_inliers / n_matches
        warped, valid = _warp(after_bgr, H, shape, params)
        return AlignResult(True, None, H, n_matches, n_inliers, ratio, warped, valid,
                           method="orb", score=ratio)

    if reason in _ECC_FALLBACK_REASONS:
        found = _ecc(g_before, g_after, params)
        if found is not None and found[1] >= params.ecc_min_corr:
            H, corr = found
            warped, valid = _warp(after_bgr, H, shape, params)
            return AlignResult(True, None, H, n_matches, n_inliers, 0.0, warped, valid,
                               method="ecc", score=corr)
    return _fail(reason, n_matches, n_inliers)
