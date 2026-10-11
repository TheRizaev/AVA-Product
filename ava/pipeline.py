"""Reference-based inspection pipeline: golden set -> per-frame verdict (see docs/SPEC.md).

    prepare_reference(shots)  resize -> register extra shots to shot 0 -> blurred gray
                              -> features per method (computed once, cached by the server)
    inspect_frame(ref, frame) resize -> capture quality (blur / dark) -> reference scale
                              -> align to shot 0
                              -> checked cells (valid, on the part if it moved, no glare)
                              -> glare / coverage gates -> blurred gray -> features
                              -> multi-reference change map (PBP: low-contrast gate; min over
                                 background-registered views if the part moved) -> smoothing
                              -> zones -> mismatch gate -> verdict (never PASS with an
                                 unexplained glare-like blob in the frame) + overlay JPEGs

`inspect_compare` aligns once and scores several methods (PBP and the baselines)
on the same aligned frame, so the UI and the benchmark compare like with like.
All coordinates in results are reference pixels at the working resolution.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import baselines, quality
from .align import align, corner_shift, resize_max_side
from .change import multi_reference_change_map
from .config import Params
from .pbp import Signatures, cell_mask_from_pixel_mask, grid_shape, serialize_signatures
from .zones import Zone, coverage, extract_zones

VERDICTS = ("pass", "fail", "retake")
REASONS = (None, "alignment", "glare", "blurry", "dark", "low_coverage", "mismatch")
HINT_CHANGE_ANGLE = "change_angle"
HINT_BRIGHT_SPOT = "bright_spot"   # a large glare-like blob was not checked: glare or a white object

_JPEG_QUALITY = 85
_RED = (40, 40, 230)          # BGR
_GREY = (150, 150, 150)
_HATCH_PERIOD = 10            # px between hatch lines of unchecked areas


@dataclass
class Reference:
    """A prepared golden set at the working resolution (all shots registered to shot 0)."""

    images: list                 # BGR uint8 [h, w, 3]
    grays: list                  # float32 [h, w], blurred with params.blur_sigma
    features: dict               # method -> list of per-shot features
    glare: np.ndarray            # bool [h, w], glare in any shot
    work_size: tuple[int, int]   # (w, h)
    scale: float                 # work / original size of shot 0
    outline_png: bytes
    capture: dict                # quality.capture_report of shot 0
    # Running sha256 over the serialised features of shot 0, per method (fingerprint prefix).
    _hash_prefix: dict = field(default_factory=dict, repr=False)


@dataclass
class InspectResult:
    verdict: str                 # "pass" | "fail" | "retake"
    reason: str | None           # see REASONS
    zones: list                  # list[Zone], ranked
    coverage: float              # fraction of grid cells checked
    changed_frac: float          # fraction of checked cells at/above the threshold
    align: dict                  # method, n_inliers, inlier_ratio, score
    timing_ms: dict              # stage -> ms, plus "total"
    method: str
    threshold: float
    images: dict                 # "frame" / "overlay" -> JPEG bytes
    signature_sha256: str | None
    hints: list = field(default_factory=list)
    work_size: tuple[int, int] = (0, 0)   # (w, h) of the reference working image
    # Smoothed change map and checked cells on the window grid (None if the frame was
    # rejected before scoring). Not serialised; used by eval/benchmark.py to re-threshold.
    score: np.ndarray | None = field(default=None, repr=False)
    checked: np.ndarray | None = field(default=None, repr=False)
    # A large glare-like blob was masked (see `bright_spot`): the verdict is never "pass".
    # Not serialised (the hints carry it); eval passes it to `decide` when re-thresholding.
    bright_spot: bool = field(default=False, repr=False)


# --------------------------------------------------------------------------- helpers

def _gray(bgr: np.ndarray) -> np.ndarray:
    """Gray float32 0..255 of a BGR (or gray) uint8 image."""
    if bgr.ndim == 3:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr.astype(np.float32)


def _work_gray(bgr: np.ndarray, params: Params) -> np.ndarray:
    """Gray plus the working blur that tames sensor noise before signatures."""
    g = _gray(bgr)
    if params.blur_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), params.blur_sigma)
    return g


def _as_bgr(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(img, dtype=np.uint8)


def _method_fns(method: str):
    methods = baselines.get_methods()
    if method not in methods:
        raise ValueError(f"unknown method: {method!r} (known: {', '.join(methods)})")
    return methods[method]


def method_threshold(method: str, params: Params) -> float:
    """`params.threshold` for PBP, `params.threshold_<method>` for the baselines."""
    if method == "pbp":
        return float(params.threshold)
    return float(getattr(params, f"threshold_{method}"))


def _feature_bytes(feat) -> list[bytes]:
    """Deterministic bytes of one feature object (PBP signatures or baseline arrays)."""
    if isinstance(feat, Signatures):
        return [serialize_signatures(feat)]
    out = [np.ascontiguousarray(feat.blocks).tobytes()]
    for extra in (feat.mean, feat.var):
        if extra is not None:
            out.append(np.ascontiguousarray(extra).tobytes())
    return out


def _signature_sha256(ref: Reference, method: str, frame_feat) -> str:
    """sha256(bytes(reference shot 0 features) + bytes(frame features))."""
    prefix = ref._hash_prefix.get(method)
    if prefix is None:
        prefix = hashlib.sha256()
        for b in _feature_bytes(ref.features[method][0]):
            prefix.update(b)
        ref._hash_prefix[method] = prefix
    h = prefix.copy()
    for b in _feature_bytes(frame_feat):
        h.update(b)
    return h.hexdigest()


def _ensure_features(ref: Reference, method: str, params: Params) -> list:
    """Reference features for `method`, computed on first use if not prepared."""
    feats = ref.features.get(method)
    if feats is None:
        feature_fn, _ = _method_fns(method)
        feats = [feature_fn(g, params) for g in ref.grays]
        ref.features[method] = feats
    return feats


def _jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _cells_to_pixels(cells: np.ndarray, shape: tuple[int, int], win: int, step: int) -> np.ndarray:
    """Pixel mask covered by the windows of the True cells (cell grid -> image)."""
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    if not cells.any():
        return out.astype(bool)
    up = np.kron(cells.astype(np.uint8), np.ones((step, step), np.uint8))
    hh, ww = min(h, up.shape[0]), min(w, up.shape[1])
    out[:hh, :ww] = up[:hh, :ww]
    extra = win - step
    if extra > 0:  # each cell's window extends `win - step` px beyond its stride block
        k = extra + 1
        out = cv2.dilate(out, np.ones((k, k), np.uint8), anchor=(k - 1, k - 1))
    return out.astype(bool)


def smooth_scores(score: np.ndarray, checked: np.ndarray, k: int) -> np.ndarray:
    """k x k box mean of the change map over checked cells only (normalised convolution).

    Suppresses isolated noisy cells while a defect a few cells wide keeps its score.
    """
    if k is None or k <= 1:
        return score
    w = checked.astype(np.float32)
    num = cv2.blur(score * w, (k, k), borderType=cv2.BORDER_REFLECT)
    den = cv2.blur(w, (k, k), borderType=cv2.BORDER_REFLECT)
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)
    return out.astype(np.float32)


def render_overlay(img: np.ndarray, zones: list, unchecked: np.ndarray | None) -> np.ndarray:
    """Overlay: unchecked area grey-hatched, zones outlined in red with their rank number."""
    out = img.copy()
    h, w = out.shape[:2]
    if unchecked is not None and unchecked.any():
        yy, xx = np.mgrid[0:h, 0:w]
        lines = ((xx + yy) % _HATCH_PERIOD) < 2
        shade = out.astype(np.float32)
        grey = np.array(_GREY, np.float32)
        shade[unchecked] = 0.55 * shade[unchecked] + 0.45 * grey
        hatch = unchecked & lines
        shade[hatch] = 0.3 * shade[hatch] + 0.7 * grey
        out = shade.astype(np.uint8)

    thick = max(2, round(max(h, w) / 400))
    font_scale = max(0.5, max(h, w) / 1600)
    for z in zones:
        x, y, bw, bh = z.bbox
        if len(z.contour) >= 3:
            cv2.polylines(out, [np.array(z.contour, np.int32).reshape(-1, 1, 2)], True, _RED,
                          thick, cv2.LINE_AA)
        cv2.rectangle(out, (x, y), (x + bw - 1, y + bh - 1), _RED, max(1, thick // 2), cv2.LINE_AA)
        label = str(z.id)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        pad = thick + 2
        lx = min(max(0, x), max(0, w - tw - 2 * pad))
        ly = y - th - base - 2 * pad
        if ly < 0:  # no room above the box: put the tag inside its top-left corner
            ly = min(max(0, y), max(0, h - th - base - 2 * pad))
        cv2.rectangle(out, (lx, ly), (lx + tw + 2 * pad, ly + th + base + 2 * pad), _RED, -1)
        cv2.putText(out, label, (lx + pad, ly + pad + th), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thick, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- reference

def prepare_reference(images_bgr: list, params: Params,
                      methods: tuple = ("pbp", "pixdiff")) -> Reference:
    """Prepare 1..params.max_references golden shots (see module docstring).

    Raises ValueError if there is no shot, too many shots, or an extra shot cannot be
    registered to shot 0 (it would not be a view of the same part).
    """
    if not images_bgr:
        raise ValueError("at least one reference image is required")
    if len(images_bgr) > params.max_references:
        raise ValueError(f"at most {params.max_references} reference images")

    first, scale = resize_max_side(_as_bgr(images_bgr[0]), params.work_max_side)
    h, w = first.shape[:2]
    images = [first]
    for k, extra in enumerate(images_bgr[1:], start=1):
        small, _ = resize_max_side(_as_bgr(extra), params.work_max_side)
        small = match_scale(small, (w, h))
        res = align(first, small, params)
        if not res.ok:
            raise ValueError(f"reference shot {k + 1} does not match shot 1 ({res.reason})")
        # Uncovered border pixels copy shot 0, so they never disagree with it.
        images.append(np.where(res.valid[..., None], res.after_warped, first))

    grays = [_work_gray(img, params) for img in images]
    glare = np.zeros((h, w), bool)
    for img in images:
        glare |= quality.glare_mask(img, params)

    ref = Reference(images=images, grays=grays, features={}, glare=glare, work_size=(w, h),
                    scale=float(scale), outline_png=quality.outline_png(first),
                    capture=quality.capture_report(first, params))
    for m in methods:
        _ensure_features(ref, m, params)
    return ref


# --------------------------------------------------------------------------- inspection

def part_roi(points: np.ndarray | None, shape: tuple[int, int], margin_px: int) -> np.ndarray | None:
    """Pixel mask of the part: convex hull of its keypoints, dilated by `margin_px` (None if unknown).

    Used when the part moved over a static background: then only the part is inspected
    and the background is shown as unchecked.
    """
    if points is None or len(points) < 3:
        return None
    mask = np.zeros(shape, np.uint8)
    hull = cv2.convexHull(np.asarray(points, np.float32).reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
    if margin_px > 0:
        k = 2 * int(margin_px) + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask.astype(bool)


def match_scale(img: np.ndarray, work_size: tuple[int, int], aspect_tol: float = 0.01,
                max_up: float = 2.0) -> np.ndarray:
    """Resize a photo to the reference's working scale before registration.

    Both photos are resized independently to `work_max_side` (never up), so a small Check
    (or a small Reference) would leave the whole resolution ratio to the homography, whose
    scale is limited (align._degenerate). Here the frame is brought to the same long side
    as the reference (`work_size` = (w, h)); with the same aspect ratio (within
    `aspect_tol`) it gets exactly the reference's size, so the identity and ECC paths
    (which need equal shapes) stay available. Upscaling is limited to `max_up`: a much
    smaller photo has too little detail to compare and is left to the alignment checks.
    """
    rw, rh = int(work_size[0]), int(work_size[1])
    h, w = img.shape[:2]
    if (w, h) == (rw, rh) or min(rw, rh, w, h) <= 0:
        return img
    s = max(rw, rh) / max(w, h)
    if s <= max_up and abs((w / h) / (rw / rh) - 1.0) <= aspect_tol:
        size = (rw, rh)
    else:
        s = min(s, max_up)
        size = (max(1, round(w * s)), max(1, round(h * s)))
    if size == (w, h):
        return img
    interp = cv2.INTER_AREA if size[0] < w else cv2.INTER_LINEAR
    return cv2.resize(img, size, interpolation=interp)


def bright_spot(core: np.ndarray, ref_gray: np.ndarray, frame_gray: np.ndarray, area: np.ndarray,
                params: Params) -> bool:
    """Does the frame show a glare-like blob that the reference does not explain?

    `core` = the frame's undilated glare pixels (bright, unsaturated) outside the
    reference's glare mask. Glare cells are not checked, and colour alone cannot tell a
    highlight from a white label, white tape or bare metal clipped to 255, so such a blob
    must not end in a silent PASS. A blob of >= `params.glare_blob_min_px` counts unless
    the reference was already about as bright there once the global exposure change is
    applied (a bright area clipped by a brighter exposure hides nothing new).
    """
    if params.glare_blob_min_px <= 0 or not core.any() or not area.any():
        return False
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= params.glare_blob_min_px]
    if not big:
        return False
    gain = float(np.median(frame_gray[area])) / max(1.0, float(np.median(ref_gray[area])))
    level = 0.9 * float(params.glare_v)
    return any(gain * float(np.median(ref_gray[labels == i])) < level for i in big)


def _change_map(method: str, dist_fn, ref_feats: list, feat, params: Params) -> np.ndarray:
    """Multi-reference tolerant change map; for PBP with the low-contrast gate.

    PBP compares the ORDER of pixels; in windows whose contrast is close to the sensor
    noise that order is random. Cells where the reference and the frame are both below
    `params.contrast_floor` are therefore treated as unchanged (a soft extension of
    the flat-window rule). The frame's contrast is first divided by the global contrast
    gain between the two images (exposure / lamp changes scale all contrasts), so the
    gate itself does not react to lighting. Baselines compare intensities: no gate.

    The flat rule of the signatures (window contrast < `flat_floor`, flat vs non-flat = 1)
    uses an absolute floor, so a darker exposure turns low-contrast windows "flat" in the
    frame only. A flat/non-flat disagreement is therefore ignored in a low-contrast cell
    (< 2 x `contrast_floor`) whose gain-normalised contrasts agree within a factor of 2:
    that is the exposure, not a defect (tape or a scratch changes the contrast far more).
    """
    score = multi_reference_change_map(dist_fn, ref_feats, feat, params.tolerance)
    if method == "pbp" and params.contrast_floor > 0:
        ref_con = ref_feats[0].contrast
        gain = contrast_gain(ref_con, feat.contrast, 2.0 * params.contrast_floor)
        frame_con = feat.contrast / gain
        hi = np.maximum(ref_con, frame_con)
        low = hi < params.contrast_floor
        exposure_flat = ((ref_feats[0].flat ^ feat.flat) & (hi < 2.0 * params.contrast_floor)
                         & (np.minimum(ref_con, frame_con) >= 0.5 * hi))
        score[low | exposure_flat] = 0.0
    return score


def contrast_gain(ref_con: np.ndarray, frame_con: np.ndarray, min_con: float) -> float:
    """Median frame/reference window-contrast ratio over textured cells (1.0 if too few).

    The median ignores local changes (defects); clipped to a plausible exposure range.
    """
    sel = ref_con >= min_con
    if np.count_nonzero(sel) < 50:
        return 1.0
    return float(np.clip(np.median(frame_con[sel] / ref_con[sel]), 0.25, 4.0))


def decide(score: np.ndarray, checked: np.ndarray, params: Params, threshold: float,
           sig_ref=None, sig_frame=None, bright: bool = False) -> tuple[str, str | None, list, float]:
    """Verdict from a (smoothed) change map at `threshold`: (verdict, reason, zones, changed_frac).

    changed_frac = fraction of checked cells at/above the threshold; above
    `params.mismatch_frac` the frame is a different / absent part ("retake", "mismatch"),
    otherwise "fail" if any zone survives `extract_zones`, else "pass" - unless `bright`
    (an unexplained glare-like blob was masked, see `bright_spot`): then "retake", "glare".
    `sig_ref` / `sig_frame` (PBP signatures) only enable the zone hints.
    """
    changed = float(np.count_nonzero(checked & (score >= threshold))) / max(1, int(checked.sum()))
    if changed > params.mismatch_frac:
        return "retake", "mismatch", [], changed
    zones = extract_zones(score, checked, dataclasses.replace(params, threshold=float(threshold)),
                          sig_ref, sig_frame, win=params.win, step=params.step)
    if zones:
        return "fail", None, zones, changed
    if bright:
        return "retake", "glare", [], changed
    return "pass", None, zones, changed


def _lap(timing: dict, name: str, since: float) -> float:
    """Record the ms since `since` under `name`; returns the new start time."""
    now = time.perf_counter()
    timing[name] = round((now - since) * 1000.0, 1)
    return now


def _align_info(res=None) -> dict:
    if res is None:
        return {"method": None, "n_inliers": 0, "inlier_ratio": 0.0, "score": 0.0}
    return {"method": res.method, "n_inliers": int(res.n_inliers),
            "inlier_ratio": round(float(res.inlier_ratio), 4), "score": round(float(res.score), 4)}


def inspect_compare(ref: Reference, frame_bgr: np.ndarray, params: Params,
                    methods: tuple = ("pbp", "pixdiff")) -> dict:
    """Align `frame_bgr` to the reference ONCE and evaluate each method on it.

    Returns {method: InspectResult}. Shared stages (resize, quality, align, mask) appear
    in every result's timing_ms; "total" is shared stages + that method's own stages.
    """
    methods = tuple(dict.fromkeys(methods))
    for m in methods:
        _method_fns(m)  # validate names early
    timing: dict = {}
    t0 = time.perf_counter()

    def finish(method, verdict, reason, own_timing, *, zones=(), cov=0.0, changed=0.0,
               al=None, images=None, sha=None, hints=()) -> InspectResult:
        stages = {**timing, **own_timing}
        stages["total"] = round(sum(stages.values()), 1)
        return InspectResult(verdict=verdict, reason=reason, zones=list(zones), coverage=float(cov),
                             changed_frac=float(changed), align=_align_info(al), timing_ms=stages,
                             method=method, threshold=method_threshold(method, params),
                             images=images or {}, signature_sha256=sha, hints=list(hints),
                             work_size=ref.work_size)

    # 1. resize + capture quality
    frame, _ = resize_max_side(_as_bgr(frame_bgr), params.work_max_side)
    tic = _lap(timing, "resize", t0)
    g = _gray(frame)
    reason = None
    blur = quality.blur_score(g)
    # Relative test too: a smooth part is "soft" in the reference as well.
    if blur < params.min_blur and blur < params.blur_ratio * ref.capture.get("blur", 0.0):
        reason = "blurry"
    elif quality.is_too_dark(g, params):
        reason = "dark"
    tic = _lap(timing, "quality", tic)
    if reason is not None:
        imgs = {"frame": _jpeg(frame)}
        imgs["overlay"] = imgs["frame"]
        return {m: finish(m, "retake", reason, {}, images=imgs) for m in methods}

    # 2. registration onto shot 0, at the reference's working scale
    frame = match_scale(frame, ref.work_size)
    al = align(ref.images[0], frame, params)
    tic = _lap(timing, "align", tic)
    if not al.ok:
        imgs = {"frame": _jpeg(frame)}
        imgs["overlay"] = imgs["frame"]
        return {m: finish(m, "retake", "alignment", {}, al=al, images=imgs) for m in methods}
    aligned, valid = al.after_warped, al.valid

    # 3. What is inspected. If the registration moved the image noticeably (the part was
    #    moved over a static belt/table, or the camera moved), only the region explained
    #    by the registration is inspected: the hull of its ORB inliers + a margin, i.e.
    #    the part (the whole scene if the camera moved). Checked cells are fully inside
    #    the warped frame and that region, and free of glare.
    win, step = params.win, params.step
    h, w = aligned.shape[:2]
    if min(grid_shape(h, w, win, step)) <= 0:
        raise ValueError("working image smaller than one window")
    moved = params.static_background and corner_shift(al.H, (h, w)) > params.static_bg_min_shift
    roi = part_roi(al.inlier_pts, (h, w), params.part_roi_margin_px) if moved else None
    area = valid if roi is None else valid & roi
    m = int(params.border_margin_px)
    if m > 0:  # image borders: lens fall-off, partial content
        area = area.copy()
        area[:m], area[-m:], area[:, :m], area[:, -m:] = False, False, False, False
    glare = (ref.glare | quality.glare_mask(aligned, params)) & area
    glare_frac = float(glare.sum()) / max(1, int(area.sum()))
    # Glare that only the frame has may be a white defect: it must not end in a silent PASS.
    new_core = quality.glare_mask(aligned, dataclasses.replace(params, glare_dilate_px=0)) & area & ~ref.glare
    bright = bright_spot(new_core, ref.grays[0], _gray(aligned), area, params)
    cells_area = cell_mask_from_pixel_mask(area, win, step, frac=1.0)
    cells_glare = cell_mask_from_pixel_mask(glare, win, step, frac=1.0 / (win * win))
    checked = cells_area & ~cells_glare
    if roi is None:
        cov = coverage(checked)
    else:  # fraction of the part that was checked
        cells_roi = cell_mask_from_pixel_mask(roi, win, step, frac=0.5)
        cov = float(checked.sum()) / max(1, int(cells_roi.sum()))
    unchecked_px = ~_cells_to_pixels(checked, (h, w), win, step)
    gray_f = _work_gray(aligned, params)
    # Extra views for the background left inside that region: the view registered on the
    # part shows a shifted background, while a view registered on the background (ORB's
    # second motion, or the frame as shot for a fixed camera) matches it. A cell counts
    # as changed only if it differs in every view. Only when align found that second
    # motion (evidence of a static background): with a single motion the camera moved,
    # and a min with a misregistered view would only hide defects.
    extra_views = []
    if moved and frame.shape == aligned.shape and al.alt_H is not None:
        extra_views.append(cv2.warpPerspective(frame, al.alt_H, (w, h), flags=cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_REPLICATE))
        if corner_shift(al.alt_H, (h, w)) > params.static_bg_min_shift:
            extra_views.append(frame)
    gray_views = [_work_gray(v, params) for v in extra_views]
    tic = _lap(timing, "mask", tic)
    frame_jpeg = _jpeg(aligned)

    gate = None
    if glare_frac > params.max_glare_frac:
        gate = "glare"
    elif cov < params.min_coverage:
        gate = "low_coverage"
    if gate is not None:
        overlay = _jpeg(render_overlay(aligned, [], unchecked_px))
        hints = [HINT_CHANGE_ANGLE] if gate == "glare" else []
        return {m: finish(m, "retake", gate, {}, cov=cov, al=al, hints=hints,
                          images={"frame": frame_jpeg, "overlay": overlay}) for m in methods}

    # 4. per method: features -> change map -> zones -> verdict
    out = {}
    for m in methods:
        own: dict = {}
        tm = time.perf_counter()
        feature_fn, dist_fn = _method_fns(m)
        ref_feats = _ensure_features(ref, m, params)
        feat = feature_fn(gray_f, params)
        tm = _lap(own, "features", tm)
        score = _change_map(m, dist_fn, ref_feats, feat, params)
        tm = _lap(own, "change", tm)
        for gv in gray_views:
            np.minimum(score, _change_map(m, dist_fn, ref_feats, feature_fn(gv, params), params),
                       out=score)
        if gray_views:
            tm = _lap(own, "extra_views", tm)
        score = smooth_scores(score, checked, params.score_smooth)

        sig_pair = (ref_feats[0], feat) if m == "pbp" else (None, None)
        verdict, reason, zones, changed = decide(score, checked, params, method_threshold(m, params),
                                                 *sig_pair, bright=bright)
        tm = _lap(own, "zones", tm)

        overlay = _jpeg(render_overlay(aligned, zones, unchecked_px))
        sha = _signature_sha256(ref, m, feat)
        _lap(own, "render", tm)
        hints = [HINT_CHANGE_ANGLE, HINT_BRIGHT_SPOT] if reason == "glare" else []
        out[m] = finish(m, verdict, reason, own, zones=zones, cov=cov, changed=changed, al=al,
                        images={"frame": frame_jpeg, "overlay": overlay}, sha=sha, hints=hints)
        out[m].score, out[m].checked, out[m].bright_spot = score, checked, bright
    return out


def inspect_frame(ref: Reference, frame_bgr: np.ndarray, params: Params,
                  method: str = "pbp") -> InspectResult:
    """Inspect one frame against the prepared reference with a single method."""
    return inspect_compare(ref, frame_bgr, params, (method,))[method]


# --------------------------------------------------------------------------- JSON

def zone_to_json(z: Zone, work_size: tuple[int, int]) -> dict:
    w, h = work_size
    x, y, bw, bh = (int(v) for v in z.bbox)
    norm = [x / w, y / h, bw / w, bh / h] if w and h else [0.0, 0.0, 0.0, 0.0]
    return {"id": int(z.id), "bbox": [x, y, bw, bh], "bbox_norm": [round(v, 5) for v in norm],
            "n_cells": int(z.n_cells), "area_px": int(z.area_px),
            "mean_score": round(float(z.mean_score), 4), "max_score": round(float(z.max_score), 4),
            "hint": z.hint, "contour": [[int(px), int(py)] for px, py in z.contour]}


def result_to_json(res: InspectResult) -> dict:
    """JSON-safe dict of an InspectResult (no image bytes)."""
    return {
        "verdict": res.verdict,
        "reason": res.reason,
        "zones": [zone_to_json(z, res.work_size) for z in res.zones],
        "coverage": round(float(res.coverage), 4),
        "changed_frac": round(float(res.changed_frac), 4),
        "align": dict(res.align),
        "timing_ms": dict(res.timing_ms),
        "method": res.method,
        "threshold": float(res.threshold),
        "signature_sha256": res.signature_sha256,
        "hints": list(res.hints),
        "work_size": [int(res.work_size[0]), int(res.work_size[1])],
    }
