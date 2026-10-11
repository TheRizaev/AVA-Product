"""End-to-end tests of ava.pipeline on synthetic station pairs (ava.synth).

Pairs are generated at 576x1024 (= the default working size, so no resize) with
fixed seeds; each scenario runs over 5 seeds. Detection rates asserted here are
those of the current default Params (threshold calibration is eval/benchmark.py's job).
"""

import hashlib
import json

import cv2
import numpy as np
import pytest

from ava import synth
from ava.align import align
from ava.config import Params
from ava.pbp import serialize_signatures, window_signatures
from ava.pipeline import (InspectResult, Reference, inspect_compare, inspect_frame,
                            prepare_reference, result_to_json)

H, W = 576, 1024
SEEDS = range(5)
P = Params()


def make_pair(seed: int, **kw) -> dict:
    return synth.make_station_pair(np.random.default_rng(seed), h=H, w=W, **kw)


def run(pair: dict, method: str = "pbp"):
    ref = prepare_reference([pair["reference"]], P, methods=(method,))
    return ref, inspect_frame(ref, pair["frame"], P, method)


def top_zone_hits(ref: Reference, res: InspectResult, mask: np.ndarray, tol_px: int = 8) -> bool:
    """Does the top-ranked zone (work coords) overlap the true defect mask (pair coords)?

    The mask is dilated by `tol_px` (one PBP window): zones live on the window grid.
    """
    near = cv2.dilate(mask.astype(np.uint8), np.ones((2 * tol_px + 1,) * 2, np.uint8)) > 0
    x, y, w, h = (int(round(v / ref.scale)) for v in res.zones[0].bbox)
    return bool(near[y:y + h, x:x + w].any())


def describe(res: InspectResult) -> str:
    return (f"{res.verdict}/{res.reason} zones={[(z.bbox, round(z.mean_score, 2)) for z in res.zones]} "
            f"align={res.align} cov={res.coverage:.2f}")


# --------------------------------------------------------------------------- scenarios

@pytest.mark.parametrize("lighting", synth.MONOTONE_LIGHTING)
def test_monotone_factory_lighting_with_jitter_passes(lighting):
    for seed in SEEDS:
        pair = make_pair(100 + seed, lighting=lighting, motion="jitter")
        _, res = run(pair)
        assert res.verdict == "pass" and res.zones == [], f"seed {seed} {pair['meta']['part_kind']}: {describe(res)}"


# Hit counts at the calibrated default threshold (eval/results/calib.json). A thin scratch
# is found on 3 of these 5 seeds; burr (a small blob on an edge) on none of them, so it is
# not asserted here - its recall is reported per class in eval/RESULTS.md.
@pytest.mark.parametrize("defect,min_hits", [("scratch", 3), ("tape", 5), ("missing_hole", 5)])
def test_defects_fail_and_top_zone_is_on_the_defect(defect, min_hits):
    hits, log = 0, []
    for seed in SEEDS:
        pair = make_pair(200 + seed, defect=defect, lighting=synth.MONOTONE_LIGHTING[seed])
        ref, res = run(pair)
        ok = res.verdict == "fail" and top_zone_hits(ref, res, pair["mask_ref_frame"])
        hits += ok
        log.append(f"seed {seed} {pair['meta']['part_kind']} {pair['meta']['lighting']}: {describe(res)}")
    assert hits >= min_hits, "\n".join(log)


def test_wrong_part_is_never_passed():
    kinds = synth.PART_KINDS
    for seed in SEEDS:
        golden = make_pair(400 + seed, part_kind=kinds[seed % 4])
        other = make_pair(500 + seed, part_kind=kinds[(seed + 1) % 4])
        ref = prepare_reference([golden["reference"]], P, methods=("pbp",))
        res = inspect_frame(ref, other["frame"], P)
        assert res.verdict != "pass", f"seed {seed}: {describe(res)}"
        assert res.verdict == "retake" or res.zones


def test_part_moved_on_belt_is_registered_and_passes():
    registered, log = 0, []
    for seed in SEEDS:
        pair = make_pair(300 + seed, motion="shift")
        ref, res = run(pair)
        assert res.verdict == "pass", f"seed {seed} {pair['meta']['part_kind']}: {describe(res)}"
        assert res.align["method"] in ("orb", "ecc")
        # The registration follows the PART (not the static belt): the part's centroid,
        # moved by the true motion, must come back to where it was. (Corners would be
        # unfair: the round "machined" flange is rotation-symmetric.)
        al = align(ref.images[0], pair["frame"], P)
        ys, xs = np.nonzero(pair["meta"]["part_mask"])
        c = np.float32([[xs.mean(), ys.mean()]]).reshape(-1, 1, 2)
        back = cv2.perspectiveTransform(cv2.perspectiveTransform(c, pair["meta"]["H"]), al.H)
        err = float(np.abs(back - c).max())
        registered += err < 2.0
        log.append(f"seed {seed} {pair['meta']['part_kind']}/{pair['meta']['background']}: centre error {err:.2f} px")
    # Known: a round flange on the repetitive ESD mat (seed 4) gets an approximate (~5 px)
    # registration; the verdict is still right (part ROI + camera-fixed view).
    assert registered >= 4, "\n".join(log)


def test_inspect_compare_runs_pbp_and_pixdiff_on_one_alignment():
    pair = make_pair(600, defect="tape", lighting="gamma")
    ref = prepare_reference([pair["reference"]], P)
    out = inspect_compare(ref, pair["frame"], P)
    assert set(out) == {"pbp", "pixdiff"}
    pbp, pix = out["pbp"], out["pixdiff"]
    assert pbp.method == "pbp" and pix.method == "pixdiff"
    assert pbp.threshold == P.threshold and pix.threshold == P.threshold_pixdiff
    assert pbp.align == pix.align
    assert pbp.images["frame"] == pix.images["frame"]  # same aligned frame
    for res in (pbp, pix):
        assert res.verdict in ("pass", "fail", "retake")
        for key in ("frame", "overlay"):
            assert res.images[key][:2] == b"\xff\xd8"  # JPEG
        assert len(res.signature_sha256) == 64
    assert pbp.verdict == "fail"
    assert pbp.signature_sha256 != pix.signature_sha256


def test_handheld_camera_motion_does_not_hide_a_defect():
    # The whole frame is moved (handheld camera, no second motion): no static-background
    # view may be min-ed in. On the old code this burr passed (only the unregistered frame
    # "matched" the reference there).
    pair = make_pair(22, part_kind=synth.PART_KINDS[22 % 4], defect="burr")
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 2.0, 1.0)
    M[:, 2] += (-12, 8)
    frame = cv2.warpAffine(pair["frame"], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    ref, res = run(pair)
    res = inspect_frame(ref, frame, P)
    assert res.align["method"] == "orb"
    assert res.verdict == "fail" and res.zones, describe(res)


@pytest.mark.parametrize("ref_side,frame_side", [(480, 1280), (1280, 640), (1280, 400)])
def test_reference_and_check_at_different_resolutions_align(ref_side, frame_side):
    # Each photo is resized to the working size independently; the frame is brought to the
    # reference's scale before alignment, so a resolution mismatch is not a RETAKE.
    pair = synth.make_station_pair(np.random.default_rng(606), part_kind="stamped", h=720, w=1280)
    small = lambda img, side: cv2.resize(img, (side, round(side * 720 / 1280)), interpolation=cv2.INTER_AREA)
    ref = prepare_reference([small(pair["reference"], ref_side)], P, methods=("pbp",))
    res = inspect_frame(ref, small(pair["frame"], frame_side), P)
    assert res.verdict == "pass", describe(res)


def test_new_bright_blob_is_never_a_silent_pass():
    # A white label clipped to 255 looks exactly like glare (bright, unsaturated) and is
    # masked; the verdict must then be RETAKE (glare + bright_spot), never PASS.
    pair = make_pair(607, part_kind="stamped")
    ref = prepare_reference([pair["reference"]], P)
    frame = pair["frame"].copy()
    frame[H // 2 - 30:H // 2 + 30, W // 2 - 45:W // 2 + 45] = 255
    for res in inspect_compare(ref, frame, P).values():
        assert res.verdict != "pass", describe(res)
        if res.verdict == "retake":
            assert res.reason == "glare" and "bright_spot" in res.hints
    clean = inspect_frame(ref, pair["frame"], P)
    assert clean.verdict == "pass" and "bright_spot" not in clean.hints, describe(clean)


# --------------------------------------------------------------------------- contracts

def test_result_json_contract_and_fingerprint():
    pair = make_pair(601, defect="scratch", lighting="drift_gain")
    ref, res = run(pair)
    js = result_to_json(res)
    json.dumps(js)  # JSON-safe, no bytes
    assert set(js) >= {"verdict", "reason", "zones", "coverage", "changed_frac", "align", "timing_ms",
                       "method", "threshold", "signature_sha256", "hints"}
    assert js["verdict"] == "fail" and js["zones"]
    for z in js["zones"]:
        assert len(z["bbox"]) == 4 and all(0.0 <= v <= 1.0 for v in z["bbox_norm"])
    assert {"resize", "quality", "align", "features", "change", "zones", "render", "total"} <= set(js["timing_ms"])
    stages = sum(v for k, v in js["timing_ms"].items() if k != "total")
    assert js["timing_ms"]["total"] == pytest.approx(stages, abs=0.5)

    # sha256(serialize(reference shot 0) + serialize(frame)), recomputed independently.
    al = align(ref.images[0], pair["frame"], P)
    gray = cv2.GaussianBlur(cv2.cvtColor(al.after_warped, cv2.COLOR_BGR2GRAY).astype(np.float32),
                            (0, 0), P.blur_sigma)
    expected = hashlib.sha256(serialize_signatures(ref.features["pbp"][0])
                              + serialize_signatures(window_signatures(gray, P))).hexdigest()
    assert res.signature_sha256 == expected


def test_overlay_marks_zones_in_red():
    pair = make_pair(602, defect="tape")
    _, res = run(pair)
    assert res.verdict == "fail"
    frame = cv2.imdecode(np.frombuffer(res.images["frame"], np.uint8), cv2.IMREAD_COLOR)
    overlay = cv2.imdecode(np.frombuffer(res.images["overlay"], np.uint8), cv2.IMREAD_COLOR)
    assert frame.shape == overlay.shape == (H, W, 3)
    x, y, w, h = res.zones[0].bbox
    edge = overlay[y, x:x + w].astype(int)  # top edge of the zone box
    assert ((edge[:, 2] > 150) & (edge[:, 0] < 120)).mean() > 0.5


def test_quality_gates_retake():
    pair = make_pair(603)
    ref = prepare_reference([pair["reference"]], P, methods=("pbp",))
    blurry = cv2.GaussianBlur(pair["frame"], (0, 0), 4)
    assert inspect_frame(ref, blurry, P).reason == "blurry"
    dark = (pair["frame"] * 0.1).astype(np.uint8)
    res = inspect_frame(ref, dark, P)
    assert (res.verdict, res.reason) == ("retake", "dark")
    glare = pair["frame"].copy()
    glare[:, : W // 2] = 255
    res = inspect_frame(ref, glare, P)
    assert (res.verdict, res.reason) == ("retake", "glare") and "change_angle" in res.hints
    noise = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    res = inspect_frame(ref, noise, P)
    assert res.verdict == "retake" and res.reason in ("alignment", "mismatch")


def test_reference_set_registers_extra_shots_and_rejects_strangers():
    pair = make_pair(604, lighting="gamma", motion="jitter")
    shots = synth.make_reference_set(np.random.default_rng(1), pair["meta"]["reference_clean"], 3)
    ref = prepare_reference(shots, P)
    assert len(ref.images) == len(ref.grays) == 3
    assert len(ref.features["pbp"]) == len(ref.features["pixdiff"]) == 3
    assert ref.work_size == (W, H) and ref.scale == 1.0
    assert ref.outline_png[:4] == b"\x89PNG"
    res = inspect_frame(ref, pair["frame"], P)
    assert res.verdict == "pass", describe(res)

    stranger = make_pair(605, part_kind="machined" if pair["meta"]["part_kind"] != "machined" else "stamped")
    with pytest.raises(ValueError):
        prepare_reference([pair["reference"], stranger["frame"]], P)
    with pytest.raises(ValueError):
        prepare_reference([], P)
