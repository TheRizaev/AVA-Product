"""The built-in demo examples (web/examples/, docs/SPEC.md "Demo scenarios") give the expected verdicts.

Runs the REAL pipeline on the five shipped files. If the team replaces them with own photos
(same names), this test must still pass before a deploy.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from ava import pipeline, surface
from ava.config import Params

EXAMPLES = Path(__file__).resolve().parent.parent / "web" / "examples"
NAMES = ("surface_defect.jpg", "surface_ok.jpg", "compare_reference.jpg", "compare_defect.jpg",
         "compare_light.jpg")
P = Params()
METHODS = ("pbp", "pixdiff")


def load(name: str) -> np.ndarray:
    path = EXAMPLES / name
    assert path.is_file(), f"missing example {path} (run: python -m eval.make_examples)"
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.ndim == 3, name
    return img


def zones_of(res) -> list:
    return [(z.bbox, round(float(z.mean_score), 3)) for z in res.zones]


@pytest.fixture(scope="module")
def compare_results():
    ref = pipeline.prepare_reference([load("compare_reference.jpg")], P, METHODS)
    return {name: pipeline.inspect_compare(ref, load(f"compare_{name}.jpg"), P, METHODS)
            for name in ("defect", "light")}


def test_examples_exist_and_are_jpeg():
    for name in NAMES:
        data = (EXAMPLES / name).read_bytes()
        assert data[:2] == b"\xff\xd8", name
        assert len(data) < 15 * 1024 * 1024, name  # within the API upload limit
    assert (EXAMPLES / "README.md").is_file()


def test_surface_defect_fails():
    res = surface.scan_surface(load("surface_defect.jpg"), P)
    assert res.verdict == "fail" and res.reason is None, (res.verdict, res.reason)
    assert len(res.zones) >= 1


def test_surface_ok_passes():
    res = surface.scan_surface(load("surface_ok.jpg"), P)
    assert res.verdict == "pass" and res.zones == [], (res.verdict, res.reason, zones_of(res))


def test_compare_defect_fails_with_a_zone(compare_results):
    res = compare_results["defect"]["pbp"]
    assert res.verdict == "fail" and len(res.zones) >= 1, (res.verdict, res.reason, zones_of(res))


def test_compare_light_pbp_passes_and_plain_subtraction_does_not(compare_results):
    pbp, pix = compare_results["light"]["pbp"], compare_results["light"]["pixdiff"]
    assert pbp.verdict == "pass" and pbp.zones == [], (pbp.verdict, pbp.reason, zones_of(pbp))
    # Plain subtraction: false FAIL, or "mismatch" (almost the whole frame changed).
    assert pix.verdict == "fail" or pix.reason == "mismatch", (pix.verdict, pix.reason)


def scaled(img: np.ndarray, k: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * k, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def reference():
    return pipeline.prepare_reference([load("compare_reference.jpg")], P, METHODS)


@pytest.mark.parametrize("name,k", [("compare_reference.jpg", 0.5), ("compare_reference.jpg", 0.45),
                                    ("compare_light.jpg", 0.85), ("compare_light.jpg", 0.75)])
def test_pure_exposure_drop_passes(reference, name, k):
    # A darker Check photo turns low-contrast windows "flat" (absolute flat_floor) in the
    # frame only; that must not read as a change.
    res = pipeline.inspect_frame(reference, scaled(load(name), k), P)
    assert res.verdict == "pass" and res.zones == [], (res.verdict, res.reason, zones_of(res))


@pytest.mark.parametrize("gray", [252, 255])
def test_white_label_on_the_part_is_never_passed(reference, gray):
    img = load("compare_reference.jpg").copy()
    h, w = img.shape[:2]
    img[h // 2 - 30:h // 2 + 30, w // 2 - 45:w // 2 + 45] = gray
    for m, res in pipeline.inspect_compare(reference, img, P, METHODS).items():
        assert res.verdict != "pass", (m, res.verdict, res.reason)


@pytest.mark.parametrize("gray", [252, 255])
def test_white_label_on_a_surface_is_never_passed(gray):
    img = load("surface_ok.jpg").copy()
    h, w = img.shape[:2]
    img[h // 2 - 30:h // 2 + 30, w // 2 - 45:w // 2 + 45] = gray
    res = surface.scan_surface(img, P)
    assert res.verdict != "pass", (res.verdict, res.reason)


def test_overexposed_surface_defect_is_never_passed():
    # Brightened x1.3, the bright scratch and chip clip to 255 and fall under the glare mask.
    res = surface.scan_surface(scaled(load("surface_defect.jpg"), 1.3), P)
    assert res.verdict != "pass", (res.verdict, res.reason)
    assert res.verdict == "fail" or (res.reason == "glare" and "bright_spot" in res.hints)
