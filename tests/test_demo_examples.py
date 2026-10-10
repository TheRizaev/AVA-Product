"""The built-in demo examples (web/examples/, docs/SPEC.md "Demo scenarios") give the expected verdicts.

Runs the REAL pipeline on the five shipped files. If the team replaces them with own photos
(same names), this test must still pass before a deploy.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from dalil import pipeline, surface
from dalil.config import Params

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
