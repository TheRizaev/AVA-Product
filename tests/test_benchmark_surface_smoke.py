"""Smoke test of eval/benchmark_surface.py: a tiny calib -> test -> report run (small images, 1 process)."""

import json

import numpy as np
import pytest

from dalil.config import Params
from dalil.zones import Zone
from eval import benchmark_surface as bench


def test_wilson_and_auroc():
    lo, hi = bench.wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.35
    lo, hi = bench.wilson(10, 10)
    assert hi == 1.0 and 0.65 < lo < 0.75
    assert bench.auroc([3, 4], [1, 2]) == 1.0
    assert bench.auroc([1, 2], [1, 2]) == 0.5


def test_truth_matching_rules():
    mask = np.zeros((100, 100), bool)
    mask[40:60, 40:60] = True
    truth = bench.Truth(mask, near_px=8)

    def z(x, y, w, h):
        return Zone(id=1, bbox=(x, y, w, h), n_cells=1, area_px=w * h, mean_score=1.0,
                    max_score=1.0, rank_score=1.0)

    assert truth.match([z(40, 40, 20, 20)]) == (True, 0)
    assert truth.match([z(45, 45, 6, 6)]) == (False, 0)        # inside but small: no match, not false
    assert truth.match([z(42, 42, 12, 12)]) == (True, 0)       # covers 36 % of the bbox
    assert truth.match([z(0, 0, 10, 10)]) == (False, 1)        # far away: a false zone
    empty = bench.Truth(np.zeros((100, 100), bool), near_px=8)
    assert empty.match([z(0, 0, 10, 10), z(40, 40, 5, 5)]) == (False, 2)


def test_calibration_rule_is_stable_from_threshold_upwards():
    """The chosen threshold must keep FA <= target at every larger grid threshold."""
    grid = np.array([0.1, 0.2, 0.3, 0.4])
    # FA per threshold over 4 good photos: 1.0, 0.0 (lucky dip), 0.5, 0.0  -> pick 0.4
    codes = [[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
    recs = []
    for c in codes:
        sweep = np.zeros((4, 4), np.int16)
        sweep[:, 0] = c
        recs.append({"defect": "none", "lighting": "none", "methods": {"pbp": {"sweep": sweep}}})
    cal = bench.calibrate(recs, ["pbp"], {"pbp": grid}, 0.05)
    assert cal["pbp"]["threshold"] == 0.4 and cal["pbp"]["reached"]


def test_params_defaults_are_the_calibrated_thresholds():
    """dalil.config defaults must be the thresholds frozen by the committed calibration run."""
    path = bench.ROOT / "eval" / "results" / "surface_calib.json"
    if not path.exists():
        pytest.skip("no committed surface calibration run")
    calib = json.loads(path.read_text(encoding="utf-8"))
    P = Params()
    for method, thr in calib["thresholds"].items():
        assert getattr(P, bench.method_param(method)) == pytest.approx(thr), method


def test_tiny_benchmark_end_to_end(tmp_path):
    out, report = tmp_path / "results", tmp_path / "RESULTS_SURFACE.md"
    args = ["--n", "1", "--clean-mult", "1", "--kinds", "painted", "--lightings", "none,gamma,specular",
            "--defects", "none,chip", "--methods", "pbp,gradient", "--size", "288x512", "--workers", "1",
            "--timing-photos", "1", "--out", str(out), "--report", str(report)]
    assert bench.main(args) == 0

    calib = json.loads((out / "surface_calib.json").read_text(encoding="utf-8"))
    assert set(calib["thresholds"]) == {"pbp", "gradient"}
    assert calib["n_photos"] == 6 and calib["consistency_failures"] == 0 and not calib["errors"]
    test = json.loads((out / "surface_test_summary.json").read_text(encoding="utf-8"))
    assert test["thresholds"] == calib["thresholds"] and test["consistency_failures"] == 0
    s = test["summary"]["all"]["pbp"]
    assert s["n_photos"] == 6 and s["n_defect"] == 3 and s["n_good"] == 3
    assert 0.0 <= s["recall"] <= 1.0 and 0.0 <= s["fa_rate"] <= 1.0
    lo, hi = s["recall_ci"]
    assert 0.0 <= lo <= s["recall"] <= hi <= 1.0
    assert "pbp" in test["timing"] and test["timing"]["pbp"]["n"] >= 1
    sweep = json.loads((out / "surface_test_sweep.json").read_text(encoding="utf-8"))
    for m in ("pbp", "gradient"):
        assert len(sweep[m]["thresholds"]) == len(sweep[m]["all"]["recall"]) == len(sweep[m]["all"]["fa_rate"])
        assert "none+monotone" in sweep[m] and "non_monotone" in sweep[m]

    # TEST seeds are disjoint from CALIB seeds.
    photos = json.loads((out / "surface_test_photos.json").read_text(encoding="utf-8"))
    assert {p["entropy"][1] for p in photos} == {bench.SPLIT_CODES["test"]}

    text = report.read_text(encoding="utf-8")
    assert text.startswith(bench.TITLE)
    assert text.startswith("# Synthetic benchmark — procedurally generated uniform surfaces, NOT real photos")
    assert "Wilson" in text and "only the surface" in text and "expected failure" in text
    # The report can be rebuilt from the saved JSON alone.
    report2 = tmp_path / "R2.md"
    assert bench.main(["--split", "report", "--out", str(out), "--report", str(report2)]) == 0
    assert report2.read_text(encoding="utf-8") == text
