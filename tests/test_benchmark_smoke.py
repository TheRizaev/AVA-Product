"""Smoke test of eval/benchmark.py: a tiny calib -> test -> report run (small images, 1 process)."""

import json

import numpy as np
import pytest

from ava.config import Params
from ava.zones import Zone
from eval import benchmark as bench


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

    assert truth.match([z(40, 40, 20, 20)]) == (True, 0)       # exact
    assert truth.match([z(45, 45, 6, 6)]) == (False, 0)        # inside but small: no match, not false
    assert truth.match([z(42, 42, 12, 12)]) == (True, 0)       # covers 36 % of the bbox
    assert truth.match([z(0, 0, 10, 10)]) == (False, 1)        # far away: a false zone
    empty = bench.Truth(np.zeros((100, 100), bool), near_px=8)
    assert empty.match([z(0, 0, 10, 10), z(40, 40, 5, 5)]) == (False, 2)


def test_params_defaults_are_the_calibrated_thresholds():
    """ava.config defaults must be the thresholds frozen by the committed calibration run."""
    path = bench.ROOT / "eval" / "results" / "calib.json"
    if not path.exists():
        pytest.skip("no committed calibration run")
    calib = json.loads(path.read_text(encoding="utf-8"))
    P = Params()
    for method, thr in calib["thresholds"].items():
        assert getattr(P, bench.method_param(method)) == pytest.approx(thr), method


def test_tiny_benchmark_end_to_end(tmp_path):
    out, report = tmp_path / "results", tmp_path / "RESULTS.md"
    args = ["--n", "1", "--parts", "stamped", "--lightings", "none,gamma,specular", "--motions", "jitter",
            "--defects", "none,tape", "--methods", "pbp,pixdiff", "--size", "288x512", "--workers", "1",
            "--timing-pairs", "1", "--out", str(out), "--report", str(report)]
    assert bench.main(args) == 0

    calib = json.loads((out / "calib.json").read_text(encoding="utf-8"))
    assert set(calib["thresholds"]) == {"pbp", "pixdiff"}
    assert calib["n_pairs"] == 6 and calib["consistency_failures"] == 0
    test = json.loads((out / "test_summary.json").read_text(encoding="utf-8"))
    assert test["thresholds"] == calib["thresholds"] and test["consistency_failures"] == 0
    s = test["summary"]["all"]["pbp"]
    assert s["n_pairs"] == 6 and s["n_defect"] == 3 and s["n_good"] == 3
    assert 0.0 <= s["recall"] <= 1.0 and 0.0 <= s["fa_rate"] <= 1.0
    sweep = json.loads((out / "test_sweep.json").read_text(encoding="utf-8"))
    assert len(sweep["pbp"]["thresholds"]) == len(sweep["pbp"]["all"]["recall"])

    # TEST seeds are disjoint from CALIB seeds.
    pairs = json.loads((out / "test_pairs.json").read_text(encoding="utf-8"))
    assert {tuple(p["entropy"])[1] for p in pairs} == {bench.SPLIT_CODES["test"]}

    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Synthetic benchmark — procedurally generated parts, NOT real factory images")
    assert "Wilson" in text and "pending" in text
    # The report can be rebuilt from the saved JSON alone.
    report2 = tmp_path / "R2.md"
    assert bench.main(["--split", "report", "--out", str(out), "--report", str(report2)]) == 0
    assert report2.read_text(encoding="utf-8") == text
