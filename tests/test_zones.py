"""Tests for dalil.zones on small hand-made cell grids."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dalil.config import Params
from dalil.zones import Zone, coverage, extract_zones

GH, GW = 40, 60
WIN, STEP = 8, 4


def grid():
    """Empty score grid and an all-checked mask."""
    return np.zeros((GH, GW), np.float32), np.ones((GH, GW), bool)


def paint(score, i0, j0, h, w, value=0.8):
    score[i0:i0 + h, j0:j0 + w] = value
    return score


def sigs(degree=6, contrast=50.0):
    """Duck-typed signature object with uniform degree/contrast."""
    return SimpleNamespace(
        degree=np.full((GH, GW, 2), degree, np.int8),
        contrast=np.full((GH, GW), contrast, np.float32),
    )


def drops(score, n_rows=2, n_cols=4, i0=5, j0=5, value=0.8):
    """Grid of separate 2x2-cell blobs, 2 empty cells apart (pixel gap 4 px)."""
    for r in range(n_rows):
        for c in range(n_cols):
            paint(score, i0 + 4 * r, j0 + 4 * c, 2, 2, value)
    return score


def test_single_blob_bbox_and_stats():
    score, checked = grid()
    paint(score, 5, 10, 5, 5, 0.8)
    zones = extract_zones(score, checked, Params(), win=WIN, step=STEP)
    assert len(zones) == 1
    z = zones[0]
    assert isinstance(z, Zone)
    assert z.id == 1
    # cells rows 5..9, cols 10..14 -> pixels x [40, 64), y [20, 44)
    assert z.bbox == (40, 20, 24, 24)
    assert z.n_cells == 25
    assert z.area_px == 24 * 24
    assert z.mean_score == pytest.approx(0.8)
    assert z.max_score == pytest.approx(0.8)
    assert z.rank_score == pytest.approx(25 * 0.8)
    assert z.hint is None
    assert len(z.contour) >= 4
    xs = [p[0] for p in z.contour]
    ys = [p[1] for p in z.contour]
    assert min(xs) >= 40 and max(xs) <= 64 and min(ys) >= 20 and max(ys) <= 44


def test_no_change_gives_no_zones():
    score, checked = grid()
    assert extract_zones(score, checked, Params()) == []


def test_two_far_blobs_ranked_by_area_times_score():
    score, checked = grid()
    paint(score, 2, 2, 3, 3, 0.95)      # 9 cells * 0.95 = 8.55
    paint(score, 25, 40, 6, 6, 0.7)     # 36 cells * 0.7 = 25.2
    zones = extract_zones(score, checked, Params())
    assert len(zones) == 2
    assert [z.id for z in zones] == [1, 2]
    assert zones[0].n_cells == 36 and zones[1].n_cells == 9
    assert zones[0].rank_score > zones[1].rank_score
    assert zones[0].bbox == (160, 100, 28, 28)


def test_tiny_blob_dropped():
    score, checked = grid()
    paint(score, 10, 10, 1, 3, 0.9)     # 3 cells < min_zone_cells=4
    paint(score, 30, 40, 4, 4, 0.9)
    zones = extract_zones(score, checked, Params())
    assert len(zones) == 1
    assert zones[0].n_cells == 16


def test_below_threshold_ignored():
    score, checked = grid()
    paint(score, 5, 5, 5, 5, 0.2)       # below the default threshold
    assert extract_zones(score, checked, Params()) == []


def test_unchecked_cells_excluded():
    score, checked = grid()
    paint(score, 5, 10, 5, 5, 0.8)
    checked[5:10, 10:15] = False
    assert extract_zones(score, checked, Params()) == []

    checked[:] = True
    checked[5:10, 10:12] = False        # left 2 columns unchecked
    zones = extract_zones(score, checked, Params())
    assert len(zones) == 1
    assert zones[0].n_cells == 15
    assert zones[0].bbox[0] == 48


def test_nearby_fragments_merged():
    score, checked = grid()
    paint(score, 10, 10, 3, 3, 0.9)     # pixels x [40, 56)
    paint(score, 10, 15, 3, 3, 0.7)     # pixels x [60, 76) -> gap 4 px < 12
    zones = extract_zones(score, checked, Params())
    assert len(zones) == 1
    z = zones[0]
    assert z.n_cells == 18
    assert z.bbox == (40, 40, 36, 16)
    assert z.mean_score == pytest.approx(0.8)
    assert z.max_score == pytest.approx(0.9)

    # with merging disabled they stay apart
    zones = extract_zones(score, checked, replace(Params(), merge_gap_px=0))
    assert len(zones) == 2


def test_max_zones_cap():
    score, checked = grid()
    for k in range(5):
        paint(score, 2 + 7 * k, 2, 3, 3, 0.5 + 0.1 * k)
    zones = extract_zones(score, checked, replace(Params(), max_zones=3))
    assert len(zones) == 3
    assert [round(z.mean_score, 2) for z in zones] == [0.9, 0.8, 0.7]


def test_many_small_round_blobs_hint_water():
    score, checked = grid()
    drops(score)
    zones = extract_zones(score, checked, Params(), sigs(), sigs(degree=1))
    assert len(zones) == 1
    assert zones[0].n_cells == 8 * 4
    assert zones[0].hint == "water_or_dirt"   # takes precedence over the degree drop


def test_long_streaks_are_not_water():
    score, checked = grid()
    for k in range(7):
        paint(score, 5 + 3 * k, 5, 1, 20, 0.8)  # thin scratches, 2 cells apart
    zones = extract_zones(score, checked, Params())
    assert len(zones) == 1
    assert zones[0].hint != "water_or_dirt"


def test_degree_drop_hint_removed():
    score, checked = grid()
    paint(score, 5, 10, 5, 5, 0.8)
    after = sigs()
    after.degree[5:10, 10:15] = 1
    zones = extract_zones(score, checked, Params(), sig_before=sigs(), sig_after=after)
    assert zones[0].hint == "something_removed"


def test_degree_rise_hint_added():
    score, checked = grid()
    paint(score, 5, 10, 5, 5, 0.8)
    before = sigs(degree=1, contrast=8.0)
    zones = extract_zones(score, checked, Params(), sig_before=before,
                          sig_after=sigs(degree=6, contrast=60.0))
    assert zones[0].hint == "something_added"


def test_similar_activity_no_hint():
    score, checked = grid()
    paint(score, 5, 10, 5, 5, 0.8)
    zones = extract_zones(score, checked, Params(), sig_before=sigs(),
                          sig_after=sigs(degree=5, contrast=45.0))
    assert zones[0].hint is None


def test_coverage():
    checked = np.ones((GH, GW), bool)
    assert coverage(checked) == pytest.approx(1.0)
    checked[:10] = False
    assert coverage(checked) == pytest.approx(30 / 40)
    assert coverage(np.zeros((4, 4), bool)) == 0.0
    assert coverage(np.zeros((0, 0), bool)) == 0.0
