"""Tests for dalil.pbp: exact PBP, its invariants, vectorised signatures, distance, speed."""

import itertools
import os
import platform
import time
from dataclasses import replace

import cv2
import numpy as np
import pytest

from dalil import pbp
from dalil.config import Params

P = Params()                                   # win=8, step=4
P_EXACT = replace(P, eps=0.0, flat_floor=0.0)  # exact PBP, nothing treated as flat


# --------------------------------------------------------------------------- #
# Exact reference implementation
# --------------------------------------------------------------------------- #

def test_published_example_polynomial():
    # Matrix reconstructed to reproduce the published example polynomial of
    # Chikake & Goldengorin, Computer Optics 2025 (rows are the variables y1..y4).
    c = np.array([[9, 6, 14],
                  [11, 10, 10],
                  [4, 16, 4],
                  [17, 3, 20]])
    poly = pbp.pbp_reference(c)
    assert pbp.format_polynomial(poly) == \
        "11 + 11y3 + 3y4 + 2y1y3 + 4y2y3 + 4y1y4 + 12y1y2y3 + 6y1y2y4"
    assert pbp.poly_degree(poly) == 3


def _monomials(poly):
    return {k for k, v in poly.items() if k and v != 0}


def _random_matrices(n=2000, seed=0):
    """Small integer matrices with many ties (values 0..4)."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        m, k = rng.integers(1, 7, size=2)
        yield rng.integers(0, 5, size=(m, k)).astype(np.float64), rng


def _monotone_transforms(rng):
    """Strictly increasing maps of the values 0..4: gamma, positive affine, random LUT."""
    lut = np.cumsum(rng.integers(1, 40, size=5)).astype(np.float64) + rng.integers(0, 50)
    gamma = float(rng.uniform(0.3, 3.0))
    a, b = float(rng.uniform(0.1, 7.0)), float(rng.uniform(-30, 30))
    return [
        lambda x: 255.0 * ((x + 1) / 5.0) ** gamma,
        lambda x: a * x + b,
        lambda x: lut[x.astype(int)],
    ]


def test_properties_on_random_matrices_with_ties():
    for c, rng in _random_matrices():
        poly = pbp.pbp_reference(c)
        col_min, col_max = c.min(axis=0), c.max(axis=0)

        # Degree = max over columns of the number of entries strictly below the column max.
        assert pbp.poly_degree(poly) == int((c < col_max).sum(axis=0).max())
        # Constant = sum of column minima; non-constant coefficients sum to sum(max - min).
        assert poly[()] == pytest.approx(col_min.sum())
        assert sum(v for k, v in poly.items() if k) == pytest.approx((col_max - col_min).sum())
        assert all(v > 0 for k, v in poly.items() if k)
        # The monomial set only depends on the per-column order.
        mono = _monomials(poly)
        for f in _monotone_transforms(rng):
            assert _monomials(pbp.pbp_reference(f(c))) == mono


def test_polynomial_evaluates_to_min_over_open_rows():
    """Independent check: f(y) = sum_j min_{i: y_i = 0} C[i, j] (column max if all y_i = 1)."""
    for c, _ in itertools.islice(_random_matrices(seed=1), 300):
        poly = pbp.pbp_reference(c)
        m = c.shape[0]
        for y in itertools.product((0, 1), repeat=m):
            value = sum(v * all(y[i] for i in k) for k, v in poly.items())
            open_rows = [i for i in range(m) if y[i] == 0]
            expected = (c[open_rows].min(axis=0) if open_rows else c.max(axis=0)).sum()
            assert value == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Vectorised window signatures
# --------------------------------------------------------------------------- #

def _expected_vector(window: np.ndarray) -> np.ndarray:
    """Normalised 256-bin vector of a window's exact PBP (bin = bitmask of the variables)."""
    vec = np.zeros(pbp.NBINS)
    for key, v in pbp.pbp_reference(window).items():
        if key:
            vec[sum(1 << i for i in key)] += v
    total = vec.sum()
    return vec / total if total > 0 else vec


def _check_against_reference(img: np.ndarray, params: Params) -> None:
    sig = pbp.window_signatures(img, params)
    gh, gw = pbp.grid_shape(*img.shape, params.win, params.step)
    assert sig.coeffs.shape == (gh, gw, 2, pbp.NBINS)
    for i in range(gh):
        for j in range(gw):
            w = img[i * params.step:i * params.step + 8, j * params.step:j * params.step + 8]
            # Orientation 0: variables = window rows; orientation 1 = transposed window.
            for o, patch in enumerate((w, w.T)):
                np.testing.assert_allclose(sig.coeffs[i, j, o], _expected_vector(patch), atol=1e-6)
                assert sig.degree[i, j, o] == pbp.poly_degree(pbp.pbp_reference(patch))


def test_window_signatures_match_reference_single_windows():
    rng = np.random.default_rng(2)
    for t in range(300):
        if t % 3 == 0:
            img = rng.integers(0, 4, (8, 8)).astype(np.float32)      # heavy ties
        elif t % 3 == 1:
            img = rng.integers(0, 256, (8, 8)).astype(np.float32)
        else:
            img = rng.uniform(0, 255, (8, 8)).astype(np.float32)     # non-integer levels
        _check_against_reference(img, P_EXACT)


def test_window_signatures_match_reference_on_grid():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 12, (36, 52)).astype(np.float32)
    _check_against_reference(img, P_EXACT)


def test_coefficients_normalised():
    rng = np.random.default_rng(4)
    img = cv2.GaussianBlur(rng.uniform(0, 255, (64, 96)).astype(np.float32), (0, 0), 1.0)
    sig = pbp.window_signatures(img, P)
    sums = sig.coeffs.sum(axis=-1)
    assert np.all((np.abs(sums - 1) < 1e-4) | (sums == 0))
    assert np.all(sig.coeffs[..., 0] == 0)            # the constant is never a bin


def _monotone_image_transforms(rng):
    lut = np.cumsum(rng.uniform(0.2, 3.0, 256)).astype(np.float32)
    return {
        "gamma_0.45": lambda g: 255.0 * (g / 255.0) ** 0.45,
        "gamma_2.2": lambda g: 255.0 * (g / 255.0) ** 2.2,
        "affine": lambda g: 0.37 * g + 41.0,
        "lut": lambda g: lut[g.astype(np.int64)],
    }


def test_support_invariant_under_monotone_image_transforms():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (72, 96)).astype(np.float32)
    img[:24, :32] = rng.integers(100, 104, (24, 32))          # low-contrast area with ties
    support = pbp.window_signatures(img, P_EXACT).coeffs > 0
    for name, f in _monotone_image_transforms(rng).items():
        out = pbp.window_signatures(f(img).astype(np.float32), P_EXACT)
        assert np.array_equal(out.coeffs > 0, support), name


def test_affine_invariance_with_eps_ties():
    """eps-ties scale with the window contrast, so a positive affine change is exact too."""
    rng = np.random.default_rng(6)
    img = rng.integers(0, 256, (40, 56)).astype(np.float32)
    a = pbp.window_signatures(img, replace(P, flat_floor=0.0))
    b = pbp.window_signatures(0.5 * img + 20.0, replace(P, flat_floor=0.0))
    np.testing.assert_allclose(a.coeffs, b.coeffs, atol=1e-5)
    assert np.array_equal(a.degree, b.degree)


def test_flat_windows():
    img = np.full((24, 32), 120.0, np.float32)
    img[:, 16:] += np.random.default_rng(7).uniform(0, 3, (24, 16)).astype(np.float32)
    sig = pbp.window_signatures(img, P)                       # flat_floor = 6
    assert sig.flat.all()
    assert not sig.coeffs.any() and not sig.degree.any()
    # A constant window without the flat floor: still all-zero, no NaNs.
    sig0 = pbp.window_signatures(np.full((8, 8), 50.0, np.float32), P_EXACT)
    assert not sig0.flat.any() and not sig0.coeffs.any() and np.isfinite(sig0.coeffs).all()


def test_row_constant_window_has_one_empty_orientation():
    w = np.repeat(np.arange(8, dtype=np.float32)[:, None] * 10, 8, axis=1)   # rows constant
    sig = pbp.window_signatures(w, P)
    assert sig.coeffs[0, 0, 0].sum() == pytest.approx(1.0)
    assert sig.coeffs[0, 0, 1].sum() == 0 and sig.degree[0, 0, 1] == 0
    assert sig.degree[0, 0, 0] == 7


def test_bgr_input_and_small_image():
    rng = np.random.default_rng(8)
    bgr = rng.integers(0, 256, (20, 28, 3), dtype=np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    np.testing.assert_array_equal(pbp.window_signatures(bgr, P).coeffs,
                                  pbp.window_signatures(gray, P).coeffs)
    tiny = pbp.window_signatures(np.zeros((5, 40), np.float32), P)
    assert tiny.coeffs.shape == (0, 9, 2, pbp.NBINS)


def test_sorting_network_sorts_all_binary_inputs():
    bits = np.array(list(itertools.product((0, 1), repeat=8)), np.int32).T   # [8, 256]
    out = np.stack(pbp._sort8([bits[i].copy() for i in range(8)]))
    assert np.array_equal(out, np.sort(bits, axis=0))


# --------------------------------------------------------------------------- #
# Distance, grid helpers, serialisation
# --------------------------------------------------------------------------- #

def test_pair_distance_contract():
    rng = np.random.default_rng(9)
    g1 = cv2.GaussianBlur(rng.uniform(0, 255, (48, 64)).astype(np.float32), (0, 0), 1.0)
    g2 = cv2.GaussianBlur(rng.uniform(0, 255, (48, 64)).astype(np.float32), (0, 0), 1.0)
    g1[:16, :16] = 80.0
    g2[:16, :24] = 200.0                                       # flat in both / only in g2
    a, b = pbp.window_signatures(g1, P), pbp.window_signatures(g2, P)

    assert np.all(pbp.pair_distance(a, a) == 0)
    d = pbp.pair_distance(a, b)
    assert d.dtype == np.float32 and d.shape == a.flat.shape
    assert np.all((d >= 0) & (d <= 1))
    assert np.all(d[a.flat & b.flat] == 0)
    assert np.all(d[a.flat ^ b.flat] == 1)
    assert d[~a.flat & ~b.flat].mean() > 0.2

    # Shifts: out-of-range partners are +inf; symmetric under swapping a and b.
    gh, gw = d.shape
    for dy, dx in [(1, 0), (0, -2), (-1, 1), (gh, 0)]:
        dab, dba = pbp.pair_distance(a, b, dy, dx), pbp.pair_distance(b, a, -dy, -dx)
        valid = np.zeros((gh, gw), bool)
        valid[max(0, -dy):min(gh, gh - dy), max(0, -dx):min(gw, gw - dx)] = True
        assert np.all(np.isinf(dab[~valid])) and np.all(np.isfinite(dab[valid]))
        ii, jj = np.nonzero(valid)
        np.testing.assert_array_equal(dab[ii, jj], dba[ii + dy, jj + dx])


def test_pair_distance_matches_numpy_l1():
    rng = np.random.default_rng(10)
    a = pbp.window_signatures(rng.uniform(0, 255, (40, 60)).astype(np.float32), P)
    b = pbp.window_signatures(rng.uniform(0, 255, (44, 56)).astype(np.float32), P)
    d = pbp.pair_distance(a, b, 1, -1)
    i, j = 3, 5
    expected = 0.25 * np.abs(a.coeffs[i, j] - b.coeffs[i + 1, j - 1]).sum()
    assert d[i, j] == pytest.approx(expected, abs=1e-5)


def test_cell_mask_from_pixel_mask():
    m = np.zeros((24, 32), bool)
    m[8:16, 8:16] = True
    cells = pbp.cell_mask_from_pixel_mask(m, 8, 4, frac=0.25)
    assert cells.shape == pbp.grid_shape(24, 32, 8, 4)
    assert cells[2, 2] and not cells[0, 0] and cells[1, 1]   # 4x4 of 64 px = 0.25 -> True


def test_serialize_deterministic():
    rng = np.random.default_rng(11)
    img = rng.uniform(0, 255, (40, 56)).astype(np.float32)
    s1 = pbp.serialize_signatures(pbp.window_signatures(img, P))
    s2 = pbp.serialize_signatures(pbp.window_signatures(img.copy(), P))
    assert s1 == s2 and s1.startswith(b"DALILPBP")
    img[10, 10] += 50
    assert pbp.serialize_signatures(pbp.window_signatures(img, P)) != s1


# --------------------------------------------------------------------------- #
# Speed (reported, loosely asserted)
# --------------------------------------------------------------------------- #

def test_timing_1280x720(capsys):
    rng = np.random.default_rng(12)
    small = rng.uniform(0, 255, (90, 160)).astype(np.float32)
    img = cv2.resize(small, (1280, 720), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img + rng.normal(0, 4, img.shape).astype(np.float32), (0, 0), 1.0)
    pbp.window_signatures(img, P)                              # warm-up
    sig_ms, dist_ms = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        sig = pbp.window_signatures(img, P)
        t1 = time.perf_counter()
        pbp.pair_distance(sig, sig, 1, 1)
        t2 = time.perf_counter()
        sig_ms.append((t1 - t0) * 1e3)
        dist_ms.append((t2 - t1) * 1e3)
    with capsys.disabled():
        print(f"\n[pbp timing] 1280x720 window_signatures best {min(sig_ms):.0f} ms "
              f"(runs {', '.join(f'{t:.0f}' for t in sig_ms)}), pair_distance best "
              f"{min(dist_ms):.0f} ms; grid {sig.coeffs.shape[:2]}; CPU {platform.processor()}, "
              f"{os.cpu_count()} logical cores")
    assert min(sig_ms) < 2000
