"""Reproducible benchmark of AVA Mode 1 (single-photo surface check) on synthetic uniform surfaces.

    python -m eval.benchmark_surface --n 4 --seed 0          # calibrate, test, write eval/RESULTS_SURFACE.md
    python -m eval.benchmark_surface --split calib --n 4      # calibration only (surface_calib.json)
    python -m eval.benchmark_surface --split test --n 4       # test with the frozen surface_calib.json
    python -m eval.benchmark_surface --split report           # rebuild the report from the saved JSON

Protocol (also written into RESULTS_SURFACE.md):
1. Grid: surface kind (synth.SURFACE_KINDS) x lighting (none + synth.FACTORY_LIGHTING) x defect
   (none + DEFECTS). `--n` photos per defect cell and `--n * --clean-mult` defect-free photos per
   (kind, lighting), so the false-alarm rate rests on more photos. Every photo has its own seed,
   SeedSequence(--seed, split code, kind, lighting, defect, repeat); CALIB and TEST use different
   split codes, so their random streams are disjoint.
2. Each photo: `synth.make_uniform_surface` -> defect (if any) -> lighting (if any) ->
   `surface.scan_surface` once per method (pbp, census, gradient). The smoothed score map it
   returns is re-thresholded with `zones.extract_zones` exactly as scan_surface does, over a
   threshold grid.
3. CALIB: per method, the smallest grid threshold t such that the false-alarm rate (defect-free
   photos answered FAIL) under none + monotone lighting is <= FA_TARGET at t and at every larger
   grid threshold. Frozen.
4. TEST: every metric at the frozen thresholds.
5. Timing: a separate sequential pass in the main process (OpenCV / BLAS threads on), because
   times measured in the parallel workers include CPU contention.

A defect photo counts as detected if the verdict is FAIL and some zone bbox has IoU > 0.1 with
the true defect bbox or covers >= 30 % of it (bboxes at the working resolution).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python eval/benchmark_surface.py`
    sys.path.insert(0, str(ROOT))

from dalil import surface, synth  # noqa: E402
from dalil.config import Params  # noqa: E402
from dalil.pbp import cell_mask_from_pixel_mask  # noqa: E402
from dalil.zones import extract_zones  # noqa: E402

METHODS = tuple(surface.SURFACE_METHODS)          # pbp, census, gradient
DEFECT_FNS = {
    "scratch": synth.add_scratch,
    "chip": synth.add_chip,
    "stain": synth.add_stain,
    "dent": synth.add_dent,
    "burr": synth.add_burr,
    "sticker": synth.add_sticker,
}
BENCH_DEFECTS = tuple(DEFECT_FNS)
DEFECTS = (None,) + BENCH_DEFECTS
LIGHTINGS = (None,) + tuple(synth.FACTORY_LIGHTING)
CLEAN_LIGHTINGS = ("none",) + tuple(synth.MONOTONE_LIGHTING)   # the calibration domain
SPLIT_CODES = {"calib": 101, "test": 102}   # also distinct from eval/benchmark.py's codes
FA_TARGET = 0.05
IOU_MIN = 0.1
COVER_MIN = 0.3
UNCHECKED_MISS = 0.5     # a missed defect with < this fraction of its cells checked: "masked"
PREFIX = "surface_"

# Sweep outcome codes (column 0 of a sweep row).
PASS, FAIL, RETAKE = 0, 1, 2
_CODES = {"pass": PASS, "fail": FAIL, "retake": RETAKE}


# --------------------------------------------------------------------------- small helpers

def threshold_grid(extra=()) -> np.ndarray:
    """Geometric grid 0.02..2.5 (120 steps, ~4 % apart) plus `extra` values.

    PBP / census scores are clipped to [0, 1]; the gradient baseline (robust z / 10) is not.
    """
    g = np.geomspace(0.02, 2.5, 120)
    return np.unique(np.round(np.concatenate([g, np.asarray(list(extra), float)]), 5))


def method_param(method: str) -> str:
    return "surface_threshold" if method == "pbp" else f"threshold_surface_{method}"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95 % interval of a binomial rate k/n ((nan, nan) for n = 0)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def auroc(pos: list, neg: list) -> float:
    """Mann-Whitney AUROC (ties count 1/2); nan if a class is empty."""
    if not pos or not neg:
        return float("nan")
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    sv = allv[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


class Truth:
    """Ground-truth defect mask at the working resolution, with the zone matching rules."""

    def __init__(self, mask_work: np.ndarray, near_px: int):
        self.mask = mask_work
        self.any = bool(mask_work.any())
        if self.any:
            ys, xs = np.nonzero(mask_work)
            self.box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            k = 2 * int(near_px) + 1
            self.near = cv2.dilate(mask_work.astype(np.uint8), np.ones((k, k), np.uint8)) > 0

    def match(self, zones) -> tuple[bool, int]:
        """(some zone matches the defect, number of zones not touching the defect)."""
        if not self.any:
            return False, len(zones)
        x0, y0, x1, y1 = self.box
        area_t = (x1 - x0) * (y1 - y0)
        hit, n_false = False, 0
        for z in zones:
            zx, zy, zw, zh = z.bbox
            iw = max(0, min(x1, zx + zw) - max(x0, zx))
            ih = max(0, min(y1, zy + zh) - max(y0, zy))
            inter = iw * ih
            union = area_t + zw * zh - inter
            if inter / max(1, union) > IOU_MIN or inter / max(1, area_t) >= COVER_MIN:
                hit = True
            if not self.near[zy:zy + zh, zx:zx + zw].any():
                n_false += 1
        return hit, n_false


def mask_to_work(mask: np.ndarray, work_size: tuple[int, int]) -> np.ndarray:
    w, h = work_size
    if mask.shape[:2] == (h, w):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0


def zones_at(res, params: Params, t: float) -> list:
    """Zones of a scored result at threshold t, exactly as scan_surface extracts them."""
    return extract_zones(res.score, res.checked, dataclasses.replace(params, threshold=float(t)),
                         win=params.win, step=params.step)


def sweep_rows(res, truth: Truth, params: Params, grid: np.ndarray) -> np.ndarray:
    """int16[len(grid), 4] = (code, n_zones, n_false, detected) per threshold."""
    rows = np.zeros((len(grid), 4), np.int16)
    if res.score is None:  # rejected before scoring (glare / blur / dark / coverage)
        rows[:, 0] = RETAKE
        return rows
    top = float(res.score[res.checked].max()) if res.checked.any() else -1.0
    for i, t in enumerate(grid):
        if t > top:  # no cell reaches t: no zones, PASS (rows stay 0)
            break
        zones = zones_at(res, params, float(t))
        hit, n_false = truth.match(zones)
        rows[i] = (FAIL if zones else PASS, len(zones), n_false, int(hit and bool(zones)))
    return rows


def defect_checked_frac(res, truth: Truth, params: Params) -> float | None:
    """Fraction of the defect's cells (windows touching the mask) that were checked."""
    if not truth.any or res.checked is None:
        return None
    cells = cell_mask_from_pixel_mask(truth.mask, params.win, params.step, frac=1.0 / (params.win ** 2))
    if not cells.any():
        return None
    return round(float((cells & res.checked).sum()) / float(cells.sum()), 4)


def max_score(res) -> float | None:
    if res.score is None or not res.checked.any():
        return None
    return float(res.score[res.checked].max())


# --------------------------------------------------------------------------- synthetic grid

@dataclass(frozen=True)
class Case:
    kind: str
    lighting: str | None
    defect: str | None
    rep: int
    entropy: tuple


def build_cases(split: str, seed: int, n: int, clean_mult: int, kinds, lightings, defects) -> list[Case]:
    """All grid cells x repeats; seeds depend only on the full-grid position (filter-stable)."""
    cases = []
    for kd in kinds:
        for li in lightings:
            for d in defects:
                reps = n * max(1, int(clean_mult)) if d is None else n
                for r in range(reps):
                    ent = (int(seed), SPLIT_CODES[split], synth.SURFACE_KINDS.index(kd),
                           LIGHTINGS.index(li), DEFECTS.index(d), r)
                    cases.append(Case(kd, li, d, r, ent))
    return cases


def make_photo(case: Case, size) -> tuple[np.ndarray, np.ndarray]:
    """(BGR photo, defect mask at the generated size): surface -> defect -> lighting."""
    h, w = size
    rng = np.random.default_rng(np.random.SeedSequence(list(case.entropy)))
    img = synth.make_uniform_surface(rng, case.kind, h=h, w=w)
    mask = np.zeros((h, w), bool)
    if case.defect is not None:
        img, mask = DEFECT_FNS[case.defect](img, rng)
    if case.lighting is not None:
        img = synth.apply_factory_lighting(img, rng, case.lighting)
    return img, mask


def _init_worker():
    cv2.setNumThreads(1)          # one process per core: no nested OpenCV threads ...
    surface._MAX_THREADS = 1      # ... and no kNN chunk thread pool


def run_case(task) -> dict:
    """Worker entry: `_run_case`, or an error record if it raised (reported, never hidden)."""
    try:
        return _run_case(task)
    except Exception as e:  # noqa: BLE001 - one bad photo must not abort the run
        case = task[0]
        return {"error": f"{type(e).__name__}: {e}"[:500], "kind": case.kind,
                "lighting": case.lighting or "none", "defect": case.defect or "none",
                "rep": case.rep, "entropy": list(case.entropy)}


def _run_case(task) -> dict:
    case, size, methods, params_dict, grids = task
    P = Params(**params_dict)
    img, mask = make_photo(case, size)
    rec = {"kind": case.kind, "lighting": case.lighting or "none", "defect": case.defect or "none",
           "rep": case.rep, "entropy": list(case.entropy), "mask_px_generated": int(mask.sum()),
           "methods": {}}
    for m in methods:
        res = surface.scan_surface(img, P, m)
        truth = Truth(mask_to_work(mask, res.work_size), P.win)
        grid = grids[m]
        rows = sweep_rows(res, truth, P, grid)
        hit, n_false = truth.match(res.zones)
        i_op = int(np.searchsorted(grid, res.threshold))
        consistent = (i_op < len(grid) and abs(grid[i_op] - res.threshold) < 1e-9
                      and rows[i_op, 0] == _CODES[res.verdict] and rows[i_op, 1] == len(res.zones))
        rec["methods"][m] = {
            "verdict": res.verdict, "reason": res.reason, "n_zones": len(res.zones),
            "n_false": n_false, "detected": bool(hit and res.verdict == "fail"),
            "zones": [list(z.bbox) for z in res.zones], "threshold": res.threshold,
            "coverage": round(res.coverage, 4), "max_score": max_score(res),
            "defect_checked_frac": defect_checked_frac(res, truth, P),
            "ms_parallel": res.timing_ms.get("total"), "consistent": bool(consistent), "sweep": rows,
        }
    return rec


_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def run_grid(cases: list, size, methods, params: Params, grids: dict, workers: int,
             label: str) -> tuple[list, list]:
    """Run every case; returns (records, error records), both sorted by seed entropy."""
    tasks = [(c, size, tuple(methods), params.to_dict(), grids) for c in cases]
    recs = []
    t0 = time.perf_counter()
    step = max(1, len(tasks) // 20)

    def progress(i):
        if (i + 1) % step == 0 or i + 1 == len(tasks):
            print(f"[{label}] {i + 1}/{len(tasks)} photos, {time.perf_counter() - t0:.0f} s", flush=True)

    if workers <= 1:
        for i, t in enumerate(tasks):
            recs.append(run_case(t))
            progress(i)
    else:
        saved = {k: os.environ.get(k) for k in _THREAD_ENV}
        os.environ.update({k: "1" for k in _THREAD_ENV})  # inherited by the spawned workers
        try:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers, initializer=_init_worker, maxtasksperchild=60) as pool:
                for i, r in enumerate(pool.imap_unordered(run_case, tasks, chunksize=2)):
                    recs.append(r)
                    progress(i)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    recs.sort(key=lambda r: r["entropy"])
    errors = [r for r in recs if "error" in r]
    for e in errors:
        print(f"[{label}] ERROR in photo {e['entropy']}: {e['error']}", flush=True)
    return [r for r in recs if "error" not in r], errors


# --------------------------------------------------------------------------- aggregation

def lighting_group(name: str) -> str:
    if name == "none":
        return "none"
    return "monotone" if name in synth.MONOTONE_LIGHTING else "non_monotone"


def summarize(recs: list, method: str, ti: int | None = None) -> dict:
    """Metrics of `method` over `recs` at sweep index `ti` (None = the stored operating point)."""
    n_def = n_good = det = esc = fa = rt_all = rt_good = rt_def = fz_all = fz_good = 0
    miss_masked = miss_scored = 0
    cov, ms, reasons = [], [], {}
    for r in recs:
        mm = r["methods"][method]
        if ti is None:
            code, nz, nf, hit = _CODES[mm["verdict"]], mm["n_zones"], mm["n_false"], mm["detected"]
        else:
            code, nz, nf, hit = (int(v) for v in mm["sweep"][ti])
        retake = code == RETAKE
        rt_all += retake
        fz_all += nf
        if retake:
            reasons[mm["reason"]] = reasons.get(mm["reason"], 0) + 1
        if mm["max_score"] is not None:
            cov.append(mm["coverage"])
        if mm.get("ms_parallel") is not None:
            ms.append(mm["ms_parallel"])
        if r["defect"] == "none":
            n_good += 1
            fa += code == FAIL
            rt_good += retake
            fz_good += nf
        else:
            n_def += 1
            det += bool(hit)
            esc += code == PASS
            rt_def += retake
            if not hit and not retake:
                cf = mm.get("defect_checked_frac")
                if cf is not None and cf < UNCHECKED_MISS:
                    miss_masked += 1   # the defect lies mostly in the unchecked (glare / border) area
                else:
                    miss_scored += 1   # checked, but no zone reached the threshold / min size
    nan = float("nan")
    return {
        "n_photos": len(recs), "n_defect": n_def, "n_good": n_good,
        "recall": det / n_def if n_def else nan, "recall_ci": wilson(det, n_def), "detected": det,
        "escape_rate": esc / n_def if n_def else nan, "escape_ci": wilson(esc, n_def),
        "retake_rate_defect": rt_def / n_def if n_def else nan,
        "miss_retake": rt_def, "miss_masked": miss_masked, "miss_scored": miss_scored,
        "fa_rate": fa / n_good if n_good else nan, "fa_ci": wilson(fa, n_good), "false_alarms": fa,
        "false_zones_per_good": fz_good / n_good if n_good else nan,
        "false_zones_per_photo": fz_all / len(recs) if recs else nan,
        "retake_rate": rt_all / len(recs) if recs else nan, "retake_ci": wilson(rt_all, len(recs)),
        "retake_rate_good": rt_good / n_good if n_good else nan,
        "coverage": float(np.mean(cov)) if cov else nan,
        "mean_ms_parallel": float(np.mean(ms)) if ms else nan,
        "retake_reasons": reasons,
    }


def separability(recs: list, method: str) -> dict:
    """Threshold-free AUROC of the max smoothed cell score (defect vs defect-free photos)."""
    pos = [r["methods"][method]["max_score"] for r in recs if r["defect"] != "none"]
    neg = [r["methods"][method]["max_score"] for r in recs if r["defect"] == "none"]
    excl = sum(v is None for v in pos + neg)
    pos = [v for v in pos if v is not None]
    neg = [v for v in neg if v is not None]
    return {"auroc": auroc(pos, neg), "n_pos": len(pos), "n_neg": len(neg), "n_unscored": excl}


def subsets(recs: list) -> dict:
    out = {"all": recs,
           "none+monotone": [r for r in recs if r["lighting"] in CLEAN_LIGHTINGS],
           "none": [r for r in recs if r["lighting"] == "none"],
           "monotone": [r for r in recs if lighting_group(r["lighting"]) == "monotone"],
           "non_monotone": [r for r in recs if lighting_group(r["lighting"]) == "non_monotone"]}
    for k in synth.FACTORY_LIGHTING:
        out[f"lighting:{k}"] = [r for r in recs if r["lighting"] == k]
    for k in synth.SURFACE_KINDS:
        out[f"kind:{k}"] = [r for r in recs if r["kind"] == k]
    return {k: v for k, v in out.items() if v}


SWEEP_GROUPS = ("all", "none+monotone", "none", "monotone", "non_monotone")


def sweep_curves(recs: list, methods, grids) -> dict:
    """Per method and lighting group: recall / FA rate / false zones / retake rate per threshold."""
    subs = subsets(recs)
    out = {}
    for m in methods:
        out[m] = {"thresholds": [float(t) for t in grids[m]]}
        for g in SWEEP_GROUPS:
            sub = subs.get(g)
            if not sub:
                continue
            arr = np.stack([r["methods"][m]["sweep"] for r in sub])          # [n, T, 4]
            good = np.array([r["defect"] == "none" for r in sub])
            code, nf, hit = arr[..., 0], arr[..., 2], arr[..., 3]
            ng, nd = int(good.sum()), int((~good).sum())
            out[m][g] = {
                "n_good": ng, "n_defect": nd,
                "recall": (hit[~good].sum(0) / nd).round(4).tolist() if nd else None,
                "fa_rate": ((code[good] == FAIL).sum(0) / ng).round(4).tolist() if ng else None,
                "false_zones_per_photo": (nf.sum(0) / len(sub)).round(4).tolist(),
                "retake_rate": ((code == RETAKE).sum(0) / len(sub)).round(4).tolist(),
            }
    return out


def calibrate(recs: list, methods, grids, target: float) -> dict:
    """Per method: smallest grid threshold t with FA rate <= target at t and every larger t."""
    good = [r for r in recs if r["defect"] == "none" and r["lighting"] in CLEAN_LIGHTINGS]
    if not good:
        raise SystemExit("calibration needs defect-free photos under none/monotone lighting")
    defect = [r for r in recs if r["defect"] != "none" and r["lighting"] in CLEAN_LIGHTINGS]
    out = {}
    for m in methods:
        arr = np.stack([r["methods"][m]["sweep"] for r in good])
        fa = (arr[..., 0] == FAIL).mean(0)
        ok_from = np.flip(np.logical_and.accumulate(np.flip(fa <= target)))  # ok here and above
        reached = bool(ok_from.any())
        i = int(np.argmax(ok_from)) if reached else len(grids[m]) - 1
        rec = float("nan")
        if defect:
            darr = np.stack([r["methods"][m]["sweep"] for r in defect])
            rec = float(darr[:, i, 3].mean())
        out[m] = {"threshold": float(grids[m][i]), "reached": reached, "fa_rate": float(fa[i]),
                  "false_alarms": int((arr[:, i, 0] == FAIL).sum()), "n_good_photos": len(good),
                  "recall_calib": rec, "n_defect_photos": len(defect)}
    return out


# --------------------------------------------------------------------------- timing

def timing_pass(cases: list, size, methods, params: Params, k: int) -> dict:
    """Sequential scans in this process (OpenCV threads and the kNN thread pool enabled)."""
    if k <= 0 or not cases:
        return {}
    cv2.setNumThreads(-1)
    idx = np.linspace(0, len(cases) - 1, k + 1).round().astype(int)
    picked = [cases[i] for i in dict.fromkeys(idx.tolist())]
    per = {m: [] for m in methods}
    stages: dict = {}
    for j, c in enumerate(picked):
        img, _ = make_photo(c, size)
        for m in methods:
            res = surface.scan_surface(img, params, m)
            if j == 0:
                continue  # warm-up
            per[m].append(res.timing_ms["total"])
            if m == "pbp" and res.score is not None:
                for st, v in res.timing_ms.items():
                    stages.setdefault(st, []).append(v)
    out = {m: {"n": len(v), "mean_ms": float(np.mean(v)), "median_ms": float(np.median(v)),
               "min_ms": float(np.min(v)), "max_ms": float(np.max(v))} for m, v in per.items() if v}
    out["_pbp_stage_median_ms"] = {st: float(np.median(v)) for st, v in stages.items()}
    return out


# --------------------------------------------------------------------------- environment

def cpu_name() -> str:
    try:
        if sys.platform == "win32":
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        elif Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def git_state() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "dalil", "eval"],
                               cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        return (rev or "unknown") + (" + uncommitted changes in dalil/eval" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment(workers: int) -> dict:
    return {"cpu": cpu_name(), "logical_cores": os.cpu_count(), "os": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
            "git": git_state(), "workers": workers}


# --------------------------------------------------------------------------- report

def _na(x) -> bool:
    return x is None or x != x


def _pct(x) -> str:
    return "n/a" if _na(x) else f"{100 * x:.1f}%"


def _rate(x, ci) -> str:
    if _na(x) or ci is None or _na(ci[0]):
        return "n/a"
    return f"{100 * x:.1f}% ({100 * ci[0]:.1f}–{100 * ci[1]:.1f})"


def _f(x, nd: int = 3) -> str:
    return "n/a" if _na(x) else f"{x:.{nd}f}"


def _table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


TITLE = "# Synthetic benchmark — procedurally generated uniform surfaces, NOT real photos"
_NAMES = {"pbp": "PBP (AVA)", "census": "Census kNN", "gradient": "Gradient z-score"}


def write_report(path: Path, calib: dict, test: dict) -> None:
    methods = test["methods"]
    S = test["summary"]
    env = test["env"]
    T = test.get("timing", {})
    n_test = S["all"][methods[0]]["n_photos"]
    g = test["grid"]
    L = [TITLE + "\n"]
    L.append("Every number in this file was produced by `eval/benchmark_surface.py` (raw data in "
             "`eval/results/surface_*.json`). The photos come from `dalil/synth.py`: procedural close-ups of "
             "painted, plastic, brushed-metal and dark glossy surfaces that fill the whole frame, procedural "
             "defects and procedural shop-floor lighting. They compare the methods under controlled "
             "conditions; they are **not** evidence of accuracy on real parts.\n")
    L.append("## Run\n")
    cmds = [calib["command"]] if calib["command"] == test["command"] else [calib["command"], test["command"]]
    L.append("```\n" + "\n".join(cmds) + "\n```\n")
    L.append(_table(["", ""], [
        ["Date (calib / test)", f"{calib['date']} / {test['date']}"],
        ["CPU", f"{env['cpu']} ({env['logical_cores']} logical cores)"],
        ["Software", f"{env['os']}; Python {env['python']}, numpy {env['numpy']}, OpenCV {env['opencv']}"],
        ["Code", env["git"]],
        ["Grid", f"{len(g['kinds'])} surface kinds x {len(g['lightings'])} lightings x {len(g['defects'])} "
                 f"defect classes; {test['n']} photo(s) per defect cell, {test['n'] * test['clean_mult']} "
                 f"defect-free photo(s) per (kind, lighting) = {n_test} TEST photos ({calib['n_photos']} CALIB)"],
        ["Surface kinds", ", ".join(g["kinds"])],
        ["Lightings", ", ".join(g["lightings"])],
        ["Defects", ", ".join(g["defects"])],
        ["Image size", f"{test['size'][1]}x{test['size'][0]} px generated, working size max side "
                       f"{test['params']['work_max_side']} px"],
        ["Seeds", f"--seed {test['seed']}; per photo SeedSequence(seed, split code calib="
                  f"{SPLIT_CODES['calib']}/test={SPLIT_CODES['test']}, kind, lighting, defect, repeat)"],
        ["Wall time", f"calib {calib['wall_s']:.0f} s, test {test['wall_s']:.0f} s with "
                      + (f"{env['workers']} worker processes (1 thread each)" if env["workers"] > 1
                         else "1 process")],
        ["Photos that raised an exception", f"calib {len(calib.get('errors', []))}, test "
                                            f"{len(test.get('errors', []))} (excluded; listed in the JSON)"],
        ["Sweep vs scan_surface consistency", f"{calib.get('consistency_failures', 0)} + "
                                              f"{test.get('consistency_failures', 0)} mismatches at the operating "
                                              "point"],
    ]) + "\n")

    L.append("## Protocol\n")
    L.append(f"""1. **Photos.** `synth.make_uniform_surface` renders a surface filling the whole frame (with a soft
   illumination fall-off and sensor noise); a defect is drawn on it ({', '.join(BENCH_DEFECTS)}) or not; then
   one shop-floor lighting change is applied to the whole photo, or none. Monotone: {', '.join(synth.MONOTONE_LIGHTING)};
   non-monotone: {', '.join(synth.NON_MONOTONE_LIGHTING)}.
2. **Same pipeline for every method.** `surface.scan_surface(photo, params, method)`: identical capture
   gates (dark / blurry / glare / coverage), glare mask, smoothing and zone extraction; only the per-window
   anomaly score differs — PBP self-reference kNN + contrast term (AVA), the same kNN on Census codes, or a
   robust z-score of the Sobel gradient magnitude.
3. **Calibration (CALIB split only).** Each method's smoothed score map is re-thresholded over a geometric
   grid (0.02–2.5). Per method we take the smallest grid threshold at which the **false-alarm rate** (defect-free
   photos answered FAIL) under **none + monotone** lighting is ≤ {int(FA_TARGET * 100)} %, at that threshold and every
   larger one. The thresholds are frozen and written to `dalil/config.py`.
4. **Test (TEST split, disjoint seeds).** *Recall*: a defect photo is detected if the verdict is FAIL and
   some zone has IoU > {IOU_MIN} with the true defect bbox or covers ≥ {int(COVER_MIN * 100)} % of it (RETAKE counts as
   a miss). *False-alarm rate*: defect-free photos answered FAIL. *False zones / photo*: zones that do not
   touch the defect (dilated by one window) — on a defect-free photo every zone — averaged over all photos.
   *Retake rate*: photos answered RETAKE (glare, blurry, dark, low coverage), all photos. Brackets: Wilson
   95 % intervals.
5. **Timing.** A separate sequential pass in one process (OpenCV / BLAS threads on),
   {T.get('pbp', {}).get('n', 0)} TEST photos after one warm-up photo; times include resize, capture checks, glare
   mask, scoring, zones and the overlay JPEG. The "ms (parallel)" columns are measured inside the
   {env['workers']} benchmark workers (1 thread each, CPU shared) and are only comparable with each other.
""")

    L.append("## Calibrated thresholds (frozen before TEST)\n")
    rows = []
    for m in methods:
        c = calib["calibration"][m]
        rows.append([_NAMES.get(m, m), f"`{method_param(m)}`",
                     f"{c['threshold']:.4g}" + ("" if c["reached"] else " (target not reached)"),
                     f"{_pct(c['fa_rate'])} ({c['false_alarms']}/{c['n_good_photos']})",
                     _pct(c["recall_calib"])])
    L.append(_table(["Method", "Param", "Threshold", "CALIB false-alarm rate", "CALIB recall (none+monotone)"],
                    rows) + "\n")

    def headline(key: str, title: str) -> None:
        if key not in S:
            return
        rows = []
        for m in methods:
            s = S[key][m]
            rows.append([_NAMES.get(m, m), _rate(s["recall"], s["recall_ci"]), _rate(s["fa_rate"], s["fa_ci"]),
                         _f(s["false_zones_per_photo"]), _rate(s["retake_rate"], s["retake_ci"]),
                         _pct(s["coverage"]), _f(T.get(m, {}).get("mean_ms", float("nan")), 0),
                         _f(s["mean_ms_parallel"], 0)])
        n = S[key][methods[0]]
        L.append(f"### {title} — {n['n_defect']} defect photos, {n['n_good']} defect-free photos\n")
        L.append(_table(["Method", "Defect recall", "False-alarm rate (good → FAIL)", "False zones / photo",
                         "Retake rate", "Coverage", "Mean ms (sequential)", "Mean ms (parallel)"], rows) + "\n")

    L.append("## TEST results\n")
    headline("none+monotone", "Lighting: none + monotone (the operating domain)")
    headline("none", "Lighting: none")
    headline("monotone", "Lighting: monotone changes only")
    headline("non_monotone", "Lighting: non-monotone (specular highlight, hand shadow) — expected failure")
    headline("all", "All lighting conditions")

    L.append("### By lighting kind — recall / false-alarm rate / false zones per photo / retake rate\n")
    keys = ["none"] + [f"lighting:{k}" for k in synth.MONOTONE_LIGHTING] + \
           [f"lighting:{k}" for k in synth.NON_MONOTONE_LIGHTING]
    rows = []
    for k in keys:
        if k not in S:
            continue
        name = k.split(":", 1)[-1]
        grp = "—" if name == "none" else lighting_group(name).replace("_", "-")
        rows.append([name, grp] + [f"{_rate(S[k][m]['recall'], S[k][m]['recall_ci'])} / "
                                   f"{_rate(S[k][m]['fa_rate'], S[k][m]['fa_ci'])} / "
                                   f"{_f(S[k][m]['false_zones_per_photo'], 2)} / {_pct(S[k][m]['retake_rate'])}"
                                   for m in methods])
    L.append(_table(["Lighting", "Group"] + [_NAMES.get(m, m) for m in methods], rows) + "\n")

    L.append("### Per-defect recall (none + monotone lighting)\n")
    rows = []
    for d in BENCH_DEFECTS:
        sub = test["per_defect"].get(d)
        if sub:
            rows.append([d, sub[methods[0]]["n_defect"]] +
                        [_rate(sub[m]["recall"], sub[m]["recall_ci"]) for m in methods])
    L.append(_table(["Defect", "Photos"] + [_NAMES.get(m, m) for m in methods], rows) + "\n")

    L.append("### Per surface kind (none + monotone lighting) — recall / false-alarm rate\n")
    rows = []
    for kd in synth.SURFACE_KINDS:
        sub = test["clean_by_kind"].get(kd)
        if sub:
            rows.append([kd] + [f"{_rate(sub[m]['recall'], sub[m]['recall_ci'])} / "
                                f"{_rate(sub[m]['fa_rate'], sub[m]['fa_ci'])}" for m in methods])
    L.append(_table(["Surface"] + [_NAMES.get(m, m) for m in methods], rows) + "\n")

    if "pbp" in methods and test.get("kind_defect_pbp"):
        L.append("### PBP (AVA): detected / photos per surface kind x defect (none + monotone lighting)\n")
        kd_tab = test["kind_defect_pbp"]
        rows = [[kd] + [kd_tab[kd].get(d, "—") for d in BENCH_DEFECTS] for kd in synth.SURFACE_KINDS if kd in kd_tab]
        L.append(_table(["Surface"] + list(BENCH_DEFECTS), rows) + "\n")

    L.append("### Why defects were missed (TEST)\n")
    rows = []
    for key in ("none+monotone", "non_monotone"):
        if key not in S:
            continue
        for m in methods:
            s = S[key][m]
            rows.append([key, _NAMES.get(m, m), s["n_defect"] - s["detected"], s["miss_retake"],
                         s["miss_masked"], s["miss_scored"]])
    L.append(_table(["Lighting", "Method", "Missed", "RETAKE", f"Defect < {int(UNCHECKED_MISS * 100)} % checked "
                     "(glare / border mask)", "Checked, no zone at threshold"], rows) + "\n")
    ag = test.get("agreement")
    if ag and len(methods) > 1:
        L.append(f"On {ag['same']} of {ag['n_defect']} TEST defect photos all {len(methods)} methods give the same "
                 f"detected / missed outcome; on {ag['same_good']} of {ag['n_good']} defect-free photos they give "
                 "the same verdict. The capture gates, the glare mask and the zone rules are shared, and the "
                 "synthetic defects are high-contrast, so on this set the methods differ mainly in false alarms "
                 "and speed, not in recall.\n")

    L.append("### Retake reasons (all TEST photos)\n")
    rows = [[_NAMES.get(m, m), ", ".join(f"{k}: {v}" for k, v in sorted(S["all"][m]["retake_reasons"].items(),
                                                                      key=lambda kv: -kv[1])) or "—"]
            for m in methods]
    L.append(_table(["Method", "Reasons"], rows) + "\n")

    L.append("### Threshold-free separability (AUROC of the max smoothed cell score, defect vs defect-free)\n")
    rows = [[_NAMES.get(m, m)] + [_f(test["auroc"][grp][m]["auroc"]) for grp in ("none+monotone", "non_monotone", "all")]
            for m in methods]
    L.append(_table(["Method", "none + monotone", "non-monotone", "all"], rows))
    L.append("\nPhotos rejected before scoring (glare / blur / dark / coverage) have no score and are left out of "
             "the AUROC. Recall / false-alarm / false-zone / retake curves over the whole threshold grid, per "
             "lighting group, are in `eval/results/surface_test_sweep.json` (and `surface_calib_sweep.json`).\n")

    L.append("## Speed (sequential pass, this machine)\n")
    rows = [[_NAMES.get(m, m), _f(T[m]["mean_ms"], 0), _f(T[m]["median_ms"], 0),
             f"{T[m]['min_ms']:.0f}–{T[m]['max_ms']:.0f}", T[m]["n"]] for m in methods if m in T]
    L.append(_table(["Method", "Mean ms", "Median ms", "Min–max ms", "Photos"], rows) + "\n")
    st = T.get("_pbp_stage_median_ms", {})
    if st:
        L.append("PBP stage medians (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in st.items()) + ".\n")
    L.append(f"CPU: {env['cpu']}. Photos {test['size'][1]}x{test['size'][0]} resized to max side "
             f"{test['params']['work_max_side']} px. Other processes on the machine were not controlled; "
             "treat the times as indicative. The deployed server CPU will differ.\n")

    L.append("## Caveats (read before quoting any number)\n")
    L.append(f"""- **Synthetic data only.** Surfaces, defects and lighting are procedural (`dalil/synth.py`). Real
  scratches, chips, stains and dents look different, and real phone photos add compression, defocus, motion blur
  and perspective that are only partly modelled. The thresholds were calibrated on the same generator, so
  these numbers are an upper-bound-style comparison of methods, not a real-world accuracy claim.
- **The photo must contain only the surface.** Mode 1 treats anything rare in the photo as a defect: part
  edges, holes, logos, labels, the background or a hand in the frame become zones. Complex parts belong to
  Mode 2 (compare with a reference photo). Every synthetic photo here fills the whole frame with the surface,
  which is the favourable case.
- **Non-monotone lighting is an expected failure.** A specular highlight is masked (with its halo) and may lead
  to RETAKE (glare); a hard hand shadow is a real, rare edge on the surface and is flagged like a defect. Both
  are reported separately above and are not part of the operating domain.
- **Random micro-textures.** On plastic grain and brushed metal ordinary windows have no look-alikes either, so
  the PBP kNN term is gated off there and the (gain-invariant) contrast term does most of the detection; the
  gradient baseline is a strong competitor on this synthetic set. PBP and Census are close relatives
  (rank/order descriptors); a small gap between them is not evidence at this N.
- **Small N.** {test['n']} photo(s) per defect cell and {test['n'] * test['clean_mult']} defect-free photo(s) per
  (kind, lighting). Check the Wilson 95 % intervals; differences inside overlapping intervals are not evidence.
- **One calibration criterion for all methods** (false-alarm rate ≤ {int(FA_TARGET * 100)} % on defect-free CALIB photos
  under none + monotone lighting). A different operating point changes the trade-off; see the sweeps.
- **The ground truth is a bbox.** Zones live on a {test['params']['step']} px window grid; recall uses the bbox IoU /
  coverage rule above, false zones use "does not touch the (dilated) defect mask".
""")
    L.append("## Real data\n")
    L.append("No real-photo evaluation of Mode 1 is included in this file. No real-data metric is claimed here.\n")
    path.write_text("\n".join(L), encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- main

def _clean(o):
    """JSON-safe copy: numpy -> python, NaN -> None, tuples -> lists."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if f != f else f
    return o


def _dump(path: Path, obj, compact: bool = False) -> None:
    text = json.dumps(_clean(obj), ensure_ascii=False,
                      **({"separators": (",", ":")} if compact else {"indent": 1}))
    path.write_text(text, encoding="utf-8", newline="\n")


def _photo_rows(recs: list) -> list:
    out = []
    for r in recs:
        r2 = {k: v for k, v in r.items() if k != "methods"}
        r2["methods"] = {m: {k: v for k, v in mm.items() if k != "sweep"} for m, mm in r["methods"].items()}
        out.append(r2)
    return out


def _parse_list(s: str | None, allowed, none_word="none"):
    if s is None:
        return list(allowed)
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        val = None if tok == none_word else tok
        if val not in allowed:
            raise SystemExit(f"unknown value {tok!r}; allowed: {[a or none_word for a in allowed]}")
        out.append(val)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=4, help="photos per defect cell (default 4)")
    ap.add_argument("--clean-mult", type=int, default=3,
                    help="defect-free photos per (kind, lighting) = n x this (default 3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=("calib", "test", "all", "report"), default="all",
                    help="'report' only rebuilds the report from the saved JSON in --out")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--report", default=str(ROOT / "eval" / "RESULTS_SURFACE.md"))
    ap.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--size", default="720x1280", help="generated image HxW (default 720x1280)")
    ap.add_argument("--kinds", default=None, help="subset of surface kinds (comma list)")
    ap.add_argument("--lightings", default=None, help="subset, e.g. none,gamma,specular")
    ap.add_argument("--defects", default=None, help="subset, e.g. none,chip")
    ap.add_argument("--timing-photos", type=int, default=12, help="photos in the sequential timing pass")
    args = ap.parse_args(argv)

    methods = _parse_list(args.methods, METHODS)
    size = tuple(int(v) for v in args.size.lower().split("x"))
    grid_def = {"kinds": _parse_list(args.kinds, synth.SURFACE_KINDS),
                "lightings": _parse_list(args.lightings, LIGHTINGS),
                "defects": _parse_list(args.defects, DEFECTS)}
    grid_json = {k: [v or "none" for v in vals] for k, vals in grid_def.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    command = "python -m eval.benchmark_surface " + " ".join(argv if argv is not None else sys.argv[1:])
    base = Params()
    now = lambda: dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")  # noqa: E731
    calib_path = out / f"{PREFIX}calib.json"
    test_path = out / f"{PREFIX}test_summary.json"

    if args.split == "report":
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        test = json.loads(test_path.read_text(encoding="utf-8"))
        write_report(Path(args.report), calib, test)
        print(f"report: {args.report}")
        return 0
    env = environment(args.workers)

    if args.split in ("calib", "all"):
        grids = {m: threshold_grid([getattr(base, method_param(m))]) for m in methods}
        cases = build_cases("calib", args.seed, args.n, args.clean_mult, **grid_def)
        t0 = time.perf_counter()
        recs, errors = run_grid(cases, size, methods, base, grids, args.workers, "calib")
        cal = calibrate(recs, methods, grids, FA_TARGET)
        calib = {"command": command, "date": now(), "env": env, "seed": args.seed, "n": args.n,
                 "clean_mult": args.clean_mult, "size": size, "grid": grid_json, "methods": methods,
                 "target_fa_rate": FA_TARGET, "n_photos": len(recs), "errors": errors,
                 "wall_s": time.perf_counter() - t0, "params": base.to_dict(),
                 "thresholds": {m: cal[m]["threshold"] for m in methods},
                 "param_names": {m: method_param(m) for m in methods}, "calibration": cal,
                 "consistency_failures": sum(not mm["consistent"] for r in recs for mm in r["methods"].values())}
        _dump(calib_path, calib)
        _dump(out / f"{PREFIX}calib_sweep.json", sweep_curves(recs, methods, grids))
        print("calibrated:", {m: cal[m]["threshold"] for m in methods}, flush=True)

    if args.split in ("test", "all"):
        if not calib_path.exists():
            raise SystemExit(f"{calib_path} not found: run --split calib first")
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        missing = [m for m in methods if m not in calib["thresholds"]]
        if missing:
            raise SystemExit(f"no calibrated threshold for {missing}")
        P = dataclasses.replace(base, **{method_param(m): calib["thresholds"][m] for m in methods})
        grids = {m: threshold_grid([calib["thresholds"][m]]) for m in methods}
        cases = build_cases("test", args.seed, args.n, args.clean_mult, **grid_def)
        t0 = time.perf_counter()
        recs, errors = run_grid(cases, size, methods, P, grids, args.workers, "test")
        wall = time.perf_counter() - t0
        subs = subsets(recs)
        clean = subs.get("none+monotone", [])
        kind_defect = {}
        if "pbp" in methods:
            for kd in synth.SURFACE_KINDS:
                row = {}
                for d in BENCH_DEFECTS:
                    sub = [r for r in clean if r["kind"] == kd and r["defect"] == d]
                    if sub:
                        row[d] = f"{sum(r['methods']['pbp']['detected'] for r in sub)}/{len(sub)}"
                if row:
                    kind_defect[kd] = row
        test = {
            "command": command, "date": now(), "env": env, "seed": args.seed, "n": args.n,
            "clean_mult": args.clean_mult, "size": size, "grid": grid_json, "methods": methods,
            "thresholds": calib["thresholds"], "params": P.to_dict(), "wall_s": wall,
            "summary": {k: {m: summarize(v, m) for m in methods} for k, v in subs.items()},
            "per_defect": {d: {m: summarize([r for r in clean if r["defect"] == d], m) for m in methods}
                           for d in BENCH_DEFECTS if any(r["defect"] == d for r in clean)},
            "clean_by_kind": {kd: {m: summarize([r for r in clean if r["kind"] == kd], m) for m in methods}
                              for kd in synth.SURFACE_KINDS if any(r["kind"] == kd for r in clean)},
            "kind_defect_pbp": kind_defect,
            "agreement": {
                "n_defect": sum(r["defect"] != "none" for r in recs),
                "same": sum(r["defect"] != "none" and len({r["methods"][m]["detected"] for m in methods}) == 1
                            for r in recs),
                "n_good": sum(r["defect"] == "none" for r in recs),
                "same_good": sum(r["defect"] == "none" and len({r["methods"][m]["verdict"] for m in methods}) == 1
                                 for r in recs)},
            "auroc": {grp: {m: separability(subs.get(grp, []), m) for m in methods}
                      for grp in ("none+monotone", "non_monotone", "all")},
            "consistency_failures": sum(not mm["consistent"] for r in recs for mm in r["methods"].values()),
            "errors": errors,
        }
        print("timing pass ...", flush=True)
        test["timing"] = timing_pass(cases, size, methods, P, args.timing_photos)
        _dump(test_path, test)
        _dump(out / f"{PREFIX}test_sweep.json", sweep_curves(recs, methods, grids))
        _dump(out / f"{PREFIX}test_photos.json", _photo_rows(recs), compact=True)
        test = json.loads(test_path.read_text(encoding="utf-8"))  # report from exactly what was saved
        write_report(Path(args.report), calib, test)
        s = test["summary"]
        for k in ("none+monotone", "non_monotone", "all"):
            if k in s:
                print(k, {m: (s[k][m]["recall"], s[k][m]["fa_rate"]) for m in methods}, flush=True)
        print(f"report: {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
