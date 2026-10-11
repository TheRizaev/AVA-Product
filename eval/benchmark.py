"""Reproducible benchmark of the AVA pipeline: synthetic station pairs (+ optional VisA).

    python -m eval.benchmark --n 4 --seed 0            # calibrate, then test, then write eval/RESULTS.md
    python -m eval.benchmark --split calib --n 4        # calibration only (eval/results/calib.json)
    python -m eval.benchmark --split test --n 4         # test with the frozen calib.json thresholds

Protocol (also written into RESULTS.md):
1. Grid: part kind x lighting (none + every synth.FACTORY_LIGHTING kind) x motion
   (jitter, shift) x defect (none + BENCH_DEFECTS), `--n` pairs per cell. Every pair has
   its own seed derived from (--seed, split, cell, repeat); the CALIB and TEST splits use
   different split codes, so their random streams are disjoint.
2. Each pair goes through `pipeline.inspect_compare` once for all methods: alignment, glare
   mask, tolerance and zone extraction are identical, only features/distances differ. The
   smoothed change map it returns is re-thresholded with `pipeline.decide` (the same code
   the pipeline uses) over a threshold grid.
3. CALIB: per method, the smallest grid threshold t at which the mean number of false
   events per defect-free pair under "none + monotone" lighting is <= FALSE_EVENT_TARGET.
   A false event is a zone that does not touch the defect (on a defect-free pair: every
   zone), or a "mismatch" retake of a good part.
4. TEST: the thresholds are frozen and every metric is computed on the TEST split.
5. Timing: a separate sequential pass in the main process (all cores available to OpenCV),
   because times measured in the parallel workers include CPU contention.

A defect pair counts as detected if its verdict is "fail" and some zone bbox has IoU > 0.1
with the true defect bbox or covers >= 30 % of it (bboxes at the working resolution).
"""

from __future__ import annotations

import argparse
import csv
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
if str(ROOT) not in sys.path:  # allow `python eval/benchmark.py`
    sys.path.insert(0, str(ROOT))

from ava import pipeline, synth  # noqa: E402
from ava.config import Params  # noqa: E402

METHODS = ("pbp", "census", "ssim", "pixdiff")
BENCH_DEFECTS = ("scratch", "tape", "missing_hole", "burr", "dent", "stain")
DEFECTS = (None,) + BENCH_DEFECTS
LIGHTINGS = (None,) + synth.FACTORY_LIGHTING
MOTIONS = ("jitter", "shift")
SPLIT_CODES = {"calib": 1, "test": 2}
FALSE_EVENT_TARGET = 0.05
IOU_MIN = 0.1
COVER_MIN = 0.3
CLEAN_LIGHTINGS = ("none",) + synth.MONOTONE_LIGHTING   # the calibration domain

# Sweep outcome codes (column 0 of a sweep row).
PASS, FAIL, RETAKE_GATE, RETAKE_MISMATCH = 0, 1, 2, 3
_CODES = {"pass": PASS, "fail": FAIL}

VISA_FIXED_POSE = ("pcb1", "pcb2", "pcb3", "pcb4", "macaroni1", "candle")
VISA_LICENSE = "CC BY 4.0"


# --------------------------------------------------------------------------- small helpers

def threshold_grid(extra=()) -> np.ndarray:
    """Geometric grid 0.002..0.95 (140 steps, ~4.5 % apart) plus `extra` values."""
    g = np.geomspace(0.002, 0.95, 140)
    return np.unique(np.round(np.concatenate([g, np.asarray(list(extra), float)]), 5))


def method_param(method: str) -> str:
    return "threshold" if method == "pbp" else f"threshold_{method}"


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
    allv = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):  # average ranks over ties
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


def sweep_rows(res, truth: Truth, params: Params, grid: np.ndarray) -> np.ndarray:
    """int16[len(grid), 4] = (code, n_zones, n_false, detected) per threshold."""
    rows = np.zeros((len(grid), 4), np.int16)
    if res.score is None:  # rejected before scoring (alignment / glare / blur / dark / coverage)
        rows[:, 0] = RETAKE_GATE
        return rows
    for i, t in enumerate(grid):
        verdict, reason, zones, _ = pipeline.decide(res.score, res.checked, params, float(t),
                                                    bright=res.bright_spot)
        hit, n_false = truth.match(zones)
        code = (RETAKE_MISMATCH if reason == "mismatch"
                else RETAKE_GATE if verdict == "retake" else _CODES[verdict])  # retake: bright spot rule
        rows[i] = (code, len(zones), n_false, int(hit and verdict == "fail"))
    return rows


def max_score(res) -> float | None:
    if res.score is None or not res.checked.any():
        return None
    return float(res.score[res.checked].max())


# --------------------------------------------------------------------------- synthetic grid

@dataclass(frozen=True)
class Case:
    part: str
    lighting: str | None
    motion: str
    defect: str | None
    rep: int
    entropy: tuple


def build_cases(split: str, seed: int, n: int, parts, lightings, motions, defects) -> list[Case]:
    """All grid cells x `n` repeats; seeds depend only on the full-grid position (filter-stable)."""
    cases = []
    for p in parts:
        for li in lightings:
            for mo in motions:
                for d in defects:
                    for r in range(n):
                        ent = (int(seed), SPLIT_CODES[split], synth.PART_KINDS.index(p),
                               LIGHTINGS.index(li), MOTIONS.index(mo), DEFECTS.index(d), r)
                        cases.append(Case(p, li, mo, d, r, ent))
    return cases


def _init_worker():
    cv2.setNumThreads(1)  # one process per core; no nested OpenCV threads


def run_case(task) -> dict:
    """Worker entry: `_run_case`, or an error record if it raised (reported, never hidden)."""
    try:
        return _run_case(task)
    except Exception as e:  # noqa: BLE001 - one bad pair must not abort an hour-long run
        case = task[0]
        return {"error": f"{type(e).__name__}: {e}"[:500], "part": case.part,
                "lighting": case.lighting or "none", "motion": case.motion,
                "defect": case.defect or "none", "rep": case.rep, "entropy": list(case.entropy)}


def _run_case(task) -> dict:
    """Generate one pair, inspect it with every method, sweep the thresholds (worker)."""
    case, size, methods, params_dict, grids = task
    P = Params(**params_dict)
    rng = np.random.default_rng(np.random.SeedSequence(list(case.entropy)))
    pair = synth.make_station_pair(rng, part_kind=case.part, defect=case.defect,
                                   lighting=case.lighting, motion=case.motion, h=size[0], w=size[1])
    ref = pipeline.prepare_reference([pair["reference"]], P, methods=methods)
    out = pipeline.inspect_compare(ref, pair["frame"], P, methods)
    truth = Truth(mask_to_work(pair["mask_ref_frame"], ref.work_size), P.win)
    meta = pair["meta"]
    rec = {"part": case.part, "background": meta["background"],
           "lighting": case.lighting or "none", "motion": case.motion,
           "defect": case.defect or "none", "rep": case.rep, "entropy": list(case.entropy),
           "mask_px": int(truth.mask.sum()), "methods": {}}
    for m in methods:
        res = out[m]
        grid = grids[m]
        rows = sweep_rows(res, truth, P, grid)
        hit, n_false = truth.match(res.zones)
        i_op = int(np.searchsorted(grid, res.threshold))
        expect = RETAKE_MISMATCH if res.reason == "mismatch" else \
            (RETAKE_GATE if res.verdict == "retake" else _CODES[res.verdict])
        consistent = (i_op < len(grid) and abs(grid[i_op] - res.threshold) < 1e-9
                      and rows[i_op, 0] == expect and rows[i_op, 1] == len(res.zones))
        rec["methods"][m] = {
            "verdict": res.verdict, "reason": res.reason, "n_zones": len(res.zones),
            "n_false": n_false, "detected": bool(hit and res.verdict == "fail"),
            "zones": [list(z.bbox) for z in res.zones], "threshold": res.threshold,
            "coverage": round(res.coverage, 4), "align": res.align.get("method"),
            "max_score": max_score(res), "ms_parallel": res.timing_ms.get("total"),
            "consistent": bool(consistent), "sweep": rows,
        }
    return rec


def run_grid(cases: list, size, methods, params: Params, grids: dict, workers: int,
             label: str) -> tuple[list, list]:
    """Run every case; returns (records, error records), both sorted by seed entropy."""
    tasks = [(c, size, tuple(methods), params.to_dict(), grids) for c in cases]
    recs = []
    t0 = time.perf_counter()
    step = max(1, len(tasks) // 20)

    def progress(i):
        if (i + 1) % step == 0 or i + 1 == len(tasks):
            el = time.perf_counter() - t0
            print(f"[{label}] {i + 1}/{len(tasks)} pairs, {el:.0f} s", flush=True)

    if workers <= 1:
        for i, t in enumerate(tasks):
            recs.append(run_case(t))
            progress(i)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, maxtasksperchild=40) as pool:
            for i, r in enumerate(pool.imap_unordered(run_case, tasks, chunksize=1)):
                recs.append(r)
                progress(i)
    recs.sort(key=lambda r: r["entropy"])
    errors = [r for r in recs if "error" in r]
    for e in errors:
        print(f"[{label}] ERROR in pair {e['entropy']}: {e['error']}", flush=True)
    return [r for r in recs if "error" not in r], errors


# --------------------------------------------------------------------------- aggregation

def lighting_group(name: str) -> str:
    if name == "none":
        return "none"
    return "monotone" if name in synth.MONOTONE_LIGHTING else "non_monotone"


def summarize(recs: list, method: str, ti: int | None = None) -> dict:
    """Metrics of `method` over `recs`, at sweep index `ti` (None = the stored operating point)."""
    n_def = n_good = det = esc = fa = rt_good = rt_all = fz_good = fz_all = 0
    cov, reasons = [], {}
    for r in recs:
        mm = r["methods"][method]
        if ti is None:
            code = (RETAKE_MISMATCH if mm["reason"] == "mismatch" else
                    RETAKE_GATE if mm["verdict"] == "retake" else _CODES[mm["verdict"]])
            nz, nf, hit = mm["n_zones"], mm["n_false"], mm["detected"]
            reason = mm["reason"]
        else:
            code, nz, nf, hit = (int(v) for v in mm["sweep"][ti])
            # A RETAKE that appears only in the sweep is the bright spot rule (decide(..., bright=True)).
            reason = ("mismatch" if code == RETAKE_MISMATCH
                      else (mm["reason"] or "glare") if code == RETAKE_GATE else None)
        retake = code in (RETAKE_GATE, RETAKE_MISMATCH)
        rt_all += retake
        fz_all += nf
        if retake:
            reasons[reason] = reasons.get(reason, 0) + 1
        if mm["max_score"] is not None:
            cov.append(mm["coverage"])
        if r["defect"] == "none":
            n_good += 1
            fa += code == FAIL
            rt_good += retake
            fz_good += nf + (code == RETAKE_MISMATCH)
        else:
            n_def += 1
            det += bool(hit)
            esc += code == PASS
    out = {
        "n_pairs": len(recs), "n_defect": n_def, "n_good": n_good,
        "recall": det / n_def if n_def else float("nan"), "recall_ci": wilson(det, n_def), "detected": det,
        "escape_rate": esc / n_def if n_def else float("nan"), "escape_ci": wilson(esc, n_def),
        "fa_rate": fa / n_good if n_good else float("nan"), "fa_ci": wilson(fa, n_good), "false_alarms": fa,
        "false_events_per_good": fz_good / n_good if n_good else float("nan"),
        "false_zones_per_pair": fz_all / len(recs) if recs else float("nan"),
        "retake_rate": rt_all / len(recs) if recs else float("nan"), "retake_ci": wilson(rt_all, len(recs)),
        "retake_rate_good": rt_good / n_good if n_good else float("nan"),
        "false_reject_rate": (fa + rt_good) / n_good if n_good else float("nan"),
        "false_reject_ci": wilson(fa + rt_good, n_good),
        "coverage": float(np.mean(cov)) if cov else float("nan"),
        "retake_reasons": reasons,
    }
    return out


def separability(recs: list, method: str) -> dict:
    """Threshold-free AUROC of the max smoothed cell score (defect vs defect-free pairs)."""
    pos = [r["methods"][method]["max_score"] for r in recs if r["defect"] != "none"]
    neg = [r["methods"][method]["max_score"] for r in recs if r["defect"] == "none"]
    excl = sum(v is None for v in pos + neg)
    pos = [v for v in pos if v is not None]
    neg = [v for v in neg if v is not None]
    return {"auroc": auroc(pos, neg), "n_pos": len(pos), "n_neg": len(neg), "n_unscored": excl}


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts b and c.

    Two-sided exact binomial test with p = 0.5 (symmetric): 2 * P(X <= min(b, c)), capped at 1;
    the same value as scipy.stats.binomtest(min(b, c), b + c, 0.5).pvalue, without the dependency.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def paired_comparisons(pairs: list, methods, ref: str = "pbp") -> dict:
    """PBP vs each other method on the SAME pairs (operating point), per lighting group.

    Recall: discordant defect pairs (detected by one method only); false alarms: discordant
    defect-free pairs (FAIL by one method only); exact McNemar p-values.
    """
    groups = {"none+monotone": lambda r: r["lighting"] in CLEAN_LIGHTINGS,
              "non_monotone": lambda r: lighting_group(r["lighting"]) == "non_monotone"}
    out = {}
    for g, sel in groups.items():
        sub = [r for r in pairs if sel(r)]
        for m in methods:
            if m == ref or not sub:
                continue
            d = [r for r in sub if r["defect"] != "none"]
            n = [r for r in sub if r["defect"] == "none"]
            b_rec = sum(r["methods"][ref]["detected"] and not r["methods"][m]["detected"] for r in d)
            c_rec = sum(r["methods"][m]["detected"] and not r["methods"][ref]["detected"] for r in d)
            fail = lambda r, k: r["methods"][k]["verdict"] == "fail"  # noqa: E731
            b_fa = sum(fail(r, ref) and not fail(r, m) for r in n)
            c_fa = sum(fail(r, m) and not fail(r, ref) for r in n)
            out[f"{g}:{m}"] = {"n_defect": len(d), "n_good": len(n),
                               "recall_only_ref": b_rec, "recall_only_other": c_rec, "recall_p": mcnemar(b_rec, c_rec),
                               "fa_only_ref": b_fa, "fa_only_other": c_fa, "fa_p": mcnemar(b_fa, c_fa)}
    return out


def subsets(recs: list) -> dict:
    """Named record subsets used by the report (lighting kinds and groups, motion, part)."""
    out = {"all": recs,
           "none+monotone": [r for r in recs if r["lighting"] in CLEAN_LIGHTINGS],
           "none": [r for r in recs if r["lighting"] == "none"],
           "monotone": [r for r in recs if lighting_group(r["lighting"]) == "monotone"],
           "non_monotone": [r for r in recs if lighting_group(r["lighting"]) == "non_monotone"]}
    for k in synth.FACTORY_LIGHTING:
        out[f"lighting:{k}"] = [r for r in recs if r["lighting"] == k]
    for k in MOTIONS:
        out[f"motion:{k}"] = [r for r in recs if r["motion"] == k]
    for k in synth.PART_KINDS:
        out[f"part:{k}"] = [r for r in recs if r["part"] == k]
    return {k: v for k, v in out.items() if v}


def sweep_curves(recs: list, methods, grids) -> dict:
    """Per method and lighting group: recall / FA rate / false events per threshold."""
    groups = {"all": recs,
              "none+monotone": [r for r in recs if r["lighting"] in CLEAN_LIGHTINGS],
              "non_monotone": [r for r in recs if lighting_group(r["lighting"]) == "non_monotone"]}
    out = {}
    for m in methods:
        out[m] = {"thresholds": [float(t) for t in grids[m]]}
        for g, sub in groups.items():
            if not sub:
                continue
            arr = np.stack([r["methods"][m]["sweep"] for r in sub])          # [n, T, 4]
            good = np.array([r["defect"] == "none" for r in sub])
            code, nf, hit = arr[..., 0], arr[..., 2], arr[..., 3]
            ng, nd = int(good.sum()), int((~good).sum())
            out[m][g] = {
                "recall": (hit[~good].sum(0) / nd).round(4).tolist() if nd else None,
                "fa_rate": ((code[good] == FAIL).sum(0) / ng).round(4).tolist() if ng else None,
                "false_events_per_good": ((nf[good] + (code[good] == RETAKE_MISMATCH)).sum(0) / ng)
                .round(4).tolist() if ng else None,
                "retake_rate": (np.isin(code, (RETAKE_GATE, RETAKE_MISMATCH)).sum(0) / len(sub))
                .round(4).tolist(),
            }
    return out


def calibrate(recs: list, methods, grids, target: float) -> dict:
    """Per method: the smallest grid threshold whose false events per good pair are <= target.

    Counting "mismatch" retakes as false events keeps very low thresholds (where every
    good part is rejected as a mismatch) from qualifying.
    """
    good = [r for r in recs if r["defect"] == "none" and r["lighting"] in CLEAN_LIGHTINGS]
    if not good:
        raise SystemExit("calibration needs defect-free pairs under none/monotone lighting")
    out = {}
    for m in methods:
        arr = np.stack([r["methods"][m]["sweep"] for r in good])
        events = (arr[..., 2] + (arr[..., 0] == RETAKE_MISMATCH)).mean(0)
        ok = events <= target
        reached = bool(ok.any())
        i = int(np.argmax(ok)) if reached else len(grids[m]) - 1
        out[m] = {"threshold": float(grids[m][i]), "reached": reached,
                  "false_events_per_good": float(events[i]),
                  "fa_rate": float((arr[:, i, 0] == FAIL).mean()), "n_good_pairs": len(good)}
    return out


# --------------------------------------------------------------------------- timing

def timing_pass(cases: list, size, methods, params: Params, k: int) -> dict:
    """Sequential single-method inspections in this process (OpenCV threads enabled)."""
    if k <= 0 or not cases:
        return {}
    cv2.setNumThreads(-1)
    idx = np.linspace(0, len(cases) - 1, k + 1).round().astype(int)
    picked = [cases[i] for i in dict.fromkeys(idx.tolist())]
    per = {m: [] for m in methods}
    stages_pbp: dict = {}
    prep = []
    for j, c in enumerate(picked):
        rng = np.random.default_rng(np.random.SeedSequence(list(c.entropy)))
        pair = synth.make_station_pair(rng, part_kind=c.part, defect=c.defect, lighting=c.lighting,
                                       motion=c.motion, h=size[0], w=size[1])
        t0 = time.perf_counter()
        ref = pipeline.prepare_reference([pair["reference"]], params, methods=("pbp",))
        prep.append((time.perf_counter() - t0) * 1000)
        pipeline.prepare_reference([pair["reference"]], params, methods=tuple(methods))  # warm
        for m in methods:
            ref_m = ref if m == "pbp" else pipeline.prepare_reference(
                [pair["reference"]], params, methods=(m,))
            res = pipeline.inspect_frame(ref_m, pair["frame"], params, m)
            if j == 0:
                continue  # first round warms up caches / lazy imports
            per[m].append(res.timing_ms["total"])
            if m == "pbp" and res.score is not None:
                for st, v in res.timing_ms.items():
                    stages_pbp.setdefault(st, []).append(v)
    out = {m: {"n": len(v), "mean_ms": float(np.mean(v)), "median_ms": float(np.median(v)),
               "min_ms": float(np.min(v)), "max_ms": float(np.max(v))} for m, v in per.items() if v}
    out["_pbp_stage_median_ms"] = {st: float(np.median(v)) for st, v in stages_pbp.items()}
    out["_prepare_reference_pbp_median_ms"] = float(np.median(prep[1:] or prep))
    return out


# --------------------------------------------------------------------------- VisA (optional)

def _visa_items(root: Path, cat: str) -> tuple[list, list]:
    """(train normal image paths, test [(image path, mask path | None, is_anomaly)]) of a category."""
    csv_path = root / "split_csv" / "1cls.csv"
    train, test = [], []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["object"] != cat:
                    continue
                img = root / row["image"]
                if row["split"] == "train" and row["label"] == "normal":
                    train.append(img)
                elif row["split"] == "test":
                    mask = root / row["mask"] if row.get("mask") else None
                    test.append((img, mask, row["label"] == "anomaly"))
    else:  # no split file: the first half of the sorted normal images are "train"
        normal = sorted((root / cat / "Data" / "Images" / "Normal").glob("*.*"))
        train, rest = normal[: len(normal) // 2], normal[len(normal) // 2:]
        test = [(p, None, False) for p in rest]
        for p in sorted((root / cat / "Data" / "Images" / "Anomaly").glob("*.*")):
            test.append((p, root / cat / "Data" / "Masks" / "Anomaly" / (p.stem + ".png"), True))
    return sorted(train), sorted(test, key=lambda t: str(t[0]))


def _read(path: Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), np.uint8)  # works with non-ASCII paths on Windows
    return cv2.imdecode(data, flags)


def run_visa_category(task) -> dict:
    """Inspect one VisA category (worker): k normal train references, all test images."""
    root, cat, k, max_per_class, params_dict, methods, seed = task
    P = Params(**params_dict)
    train, test = _visa_items(Path(root), cat)
    shots = []
    for p in train:  # first k normal shots that register to the first one
        img = _read(p)
        if img is None:
            continue
        try:
            pipeline.prepare_reference(shots + [img], P, methods=())
        except ValueError:
            continue
        shots.append(img)
        if len(shots) == k:
            break
    if not shots:
        return {"category": cat, "error": "no reference"}
    ref = pipeline.prepare_reference(shots, P, methods=methods)
    rng = np.random.default_rng(seed)
    if max_per_class:
        pick = []
        for cls in (False, True):
            items = [t for t in test if t[2] == cls]
            sel = rng.choice(len(items), size=min(max_per_class, len(items)), replace=False) if items else []
            pick += [items[i] for i in sorted(sel)]
        test = pick
    rows = []
    for img_path, mask_path, anomalous in test:
        img = _read(img_path)
        if img is None:
            continue
        mask = None
        if anomalous and mask_path is not None and Path(mask_path).exists():
            mask = _read(Path(mask_path), cv2.IMREAD_GRAYSCALE)
        for lit in ("none", "monotone"):
            frame = img
            kind = None
            if lit == "monotone":
                kind = synth.MONOTONE_LIGHTING[int(rng.integers(len(synth.MONOTONE_LIGHTING)))]
                frame = synth.apply_factory_lighting(img, rng, kind)
            out = pipeline.inspect_compare(ref, frame, P, methods)
            truth = Truth(mask_to_work(mask > 0, ref.work_size) if mask is not None
                          else np.zeros(ref.work_size[::-1], bool), P.win)
            row = {"category": cat, "image": Path(img_path).name, "anomalous": anomalous,
                   "lighting": lit, "lighting_kind": kind, "methods": {}}
            for m in methods:
                res = out[m]
                hit, n_false = truth.match(res.zones)
                row["methods"][m] = {"verdict": res.verdict, "reason": res.reason,
                                     "n_zones": len(res.zones), "located": bool(hit and res.verdict == "fail"),
                                     "max_score": max_score(res), "ms": res.timing_ms.get("total")}
            rows.append(row)
    return {"category": cat, "n_refs": len(shots), "rows": rows}


def run_visa(root: Path, cats, k, max_per_class, params: Params, methods, workers, seed) -> dict:
    tasks = [(str(root), c, k, max_per_class, params.to_dict(), tuple(methods), seed + i)
             for i, c in enumerate(cats)]
    if workers <= 1:
        results = [run_visa_category(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(min(workers, len(tasks)), initializer=_init_worker) as pool:
            results = pool.map(run_visa_category, tasks, chunksize=1)
    rows = [r for res in results for r in res.get("rows", [])]
    summary = {}
    for lit in ("none", "monotone"):
        for m in methods:
            sub = [r for r in rows if r["lighting"] == lit]
            pos = [r["methods"][m]["max_score"] for r in sub if r["anomalous"]]
            neg = [r["methods"][m]["max_score"] for r in sub if not r["anomalous"]]
            n_pos, n_neg = len(pos), len(neg)
            fail_pos = sum(r["methods"][m]["verdict"] == "fail" for r in sub if r["anomalous"])
            fail_neg = sum(r["methods"][m]["verdict"] == "fail" for r in sub if not r["anomalous"])
            loc = sum(r["methods"][m]["located"] for r in sub if r["anomalous"])
            rt = sum(r["methods"][m]["verdict"] == "retake" for r in sub)
            summary[f"{lit}:{m}"] = {
                "auroc": auroc([v for v in pos if v is not None], [v for v in neg if v is not None]),
                "n_anomalous": n_pos, "n_normal": n_neg,
                "recall": fail_pos / n_pos if n_pos else float("nan"), "recall_ci": wilson(fail_pos, n_pos),
                "located": loc / n_pos if n_pos else float("nan"),
                "fa_rate": fail_neg / n_neg if n_neg else float("nan"), "fa_ci": wilson(fail_neg, n_neg),
                "retake_rate": rt / len(sub) if sub else float("nan"),
                "n_unscored": sum(v is None for v in pos + neg)}
    cats_info = [{"category": r["category"], "n_refs": r.get("n_refs"), "error": r.get("error")}
                 for r in results]
    return {"categories": cats_info, "summary": summary, "rows": rows}


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
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "ava", "eval", "server"],
                               cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        return (rev or "unknown") + (" + uncommitted changes in ava/eval/server" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment(workers: int) -> dict:
    return {"cpu": cpu_name(), "logical_cores": os.cpu_count(), "os": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
            "git": git_state(), "workers": workers}


# --------------------------------------------------------------------------- report

def _na(x) -> bool:
    return x is None or x != x  # NaN is stored as null in the JSON


def _pct(x: float) -> str:
    return "n/a" if _na(x) else f"{100 * x:.1f}%"


def _rate(x: float, ci) -> str:
    if _na(x) or _na(ci[0]):
        return "n/a"
    return f"{100 * x:.1f}% ({100 * ci[0]:.1f}–{100 * ci[1]:.1f})"


def _f(x: float, nd: int = 3) -> str:
    return "n/a" if _na(x) else f"{x:.{nd}f}"


def _table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(path: Path, calib: dict, test: dict, visa: dict | None, visa_note: str | None) -> None:
    methods = test["methods"]
    S = test["summary"]
    env = test["env"]
    thr = calib["thresholds"]
    n_test = S["all"][methods[0]]["n_pairs"]
    L = []
    L.append("# Synthetic benchmark — procedurally generated parts, NOT real factory images\n")
    L.append("Every number in this file was produced by `eval/benchmark.py` (raw data in `eval/results/`). "
             "The pairs come from `ava/synth.py`: procedural stamped / painted / molded / machined parts on "
             "belt / table / ESD-mat backgrounds, procedural defects and procedural shop-floor lighting. "
             "They show how the methods compare under controlled conditions; they are **not** evidence "
             "of accuracy on real parts.\n")
    L.append("## Run\n")
    cmds = [calib["command"]] if calib["command"] == test["command"] else [calib["command"], test["command"]]
    L.append("```\n" + "\n".join(cmds) + "\n```\n")
    L.append(_table(["", ""], [
        ["Date (calib / test)", f"{calib['date']} / {test['date']}"],
        ["CPU", f"{env['cpu']} ({env['logical_cores']} logical cores)"],
        ["Software", f"{env['os']}; Python {env['python']}, numpy {env['numpy']}, OpenCV {env['opencv']}"],
        ["Code", env["git"]],
        ["Grid", f"{len(test['grid']['parts'])} parts x {len(test['grid']['lightings'])} lightings x "
                 f"{len(test['grid']['motions'])} motions x {len(test['grid']['defects'])} defect classes "
                 f"x {test['n']} = {n_test} TEST pairs ({calib['n_pairs']} CALIB pairs)"],
        ["Image size", f"{test['size'][1]}x{test['size'][0]} px generated, pipeline working size "
                       f"max side {test['params']['work_max_side']} px; 1 reference shot per pair"],
        ["Seeds", f"--seed {test['seed']}; per pair SeedSequence(seed, split code calib=1/test=2, cell, repeat)"],
        ["Wall time", f"calib {calib['wall_s']:.0f} s, test {test['wall_s']:.0f} s with {env['workers']} worker "
                      f"processes (OpenCV 1 thread each)"],
        ["Pairs that raised an exception", f"calib {len(calib.get('errors', []))}, test {len(test.get('errors', []))}"
                                           " (excluded from every metric; listed in the JSON)"],
        ["Sweep vs pipeline consistency", f"{calib.get('consistency_failures', 0)} + "
                                          f"{test.get('consistency_failures', 0)} mismatches between the re-thresholded "
                                          "map and the pipeline's own verdict at the operating point"],
    ]) + "\n")
    L.append("## Protocol\n")
    L.append(f"""1. **Pairs.** `synth.make_station_pair` gives a (reference, frame) pair: the frame may carry a defect
   ({', '.join(BENCH_DEFECTS)}; `sticker` is not used), is moved (`jitter` = camera vibration ±2 px / 0.3°,
   `shift` = the part moved on the belt by up to 40 px / 3° with the camera fixed), and is relit with one
   shop-floor lighting kind or none. Both images get independent sensor noise.
2. **Same pipeline for every method.** `pipeline.inspect_compare` aligns once, builds one glare/validity mask
   and scores PBP, Census, SSIM and plain reference subtraction (`pixdiff`) on the same aligned frame, with the
   same tolerance, smoothing and zone rules. Only the window features and distances differ (PBP additionally
   has its low-contrast gate).
3. **Calibration (CALIB split only).** Each method's smoothed change map is re-thresholded with
   `pipeline.decide` over a geometric grid (0.002–0.95). For each method we take the smallest grid threshold
   at which the mean number of *false events* per defect-free pair under **none + monotone** lighting is
   ≤ {FALSE_EVENT_TARGET}. A false event is a zone that does not touch the defect (dilated by one window), or a
   `mismatch` retake of a good part (so the degenerate "reject everything as mismatch" thresholds do not
   qualify). The thresholds are then frozen.
4. **Test (TEST split, disjoint seeds).** All metrics below. *Recall*: a defect pair is detected if the
   verdict is FAIL and some zone has IoU > {IOU_MIN} with the true defect bbox or covers ≥ {int(COVER_MIN * 100)} % of it.
   *Escapes*: defect pairs judged PASS. *False-alarm rate*: fraction of defect-free pairs judged FAIL;
   *good parts not passed* also counts RETAKE answers on defect-free pairs. *False zones*: zones not touching
   the defect, per pair (all pairs). *Retake rate*: fraction of pairs answered RETAKE (alignment, glare, blur, low coverage, mismatch).
   *Coverage*: mean fraction of cells checked, over scored pairs. Brackets: Wilson 95 % intervals.
5. **Timing.** A separate sequential pass in one process (OpenCV threads on), one method per inspection,
   {test['timing'].get('pbp', {}).get('n', 0)} TEST pairs after one warm-up pair; times include resize, quality
   checks, alignment, masks, features, change map, zones and the overlay JPEGs.
""")
    L.append("## Calibrated thresholds (frozen before TEST)\n")
    rows = []
    for m in methods:
        c = calib["calibration"][m]
        rows.append([m, f"`{method_param(m)}`", f"{c['threshold']:.4g}" + ("" if c["reached"] else " (target not reached)"),
                     _f(c["false_events_per_good"]), _pct(c["fa_rate"]), c["n_good_pairs"]])
    L.append(_table(["Method", "Param", "Threshold", "CALIB false events / good pair", "CALIB false-alarm rate",
                     "CALIB good pairs (none+monotone)"], rows) + "\n")

    def headline(key: str, title: str) -> None:
        rows = []
        for m in methods:
            s = S[key][m]
            rows.append([m, _rate(s["recall"], s["recall_ci"]), _rate(s["escape_rate"], s["escape_ci"]),
                         _rate(s["fa_rate"], s["fa_ci"]), _rate(s["false_reject_rate"], s["false_reject_ci"]),
                         _f(s["false_zones_per_pair"]), _rate(s["retake_rate"], s["retake_ci"]),
                         _pct(s["coverage"]), _f(test["timing"].get(m, {}).get("mean_ms", float("nan")), 0)])
        n = S[key][methods[0]]
        L.append(f"### {title} — {n['n_defect']} defect pairs, {n['n_good']} defect-free pairs\n")
        L.append(_table(["Method", "Defect recall", "Escapes (defect → PASS)", "False-alarm rate (good → FAIL)",
                         "Good parts not passed (FAIL or RETAKE)", "False zones / pair", "Retake rate (all)",
                         "Coverage", "Mean ms / inspection"], rows) + "\n")

    L.append("## TEST results\n")
    headline("none+monotone", "Lighting: none + monotone (the operating domain)")
    headline("non_monotone", "Lighting: non-monotone (specular highlight, hand shadow) — expected failure")
    headline("all", "All lighting conditions")

    L.append("### By lighting kind — defect recall / false-alarm rate\n")
    keys = ["none"] + [f"lighting:{k}" for k in synth.MONOTONE_LIGHTING] + ["monotone"] + \
           [f"lighting:{k}" for k in synth.NON_MONOTONE_LIGHTING] + ["non_monotone"]
    rows = []
    for k in keys:
        if k not in S:
            continue
        name = k.split(":", 1)[-1]
        name = {"monotone": "**monotone (all)**", "non_monotone": "**non-monotone (all)**"}.get(name, name)
        grp = "" if k in ("monotone", "non_monotone") else (
            "—" if name == "none" else lighting_group(name).replace("_", "-"))
        rows.append([name, grp] + [f"{_rate(S[k][m]['recall'], S[k][m]['recall_ci'])} / "
                                   f"{_rate(S[k][m]['fa_rate'], S[k][m]['fa_ci'])}" for m in methods])
    L.append(_table(["Lighting", "Group"] + methods, rows) + "\n")

    L.append("### Per-defect recall (none + monotone lighting)\n")
    rows = []
    for d in BENCH_DEFECTS:
        sub = test["per_defect"].get(d)
        if sub:
            rows.append([d] + [_rate(sub[m]["recall"], sub[m]["recall_ci"]) for m in methods])
    L.append(_table(["Defect"] + methods, rows) + "\n")

    L.append("### By motion and part kind (none + monotone lighting) — recall / false-alarm rate\n")
    rows = []
    for k in [f"motion:{k}" for k in MOTIONS] + [f"part:{k}" for k in synth.PART_KINDS]:
        sub = test["clean_by"].get(k)
        if sub:
            rows.append([k] + [f"{_pct(sub[m]['recall'])} / {_pct(sub[m]['fa_rate'])}" for m in methods])
    L.append(_table(["Subset"] + methods, rows) + "\n")

    L.append("### Retake reasons (all TEST pairs)\n")
    rows = [[m, ", ".join(f"{k}: {v}" for k, v in sorted(S["all"][m]["retake_reasons"].items(),
                                                       key=lambda kv: -kv[1])) or "—"] for m in methods]
    L.append(_table(["Method", "Reasons"], rows) + "\n")

    L.append("### Threshold-free separability (AUROC of the max smoothed cell score, defect vs defect-free)\n")
    rows = []
    for m in methods:
        rows.append([m] + [_f(test["auroc"][g][m]["auroc"]) for g in ("none+monotone", "non_monotone", "all")])
    L.append(_table(["Method", "none + monotone", "non-monotone", "all"], rows))
    L.append("\nPairs rejected before scoring (alignment / glare / blur / coverage) have no score and are left out "
             "of the AUROC. The full recall-vs-false-alarm curves are in `eval/results/test_sweep.json` "
             "(and `calib_sweep.json`).\n")

    paired = test.get("paired")
    if paired:
        L.append("### Paired comparison: PBP vs each method on the same TEST pairs\n")
        rows = []
        for g in ("none+monotone", "non_monotone"):
            for m in methods:
                c = paired.get(f"{g}:{m}")
                if not c:
                    continue

                def verdict(b, cc, pv, higher_is_better):
                    if pv >= 0.05:
                        return "no significant difference"
                    better = (b > cc) == higher_is_better
                    return "PBP better" if better else f"{m} better"

                rows.append([g, m,
                             f"{_pct(S[g]['pbp']['recall'])} vs {_pct(S[g][m]['recall'])}",
                             f"{c['recall_only_ref']} / {c['recall_only_other']}", f"{c['recall_p']:.2g}",
                             verdict(c["recall_only_ref"], c["recall_only_other"], c["recall_p"], True),
                             f"{_pct(S[g]['pbp']['fa_rate'])} vs {_pct(S[g][m]['fa_rate'])}",
                             f"{c['fa_only_ref']} / {c['fa_only_other']}", f"{c['fa_p']:.2g}",
                             verdict(c["fa_only_ref"], c["fa_only_other"], c["fa_p"], False)])
        L.append(_table(["Lighting", "vs", "Recall PBP vs it", "Defects found only by PBP / only by it",
                         "McNemar p", "Recall", "False alarms PBP vs it", "False alarms only PBP / only it",
                         "McNemar p", "False alarms"], rows))
        L.append("\nExact two-sided McNemar tests on the discordant pairs (each pair is seen by both methods), "
                 "at the frozen thresholds; p-values are not corrected for the number of comparisons.\n")

    t = test["timing"]
    L.append("## Speed (sequential pass, this machine)\n")
    rows = [[m, _f(t[m]["mean_ms"], 0), _f(t[m]["median_ms"], 0), f"{t[m]['min_ms']:.0f}–{t[m]['max_ms']:.0f}",
             t[m]["n"]] for m in methods if m in t]
    L.append(_table(["Method", "Mean ms", "Median ms", "Min–max ms", "Inspections"], rows) + "\n")
    st = t.get("_pbp_stage_median_ms", {})
    if st:
        L.append("PBP stage medians (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in st.items()) +
                 f". `prepare_reference` (1 shot, PBP): median {t['_prepare_reference_pbp_median_ms']:.0f} ms. "
                 "Means include inspections that stopped early at a RETAKE gate; stage medians are over the "
                 "inspections that reached the stage (`extra_views`: moved part only).\n")
    L.append(f"CPU: {env['cpu']}. Frames {test['size'][1]}x{test['size'][0]} resized to max side "
             f"{test['params']['work_max_side']} px. Measured {test.get('timing_date', test['date'])}"
             + (f" with `{test['timing_command']}`" if test.get("timing_command") else "") +
             ". Other processes on the machine were not controlled; treat as indicative.\n")

    L.append("## Caveats (read before quoting any number)\n")
    L.append(f"""- **Synthetic data only.** Parts, defects and lighting are procedural (`ava/synth.py`). Real scratches, dents
  and stains on real parts look different, and real cameras add compression, rolling shutter, motion blur and
  defocus that are only partly modelled. These numbers compare the methods under identical controlled
  conditions; they are not a real-world accuracy claim.
- **Non-monotone lighting is an expected failure.** A specular highlight or a hand shadow changes the local
  intensity order, which is exactly what PBP (and Census) rely on. Under those conditions good parts are
  judged FAIL by every method (see the non-monotone table); only a part of it is turned into RETAKE by the
  glare mask and the bright spot rule (a glare-like blob that only the frame has never ends in PASS).
- **The lighting kinds are pixel-wise.** `synth.apply_factory_lighting` changes gain, gamma, gradients,
  highlights and shadows on the rendered image; it never moves the light source. Moving a directional lamp
  over a relief part flips the shading of ribs and edges and is not covered by these numbers.
- **Census is PBP's closest relative** (both compare the intensity *order*, both ignore monotone changes). In
  this implementation Census compares single pixels with their 8 neighbours with no tie margin and no
  contrast gate, so sensor noise flips bits on smooth surfaces and its calibrated threshold has to be high.
  That is a likely reason for its low recall here, not a tested one; a Census variant with a noise margin
  was not evaluated.
- **SSIM** normalises every window by its own mean and variance, so it is also largely insensitive to a
  gain change. Whether PBP differs from it on these pairs is in the paired comparison above; the large,
  consistent gap is to plain reference subtraction (pixdiff), the classic golden-template approach.
- **Part moved on the belt (`shift`) is harder than camera jitter** for every method (see "By motion"). With
  a moved part the pipeline checks only the hull of the part's keypoints and, when the alignment found the
  static background as a second motion, takes per cell the minimum over the part-registered and
  background-registered views, which also lowers defect scores near shifted content.
- **Small N.** {test['n']} pair(s) per grid cell; per-cell and per-class rates rest on few pairs, see the Wilson
  95 % intervals. Differences inside overlapping intervals are not evidence.
- **One calibration criterion for all methods** (false events ≤ {FALSE_EVENT_TARGET} per good pair under none + monotone
  lighting). A different operating point changes the trade-off; the full sweeps are in `eval/results/`.
- **The ground truth is a bbox.** Zones live on a 4 px window grid; recall uses the bbox IoU / coverage rule
  above, false zones use "does not touch the (dilated) defect mask".
- Defect classes are not equally hard: burrs are small blobs on an edge, and a bright burr under strong
  exposure can be masked as glare (unchecked; the frame is then answered RETAKE rather than PASS) — see the
  per-defect table.
""")
    L.append("## Real data\n")
    if visa:
        L.append(visa_section(visa, methods, thr))
    else:
        L.append("VisA (Amazon, CC BY 4.0) evaluation was **not run** in this benchmark (`--visa` not given). " +
                 (visa_note.strip() + " " if visa_note else "") +
                 "Real-data evaluation is **pending**: the team photo set in `data/pairs/` "
                 "(protocol in `data/pairs/README.md`). No real-data metric is claimed here.\n")
    path.write_text("\n".join(L), encoding="utf-8", newline="\n")


def visa_section(visa: dict, methods, thr) -> str:
    L = [f"### VisA — REAL images (Amazon VisA, {VISA_LICENSE}; images not redistributed)\n",
         f"Command: `{visa['command']}`. Categories: " +
         ", ".join(f"{c['category']} ({c.get('n_refs') or 0} refs{', ' + c['error'] if c.get('error') else ''})"
                   for c in visa["categories"]) +
         ". References: the first k=3 normal *train* images that register to each other; test: the normal and "
         "anomalous *test* images. Thresholds: the frozen synthetic calibration (no tuning on VisA). "
         "\"monotone\" re-runs the same test images with one random synthetic monotone lighting change each.\n"]
    rows = []
    for lit in ("none", "monotone"):
        for m in methods:
            s = visa["summary"][f"{lit}:{m}"]
            rows.append([lit, m, _f(s["auroc"]), _rate(s["recall"], s["recall_ci"]), _pct(s["located"]),
                         _rate(s["fa_rate"], s["fa_ci"]), _pct(s["retake_rate"]),
                         f"{s['n_anomalous']}/{s['n_normal']}"])
    L.append(_table(["Lighting", "Method", "Image AUROC (max score)", "Recall @ frozen thr", "Located",
                     "False-alarm rate", "Retake", "Anomalous/normal"], rows))
    L.append("\nVisA objects are photographed in a fixed pose but not pixel-registered and vary from unit to unit; "
             "a reference comparison flags that variation too. Image-level AUROC uses the max smoothed cell score "
             "(unscored = retaken images are left out).\n")
    return "\n".join(L)


# --------------------------------------------------------------------------- main

def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, float) and o != o:
        return None
    raise TypeError(type(o))


def _dump(path: Path, obj, compact: bool = False) -> None:
    text = json.dumps(obj, default=_jsonable, ensure_ascii=False,
                      **({"separators": (",", ":")} if compact else {"indent": 1}))
    path.write_text(text, encoding="utf-8", newline="\n")


def _pair_rows(recs: list) -> list:
    """Per-pair records without the sweep arrays (operating point only)."""
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
    ap.add_argument("--n", type=int, default=4, help="pairs per grid cell (default 4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=("calib", "test", "all", "report", "timing"), default="all",
                    help="'report' rebuilds the report from the saved JSON in --out; 'timing' re-runs only the "
                         "sequential timing pass (frozen thresholds) and rebuilds the report")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--report", default=str(ROOT / "eval" / "RESULTS.md"))
    ap.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--size", default="720x1280", help="generated image HxW (default 720x1280)")
    ap.add_argument("--parts", default=None, help="subset of part kinds (comma list)")
    ap.add_argument("--lightings", default=None, help="subset, e.g. none,gamma,specular")
    ap.add_argument("--motions", default=None)
    ap.add_argument("--defects", default=None, help="subset, e.g. none,tape")
    ap.add_argument("--timing-pairs", type=int, default=12, help="pairs in the sequential timing pass")
    ap.add_argument("--visa", default=None, help="path to an extracted VisA root (optional)")
    ap.add_argument("--visa-categories", default=",".join(VISA_FIXED_POSE))
    ap.add_argument("--visa-max", type=int, default=0, help="max test images per class and category (0 = all)")
    ap.add_argument("--visa-note", default=None, help="text for RESULTS.md when VisA is not run")
    args = ap.parse_args(argv)

    methods = _parse_list(args.methods, METHODS)
    size = tuple(int(v) for v in args.size.lower().split("x"))
    grid_def = {"parts": _parse_list(args.parts, synth.PART_KINDS),
                "lightings": _parse_list(args.lightings, LIGHTINGS),
                "motions": _parse_list(args.motions, MOTIONS),
                "defects": _parse_list(args.defects, DEFECTS)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    command = "python -m eval.benchmark " + " ".join(argv if argv is not None else sys.argv[1:])
    base = Params()
    env = environment(args.workers)
    now = lambda: dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")  # noqa: E731

    calib_path = out / "calib.json"
    if args.split in ("report", "timing"):
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        test = json.loads((out / "test_summary.json").read_text(encoding="utf-8"))
        if "paired" not in test:
            pairs = json.loads((out / "test_pairs.json").read_text(encoding="utf-8"))
            test["paired"] = paired_comparisons(pairs, test["methods"])
            _dump(out / "test_summary.json", test)
        if args.split == "timing":
            P = dataclasses.replace(base, **{method_param(m): calib["thresholds"][m] for m in test["methods"]})
            grid = {k: [None if v == "none" else v for v in vals] for k, vals in test["grid"].items()}
            cases = build_cases("test", test["seed"], test["n"], **grid)
            test["timing"] = timing_pass(cases, tuple(test["size"]), test["methods"], P, args.timing_pairs)
            test["timing_date"], test["timing_command"] = now(), command
            _dump(out / "test_summary.json", test)
        visa_path = out / "visa.json"
        visa = json.loads(visa_path.read_text(encoding="utf-8")) if visa_path.exists() else None
        write_report(Path(args.report), calib, test, visa, args.visa_note or test.get("visa_note"))
        print(f"report: {args.report}")
        return 0
    if args.split in ("calib", "all"):
        grids = {m: threshold_grid([getattr(base, method_param(m))]) for m in methods}
        cases = build_cases("calib", args.seed, args.n, **grid_def)
        t0 = time.perf_counter()
        recs, errors = run_grid(cases, size, methods, base, grids, args.workers, "calib")
        cal = calibrate(recs, methods, grids, FALSE_EVENT_TARGET)
        calib = {"command": command, "date": now(), "env": env, "seed": args.seed, "n": args.n,
                 "size": size, "grid": {k: [v or "none" for v in vals] for k, vals in grid_def.items()},
                 "methods": methods, "target": FALSE_EVENT_TARGET, "n_pairs": len(recs), "errors": errors,
                 "wall_s": time.perf_counter() - t0, "params": base.to_dict(),
                 "thresholds": {m: cal[m]["threshold"] for m in methods}, "calibration": cal,
                 "consistency_failures": sum(not mm["consistent"] for r in recs for mm in r["methods"].values())}
        _dump(calib_path, calib)
        _dump(out / "calib_sweep.json", sweep_curves(recs, methods, grids))
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
        cases = build_cases("test", args.seed, args.n, **grid_def)
        t0 = time.perf_counter()
        recs, errors = run_grid(cases, size, methods, P, grids, args.workers, "test")
        wall = time.perf_counter() - t0
        subs = subsets(recs)
        clean = subs["none+monotone"]
        test = {
            "command": command, "date": now(), "env": env, "seed": args.seed, "n": args.n, "size": size,
            "grid": calib["grid"] if args.split == "all" else
            {k: [v or "none" for v in vals] for k, vals in grid_def.items()},
            "methods": methods, "thresholds": calib["thresholds"], "params": P.to_dict(), "wall_s": wall,
            "summary": {k: {m: summarize(v, m) for m in methods} for k, v in subs.items()},
            "per_defect": {d: {m: summarize([r for r in clean if r["defect"] == d], m) for m in methods}
                           for d in BENCH_DEFECTS if any(r["defect"] == d for r in clean)},
            "clean_by": {k: {m: summarize([r for r in clean if r[k.split(':')[0]] == k.split(':')[1]], m)
                             for m in methods}
                         for k in [f"motion:{x}" for x in MOTIONS] + [f"part:{x}" for x in synth.PART_KINDS]
                         if any(r[k.split(':')[0]] == k.split(':')[1] for r in clean)},
            "auroc": {g: {m: separability(subs.get(g, []), m) for m in methods}
                      for g in ("none+monotone", "non_monotone", "all")},
            "consistency_failures": sum(not mm["consistent"] for r in recs for mm in r["methods"].values()),
            "visa_note": args.visa_note, "errors": errors,
            "paired": paired_comparisons(_pair_rows(recs), methods),
        }
        print("timing pass ...", flush=True)
        test["timing"] = timing_pass(cases, size, methods, P, args.timing_pairs)
        visa = None
        if args.visa:
            root = Path(args.visa)
            if not root.exists():
                raise SystemExit(f"VisA root not found: {root}")
            cats = [c.strip() for c in args.visa_categories.split(",") if c.strip()]
            visa = run_visa(root, cats, 3, args.visa_max, P, methods, args.workers, args.seed)
            visa["command"] = command
            _dump(out / "visa.json", {k: v for k, v in visa.items()})
        _dump(out / "test_summary.json", test)
        _dump(out / "test_sweep.json", sweep_curves(recs, methods, grids))
        _dump(out / "test_pairs.json", _pair_rows(recs), compact=True)
        write_report(Path(args.report), calib, test, visa, args.visa_note)
        s = test["summary"]
        for k in ("none+monotone", "non_monotone"):
            if k in s:
                print(k, {m: (round(s[k][m]["recall"], 3), round(s[k][m]["fa_rate"], 3)) for m in methods})
        print(f"report: {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
