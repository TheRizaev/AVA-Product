"""Regenerate the built-in demo examples in web/examples/ (docs/SPEC.md "Demo scenarios").

    python -m eval.make_examples            # search seeds, print scores, write the 5 JPEGs
    python -m eval.make_examples --dry-run  # search + print only

Five 1280x720 JPEGs (quality 92), all SYNTHETIC (procedural, dalil.synth; no faces, no plates):

    surface_defect.jpg     Mode 1: uniform surface WITH a clear defect (bright scratch + paint chip)
    surface_ok.jpg         Mode 1: the SAME surface without the defect
    compare_reference.jpg  Mode 2: golden part (feature-rich stamped / machined part on a belt)
    compare_defect.jpg     Mode 2: the same part with masking tape stuck on it (camera jitter)
    compare_light.jpg      Mode 2: the same part, NO defect, strong global lighting change
                           (exposure x0.65 + gamma 1.3: mid-grey drops ~47 %, warm cast) + camera jitter

Seeds are searched deterministically over a small fixed list; a candidate is accepted only if
the real pipeline gives the expected verdicts with margins well away from the thresholds (see
MARGIN below), so the examples keep working when thresholds are re-calibrated a little. The
first accepted candidate (in list order) is written; its seeds and scores are printed.
Nothing here decides a verdict for the UI: the site only loads these images and calls the API.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from dalil import pipeline, surface, synth
from dalil.config import Params
from dalil.pbp import cell_mask_from_pixel_mask

OUT_DIR = Path(__file__).resolve().parent.parent / "web" / "examples"
H, W = 720, 1280
JPEG_QUALITY = 92

# Search space (order = preference).
SURFACE_KINDS = ("painted", "glass_dark")
SURFACE_SEEDS = tuple(range(12))
COMPARE_KINDS = ("stamped", "machined")
COMPARE_SEEDS = tuple(range(12))

# Robustness: the decisive score must clear the threshold by this factor.
MARGIN = 1.35         # defect: max score near the defect >= MARGIN * threshold
CLEAN_MARGIN = 0.75   # no defect: max score anywhere <= CLEAN_MARGIN * threshold
FAR_PX = 40           # "elsewhere" = farther than this from the defect (work px; smoothing spreads it)
PIXDIFF_MIN_CHANGED = 0.30  # brightness case: plain subtraction flags >= 30 % of the checked area

# Strong, obvious global lighting change for compare_light (monotone per channel).
LIGHT_EXPOSURE = 0.65
LIGHT_GAMMA = 1.3
LIGHT_CAST_BGR = (0.84, 0.97, 1.12)   # warm: blue down, red up


# --------------------------------------------------------------------------- helpers

def jpeg_roundtrip(img: np.ndarray) -> tuple[bytes, np.ndarray]:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    assert ok
    data = buf.tobytes()
    return data, cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def strong_light(img: np.ndarray) -> np.ndarray:
    """Gamma 1.3, exposure x0.65 and a warm colour cast: monotone in every channel."""
    x = img.astype(np.float32) / 255.0
    x = x ** np.float32(LIGHT_GAMMA) * np.float32(LIGHT_EXPOSURE)
    x = x * np.asarray(LIGHT_CAST_BGR, np.float32)
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def _mask_at(mask: np.ndarray, work_size, dilate_px: int = 8) -> np.ndarray:
    m = cv2.resize(mask.astype(np.uint8), tuple(work_size), interpolation=cv2.INTER_NEAREST)
    m = cv2.dilate(m, np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8))
    return m > 0


def _cells_of(px_mask: np.ndarray, params: Params) -> np.ndarray:
    """Grid cells whose window touches the pixel mask."""
    return cell_mask_from_pixel_mask(px_mask, params.win, params.step, frac=1.0 / (params.win ** 2))


def max_score(res, region: np.ndarray | None = None, params: Params | None = None) -> float:
    """Max of the result's smoothed score map over checked cells (optionally near `region`)."""
    if res.score is None:
        return float("nan")
    sel = res.checked.copy()
    if region is not None:
        sel &= _cells_of(region, params)
    return float(res.score[sel].max()) if sel.any() else 0.0


def zone_hits(res, mask_work: np.ndarray) -> bool:
    """Some zone lies on the defect and every zone does (no false zones)."""
    hits = [bool(mask_work[y:y + h, x:x + w].any()) for x, y, w, h in (z.bbox for z in res.zones)]
    return bool(hits) and all(hits)


# --------------------------------------------------------------------------- Mode 1

def make_surface_case(kind: str, seed: int):
    """(ok, defect, mask) for one surface seed: a long bright scratch ending in a paint chip."""
    ok = synth.make_uniform_surface(np.random.default_rng(seed), kind, H, W)
    region = np.zeros((H, W), bool)
    region[int(0.3 * H):int(0.7 * H), int(0.3 * W):int(0.7 * W)] = True
    # Defect sub-seed: first one giving a long (>= 150 px), clearly visible scratch.
    for sub in range(64):
        rng = np.random.default_rng([seed, 7, sub])
        img, m1 = synth.add_scratch(ok, rng, region=region)
        ys, xs = np.nonzero(m1)
        extent = math.hypot(np.ptp(xs), np.ptp(ys)) if xs.size else 0.0
        delta = float(np.abs(img.astype(np.float32) - ok.astype(np.float32)).max(-1)[m1].mean())
        if extent >= 150 and delta >= 70 and m1.sum() >= 2.0 * extent:
            break
    # A paint chip right next to the scratch (an impact that also chipped the paint).
    cy, cx = int(np.median(ys)), int(np.median(xs))
    near = np.zeros((H, W), bool)
    near[max(0, cy - 40):cy + 40, max(0, cx - 40):cx + 40] = True
    near &= ~cv2.dilate(m1.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    img, m2 = synth.add_chip(img, np.random.default_rng([seed, 8, sub]), region=near)
    return ok, img, m1 | m2, {"defect_subseed": sub, "scratch_extent_px": round(extent),
                              "scratch_delta": round(delta)}


def eval_surface(kind: str, seed: int, params: Params):
    ok, bad, mask, info = make_surface_case(kind, seed)
    ok_bytes, ok_dec = jpeg_roundtrip(ok)
    bad_bytes, bad_dec = jpeg_roundtrip(bad)
    r_ok = surface.scan_surface(ok_dec, params)
    r_bad = surface.scan_surface(bad_dec, params)
    thr = r_ok.threshold
    mw = _mask_at(mask, r_bad.work_size)
    s_ok = max_score(r_ok)
    s_bad = max_score(r_bad, mw, params)
    s_bad_off = max_score(r_bad, ~_mask_at(mask, r_bad.work_size, FAR_PX), params)
    good = (r_ok.verdict == "pass" and r_bad.verdict == "fail" and zone_hits(r_bad, mw)
            and s_ok <= CLEAN_MARGIN * thr and s_bad >= MARGIN * thr and s_bad_off <= CLEAN_MARGIN * thr)
    row = {"kind": kind, "seed": seed, **info, "threshold": round(thr, 4),
           "ok_verdict": r_ok.verdict, "ok_max": round(s_ok, 4),
           "defect_verdict": r_bad.verdict, "defect_zones": len(r_bad.zones),
           "defect_max_at_defect": round(s_bad, 4), "defect_max_elsewhere": round(s_bad_off, 4),
           "ms": r_bad.timing_ms.get("total")}
    return good, row, {"surface_ok.jpg": ok_bytes, "surface_defect.jpg": bad_bytes}


# --------------------------------------------------------------------------- Mode 2

def make_compare_case(kind: str, seed: int):
    """(reference, defect frame, light frame, mask) of one part seed."""
    pd = synth.make_station_pair(np.random.default_rng(seed), part_kind=kind, defect="tape",
                                 motion="jitter", h=H, w=W)
    pl = synth.make_station_pair(np.random.default_rng(seed), part_kind=kind, defect=None,
                                 motion="jitter", h=H, w=W)
    # Same scene: the part is rendered before any defect / motion randomness is drawn.
    assert np.array_equal(pd["meta"]["reference_clean"], pl["meta"]["reference_clean"])
    return pd["reference"], pd["frame"], strong_light(pl["frame"]), pd["mask_ref_frame"]


def eval_compare(kind: str, seed: int, params: Params):
    ref, bad, light, mask = make_compare_case(kind, seed)
    ref_bytes, ref_dec = jpeg_roundtrip(ref)
    bad_bytes, bad_dec = jpeg_roundtrip(bad)
    light_bytes, light_dec = jpeg_roundtrip(light)
    # Visible tape: masking tape (opaque paper), not near-invisible clear tape.
    tape_delta = float(np.abs(bad.astype(np.float32) - ref.astype(np.float32)).max(-1)[mask].mean())

    reference = pipeline.prepare_reference([ref_dec], params, ("pbp", "pixdiff"))
    rb = pipeline.inspect_compare(reference, bad_dec, params, ("pbp", "pixdiff"))
    rl = pipeline.inspect_compare(reference, light_dec, params, ("pbp", "pixdiff"))
    thr = rb["pbp"].threshold
    mw = _mask_at(mask, rb["pbp"].work_size)
    s_bad = max_score(rb["pbp"], mw, params)
    s_bad_off = max_score(rb["pbp"], ~_mask_at(mask, rb["pbp"].work_size, FAR_PX), params)
    s_light = max_score(rl["pbp"])
    px = rl["pixdiff"]
    good = (tape_delta >= 40
            and rb["pbp"].verdict == "fail" and zone_hits(rb["pbp"], mw)
            and s_bad >= MARGIN * thr and s_bad_off <= CLEAN_MARGIN * thr
            and rl["pbp"].verdict == "pass" and s_light <= CLEAN_MARGIN * thr
            and px.verdict != "pass" and px.changed_frac >= PIXDIFF_MIN_CHANGED)
    row = {"kind": kind, "seed": seed, "tape_delta": round(tape_delta), "threshold": round(thr, 4),
           "defect_pbp": rb["pbp"].verdict, "defect_zones": len(rb["pbp"].zones),
           "defect_max_at_defect": round(s_bad, 4), "defect_max_elsewhere": round(s_bad_off, 4),
           "light_pbp": rl["pbp"].verdict, "light_pbp_max": round(s_light, 4),
           "light_pixdiff": f"{px.verdict}/{px.reason}", "light_pixdiff_changed": round(px.changed_frac, 3),
           "light_mean_gray": round(float(cv2.cvtColor(light, cv2.COLOR_BGR2GRAY).mean()), 1),
           "align": rl["pbp"].align.get("method"), "ms": rl["pbp"].timing_ms.get("total")}
    files = {"compare_reference.jpg": ref_bytes, "compare_defect.jpg": bad_bytes,
             "compare_light.jpg": light_bytes}
    return good, row, files


# --------------------------------------------------------------------------- main

def search(evaluate, kinds, seeds, params: Params, label: str):
    for kind in kinds:
        for seed in seeds:
            good, row, files = evaluate(kind, seed, params)
            print(f"[{label}] {'ACCEPT' if good else 'reject'} {row}", flush=True)
            if good:
                return row, files
    raise SystemExit(f"no robust {label} example found; widen the seed list")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="search and print only, write nothing")
    args = ap.parse_args(argv)
    params = Params()
    s_row, s_files = search(eval_surface, SURFACE_KINDS, SURFACE_SEEDS, params, "surface")
    c_row, c_files = search(eval_compare, COMPARE_KINDS, COMPARE_SEEDS, params, "compare")
    print(f"chosen surface: kind={s_row['kind']} seed={s_row['seed']}")
    print(f"chosen compare: kind={c_row['kind']} seed={c_row['seed']}")
    if args.dry_run:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in {**s_files, **c_files}.items():
        (OUT_DIR / name).write_bytes(data)
        print(f"wrote {OUT_DIR / name} ({len(data) // 1024} KB)")


if __name__ == "__main__":
    main()
