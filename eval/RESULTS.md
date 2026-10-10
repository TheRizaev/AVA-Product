# Synthetic benchmark — procedurally generated parts, NOT real factory images

Every number in this file was produced by `eval/benchmark.py` (raw data in `eval/results/`). The pairs come from `dalil/synth.py`: procedural stamped / painted / molded / machined parts on belt / table / ESD-mat backgrounds, procedural defects and procedural shop-floor lighting. They show how the methods compare under controlled conditions; they are **not** evidence of accuracy on real parts.

## Run

```
python -m eval.benchmark --n 4 --seed 0 --workers 8 --visa-note A download of VisA_20220922.tar (about 1.8 GB) was attempted on 2026-10-10; measured speed was 40-50 KB/s (about 10 hours), so it was abandoned.
python -m eval.benchmark --split test --n 4 --seed 0 --workers 8 --visa-note A download of VisA_20220922.tar (about 1.8 GB) was attempted on 2026-10-10; measured speed was 40-50 KB/s (about 10 hours), so it was abandoned.
```

|  |  |
|---|---|
| Date (calib / test) | 2026-10-10 17:15 UTC / 2026-10-10 17:47 UTC |
| CPU | 13th Gen Intel(R) Core(TM) i5-13450HX (16 logical cores) |
| Software | Windows-10-10.0.26200-SP0; Python 3.11.9, numpy 2.2.6, OpenCV 4.12.0 |
| Code | 26c853e + uncommitted changes in dalil/eval/server |
| Grid | 4 parts x 8 lightings x 2 motions x 7 defect classes x 4 = 1792 TEST pairs (1792 CALIB pairs) |
| Image size | 1280x720 px generated, pipeline working size max side 1024 px; 1 reference shot per pair |
| Seeds | --seed 0; per pair SeedSequence(seed, split code calib=1/test=2, cell, repeat) |
| Wall time | calib 940 s, test 1131 s with 8 worker processes (OpenCV 1 thread each) |
| Pairs that raised an exception | calib 0, test 0 (excluded from every metric; listed in the JSON) |
| Sweep vs pipeline consistency | 0 + 0 mismatches between the re-thresholded map and the pipeline's own verdict at the operating point |

## Protocol

1. **Pairs.** `synth.make_station_pair` gives a (reference, frame) pair: the frame may carry a defect
   (scratch, tape, missing_hole, burr, dent, stain; `sticker` is not used), is moved (`jitter` = camera vibration ±2 px / 0.3°,
   `shift` = the part moved on the belt by up to 40 px / 3° with the camera fixed), and is relit with one
   shop-floor lighting kind or none. Both images get independent sensor noise.
2. **Same pipeline for every method.** `pipeline.inspect_compare` aligns once, builds one glare/validity mask
   and scores PBP, Census, SSIM and plain reference subtraction (`pixdiff`) on the same aligned frame, with the
   same tolerance, smoothing and zone rules. Only the window features and distances differ (PBP additionally
   has its low-contrast gate).
3. **Calibration (CALIB split only).** Each method's smoothed change map is re-thresholded with
   `pipeline.decide` over a geometric grid (0.002–0.95). For each method we take the smallest grid threshold
   at which the mean number of *false events* per defect-free pair under **none + monotone** lighting is
   ≤ 0.05. A false event is a zone that does not touch the defect (dilated by one window), or a
   `mismatch` retake of a good part (so the degenerate "reject everything as mismatch" thresholds do not
   qualify). The thresholds are then frozen.
4. **Test (TEST split, disjoint seeds).** All metrics below. *Recall*: a defect pair is detected if the
   verdict is FAIL and some zone has IoU > 0.1 with the true defect bbox or covers ≥ 30 % of it.
   *Escapes*: defect pairs judged PASS. *False-alarm rate*: fraction of defect-free pairs judged FAIL;
   *good parts not passed* also counts RETAKE answers on defect-free pairs. *False zones*: zones not touching
   the defect, per pair (all pairs). *Retake rate*: fraction of pairs answered RETAKE (alignment, glare, blur, low coverage, mismatch).
   *Coverage*: mean fraction of cells checked, over scored pairs. Brackets: Wilson 95 % intervals.
5. **Timing.** A separate sequential pass in one process (OpenCV threads on), one method per inspection,
   24 TEST pairs after one warm-up pair; times include resize, quality
   checks, alignment, masks, features, change map, zones and the overlay JPEGs.

## Calibrated thresholds (frozen before TEST)

| Method | Param | Threshold | CALIB false events / good pair | CALIB false-alarm rate | CALIB good pairs (none+monotone) |
|---|---|---|---|---|---|
| pbp | `threshold` | 0.6098 | 0.042 | 2.1% | 192 |
| census | `threshold_census` | 0.3914 | 0.021 | 1.6% | 192 |
| ssim | `threshold_ssim` | 0.135 | 0.047 | 2.6% | 192 |
| pixdiff | `threshold_pixdiff` | 0.2512 | 0.042 | 2.1% | 192 |

## TEST results

### Lighting: none + monotone (the operating domain) — 1152 defect pairs, 192 defect-free pairs

| Method | Defect recall | Escapes (defect → PASS) | False-alarm rate (good → FAIL) | Good parts not passed (FAIL or RETAKE) | False zones / pair | Retake rate (all) | Coverage | Mean ms / inspection |
|---|---|---|---|---|---|---|---|---|
| pbp | 62.4% (59.6–65.2) | 29.4% (26.9–32.1) | 1.0% (0.3–3.7) | 2.1% (0.8–5.2) | 0.032 | 1.3% (0.8–2.1) | 94.3% | 550 |
| census | 27.4% (24.9–30.1) | 61.6% (58.8–64.4) | 0.5% (0.1–2.9) | 1.6% (0.5–4.5) | 0.025 | 1.3% (0.8–2.1) | 94.3% | 252 |
| ssim | 65.6% (62.8–68.3) | 28.0% (25.4–30.6) | 1.0% (0.3–3.7) | 2.1% (0.8–5.2) | 0.033 | 1.3% (0.8–2.1) | 94.3% | 195 |
| pixdiff | 14.8% (12.8–16.9) | 78.8% (76.4–81.1) | 2.1% (0.8–5.2) | 3.1% (1.4–6.6) | 0.054 | 2.5% (1.8–3.5) | 94.3% | 219 |

### Lighting: non-monotone (specular highlight, hand shadow) — expected failure — 384 defect pairs, 64 defect-free pairs

| Method | Defect recall | Escapes (defect → PASS) | False-alarm rate (good → FAIL) | Good parts not passed (FAIL or RETAKE) | False zones / pair | Retake rate (all) | Coverage | Mean ms / inspection |
|---|---|---|---|---|---|---|---|---|
| pbp | 65.6% (60.7–70.2) | 7.6% (5.3–10.6) | 93.8% (85.0–97.5) | 96.9% (89.3–99.1) | 3.846 | 2.5% (1.4–4.3) | 92.1% | 550 |
| census | 38.8% (34.1–43.8) | 12.5% (9.6–16.2) | 85.9% (75.4–92.4) | 89.1% (79.1–94.6) | 4.355 | 2.2% (1.2–4.1) | 92.1% | 252 |
| ssim | 63.0% (58.1–67.7) | 10.7% (8.0–14.2) | 84.4% (73.6–91.3) | 90.6% (81.0–95.6) | 1.478 | 6.9% (4.9–9.7) | 92.1% | 195 |
| pixdiff | 22.9% (19.0–27.4) | 34.1% (29.6–39.0) | 62.5% (50.3–73.3) | 65.6% (53.4–76.1) | 0.730 | 3.8% (2.4–6.0) | 92.1% | 219 |

### All lighting conditions — 1536 defect pairs, 256 defect-free pairs

| Method | Defect recall | Escapes (defect → PASS) | False-alarm rate (good → FAIL) | Good parts not passed (FAIL or RETAKE) | False zones / pair | Retake rate (all) | Coverage | Mean ms / inspection |
|---|---|---|---|---|---|---|---|---|
| pbp | 63.2% (60.8–65.6) | 24.0% (21.9–26.2) | 24.2% (19.4–29.8) | 25.8% (20.8–31.5) | 0.985 | 1.6% (1.1–2.3) | 93.8% | 550 |
| census | 30.3% (28.0–32.6) | 49.3% (46.9–51.8) | 21.9% (17.2–27.3) | 23.4% (18.7–29.0) | 1.108 | 1.6% (1.1–2.2) | 93.8% | 252 |
| ssim | 65.0% (62.6–67.3) | 23.6% (21.6–25.8) | 21.9% (17.2–27.3) | 24.2% (19.4–29.8) | 0.395 | 2.7% (2.1–3.6) | 93.8% | 195 |
| pixdiff | 16.8% (15.0–18.7) | 67.6% (65.3–69.9) | 17.2% (13.1–22.3) | 18.8% (14.4–24.0) | 0.223 | 2.8% (2.2–3.7) | 93.8% | 219 |

### By lighting kind — defect recall / false-alarm rate

| Lighting | Group | pbp | census | ssim | pixdiff |
|---|---|---|---|---|---|
| none | — | 62.0% (54.9–68.5) / 0.0% (0.0–10.7) | 30.7% (24.6–37.6) / 0.0% (0.0–10.7) | 65.1% (58.1–71.5) / 0.0% (0.0–10.7) | 10.9% (7.3–16.1) / 0.0% (0.0–10.7) |
| drift_gain | monotone | 60.9% (53.9–67.6) / 0.0% (0.0–10.7) | 28.6% (22.7–35.4) / 0.0% (0.0–10.7) | 66.1% (59.2–72.5) / 0.0% (0.0–10.7) | 19.3% (14.3–25.4) / 0.0% (0.0–10.7) |
| gamma | monotone | 62.0% (54.9–68.5) / 3.1% (0.6–15.7) | 25.0% (19.4–31.6) / 0.0% (0.0–10.7) | 65.6% (58.7–72.0) / 3.1% (0.6–15.7) | 15.6% (11.2–21.4) / 0.0% (0.0–10.7) |
| window_gradient | monotone | 67.7% (60.8–73.9) / 0.0% (0.0–10.7) | 29.2% (23.2–36.0) / 0.0% (0.0–10.7) | 66.7% (59.7–73.0) / 0.0% (0.0–10.7) | 14.6% (10.3–20.3) / 0.0% (0.0–10.7) |
| lamp_aging | monotone | 67.7% (60.8–73.9) / 3.1% (0.6–15.7) | 28.6% (22.7–35.4) / 3.1% (0.6–15.7) | 70.8% (64.0–76.8) / 3.1% (0.6–15.7) | 14.6% (10.3–20.3) / 0.0% (0.0–10.7) |
| auto_exposure | monotone | 54.2% (47.1–61.1) / 0.0% (0.0–10.7) | 22.4% (17.1–28.8) / 0.0% (0.0–10.7) | 59.4% (52.3–66.1) / 0.0% (0.0–10.7) | 13.5% (9.4–19.1) / 12.5% (5.0–28.1) |
| **monotone (all)** |  | 62.5% (59.4–65.5) / 1.2% (0.3–4.4) | 26.8% (24.1–29.7) / 0.6% (0.1–3.5) | 65.7% (62.7–68.7) / 1.2% (0.3–4.4) | 15.5% (13.4–17.9) / 2.5% (1.0–6.3) |
| specular | non-monotone | 62.0% (54.9–68.5) / 90.6% (75.8–96.8) | 34.9% (28.5–41.9) / 90.6% (75.8–96.8) | 62.5% (55.5–69.0) / 78.1% (61.2–89.0) | 15.6% (11.2–21.4) / 65.6% (48.3–79.6) |
| hand_shadow | non-monotone | 69.3% (62.4–75.4) / 96.9% (84.3–99.4) | 42.7% (35.9–49.8) / 81.2% (64.7–91.1) | 63.5% (56.5–70.0) / 90.6% (75.8–96.8) | 30.2% (24.2–37.0) / 59.4% (42.3–74.5) |
| **non-monotone (all)** |  | 65.6% (60.7–70.2) / 93.8% (85.0–97.5) | 38.8% (34.1–43.8) / 85.9% (75.4–92.4) | 63.0% (58.1–67.7) / 84.4% (73.6–91.3) | 22.9% (19.0–27.4) / 62.5% (50.3–73.3) |

### Per-defect recall (none + monotone lighting)

| Defect | pbp | census | ssim | pixdiff |
|---|---|---|---|---|
| scratch | 60.9% (53.9–67.6) | 3.6% (1.8–7.3) | 56.8% (49.7–63.6) | 1.0% (0.3–3.7) |
| tape | 70.8% (64.0–76.8) | 32.8% (26.6–39.7) | 70.3% (63.5–76.3) | 27.1% (21.3–33.8) |
| missing_hole | 72.4% (65.7–78.2) | 54.2% (47.1–61.1) | 72.9% (66.2–78.7) | 24.5% (18.9–31.0) |
| burr | 19.3% (14.3–25.4) | 4.2% (2.1–8.0) | 42.2% (35.4–49.3) | 1.6% (0.5–4.5) |
| dent | 81.2% (75.1–86.1) | 56.2% (49.2–63.1) | 78.1% (71.8–83.4) | 10.4% (6.8–15.5) |
| stain | 69.8% (63.0–75.8) | 13.5% (9.4–19.1) | 73.4% (66.8–79.2) | 24.0% (18.5–30.5) |

### By motion and part kind (none + monotone lighting) — recall / false-alarm rate

| Subset | pbp | census | ssim | pixdiff |
|---|---|---|---|---|
| motion:jitter | 83.2% / 0.0% | 45.7% / 0.0% | 87.7% / 0.0% | 21.5% / 4.2% |
| motion:shift | 41.7% / 2.1% | 9.2% / 1.0% | 43.6% / 2.1% | 8.0% / 0.0% |
| part:stamped | 62.5% / 0.0% | 20.8% / 0.0% | 66.7% / 0.0% | 18.8% / 0.0% |
| part:painted | 60.1% / 0.0% | 20.8% / 0.0% | 60.8% / 0.0% | 10.8% / 0.0% |
| part:molded | 68.8% / 2.1% | 31.9% / 2.1% | 71.5% / 2.1% | 11.1% / 8.3% |
| part:machined | 58.3% / 2.1% | 36.1% / 0.0% | 63.5% / 2.1% | 18.4% / 0.0% |

### Retake reasons (all TEST pairs)

| Method | Reasons |
|---|---|
| pbp | glare: 20, blurry: 8, mismatch: 1 |
| census | glare: 20, blurry: 8 |
| ssim | mismatch: 21, glare: 20, blurry: 8 |
| pixdiff | mismatch: 23, glare: 20, blurry: 8 |

### Threshold-free separability (AUROC of the max smoothed cell score, defect vs defect-free)

| Method | none + monotone | non-monotone | all |
|---|---|---|---|
| pbp | 0.931 | 0.490 | 0.789 |
| census | 0.894 | 0.462 | 0.741 |
| ssim | 0.927 | 0.576 | 0.812 |
| pixdiff | 0.689 | 0.529 | 0.628 |

Pairs rejected before scoring (alignment / glare / blur / coverage) have no score and are left out of the AUROC. The full recall-vs-false-alarm curves are in `eval/results/test_sweep.json` (and `calib_sweep.json`).

### Paired comparison: PBP vs each method on the same TEST pairs

| Lighting | vs | Recall PBP vs it | Defects found only by PBP / only by it | McNemar p | Recall | False alarms PBP vs it | False alarms only PBP / only it | McNemar p | False alarms |
|---|---|---|---|---|---|---|---|---|---|
| none+monotone | census | 62.4% vs 27.4% | 407 / 4 | 4.5e-115 | PBP better | 1.0% vs 0.5% | 1 / 0 | 1 | no significant difference |
| none+monotone | ssim | 62.4% vs 65.6% | 58 / 95 | 0.0035 | ssim better | 1.0% vs 1.0% | 0 / 0 | 1 | no significant difference |
| none+monotone | pixdiff | 62.4% vs 14.8% | 562 / 13 | 1.7e-147 | PBP better | 1.0% vs 2.1% | 2 / 4 | 0.69 | no significant difference |
| non_monotone | census | 65.6% vs 38.8% | 109 / 6 | 1.4e-25 | PBP better | 93.8% vs 85.9% | 5 / 0 | 0.062 | no significant difference |
| non_monotone | ssim | 65.6% vs 63.0% | 23 / 13 | 0.13 | no significant difference | 93.8% vs 84.4% | 6 / 0 | 0.031 | ssim better |
| non_monotone | pixdiff | 65.6% vs 22.9% | 169 / 5 | 1.1e-43 | PBP better | 93.8% vs 62.5% | 20 / 0 | 1.9e-06 | pixdiff better |

Exact two-sided McNemar tests on the discordant pairs (each pair is seen by both methods), at the frozen thresholds; p-values are not corrected for the number of comparisons.

## Speed (sequential pass, this machine)

| Method | Mean ms | Median ms | Min–max ms | Inspections |
|---|---|---|---|---|
| pbp | 550 | 526 | 105–819 | 24 |
| census | 252 | 275 | 110–332 | 24 |
| ssim | 195 | 209 | 105–255 | 24 |
| pixdiff | 219 | 235 | 109–283 | 24 |

PBP stage medians (ms): resize 1, quality 4, align 51, mask 33, features 92, change 167, zones 1, render 78, total 534, extra_views 258. `prepare_reference` (1 shot, PBP): median 119 ms. Means include inspections that stopped early at a RETAKE gate; stage medians are over the inspections that reached the stage (`extra_views`: moved part only).

CPU: 13th Gen Intel(R) Core(TM) i5-13450HX. Frames 1280x720 resized to max side 1024 px. Measured 2026-10-10 18:03 UTC with `python -m eval.benchmark --split timing --timing-pairs 24`. Other processes on the machine were not controlled; treat as indicative.

## Caveats (read before quoting any number)

- **Synthetic data only.** Parts, defects and lighting are procedural (`dalil/synth.py`). Real scratches, dents
  and stains on real parts look different, and real cameras add compression, rolling shutter, motion blur and
  defocus that are only partly modelled. These numbers compare the methods under identical controlled
  conditions; they are not a real-world accuracy claim.
- **Non-monotone lighting is an expected failure.** A specular highlight or a hand shadow changes the local
  intensity order, which is exactly what PBP (and Census) rely on. Under those conditions good parts are
  judged FAIL by every method (see the non-monotone table); only a part of it is turned into RETAKE by the
  glare mask.
- **Census is PBP's closest relative** (both compare the intensity *order*, both ignore monotone changes). In
  this implementation Census compares single pixels with their 8 neighbours with no tie margin and no
  contrast gate, so sensor noise flips bits on smooth surfaces and its calibrated threshold has to be high.
  That is a likely reason for its low recall here, not a tested one; a Census variant with a noise margin
  was not evaluated.
- **SSIM** normalises every window by its own mean and variance, so it is also largely insensitive to a
  gain change. Whether PBP differs from it on these pairs is in the paired comparison above; the large,
  consistent gap is to plain reference subtraction (pixdiff), the classic golden-template approach.
- **Part moved on the belt (`shift`) is harder than camera jitter** for every method (see "By motion"). With
  a moved part the pipeline checks only the hull of the part's keypoints and takes, per cell, the minimum
  over the part-registered and background-registered views, which also lowers defect scores near shifted
  content.
- **Small N.** 4 pair(s) per grid cell; per-cell and per-class rates rest on few pairs, see the Wilson
  95 % intervals. Differences inside overlapping intervals are not evidence.
- **One calibration criterion for all methods** (false events ≤ 0.05 per good pair under none + monotone
  lighting). A different operating point changes the trade-off; the full sweeps are in `eval/results/`.
- **The ground truth is a bbox.** Zones live on a 4 px window grid; recall uses the bbox IoU / coverage rule
  above, false zones use "does not touch the (dilated) defect mask".
- Defect classes are not equally hard: burrs are small blobs on an edge, and a bright burr under strong
  exposure can be masked as glare (unchecked) — see the per-defect table.

## Real data

VisA (Amazon, CC BY 4.0) evaluation was **not run** in this benchmark (`--visa` not given). A download of VisA_20220922.tar (about 1.8 GB) was attempted on 2026-10-10; measured speed was 40-50 KB/s (about 10 hours), so it was abandoned. Real-data evaluation is **pending**: the team photo set in `data/pairs/` (protocol in `data/pairs/README.md`). No real-data metric is claimed here.
