# Synthetic benchmark — procedurally generated parts, NOT real factory images

Every number in this file was produced by `eval/benchmark.py` (raw data in `eval/results/`). The pairs come from `ava/synth.py`: procedural stamped / painted / molded / machined parts on belt / table / ESD-mat backgrounds, procedural defects and procedural shop-floor lighting. They show how the methods compare under controlled conditions; they are **not** evidence of accuracy on real parts.

## Run

```
python -m eval.benchmark --n 4 --seed 0 --workers 8 --visa-note A download of VisA_20220922.tar (about 1.8 GB) was attempted on 2026-10-10; measured speed was 40-50 KB/s (about 10 hours), so it was abandoned.
```

|  |  |
|---|---|
| Date (calib / test) | 2026-10-10 19:40 UTC / 2026-10-10 20:11 UTC |
| CPU | 13th Gen Intel(R) Core(TM) i5-13450HX (16 logical cores) |
| Software | Windows-10-10.0.26200-SP0; Python 3.11.9, numpy 2.2.6, OpenCV 4.12.0 |
| Code | 043d4ab + uncommitted changes in ava/eval/server |
| Grid | 4 parts x 8 lightings x 2 motions x 7 defect classes x 4 = 1792 TEST pairs (1792 CALIB pairs) |
| Image size | 1280x720 px generated, pipeline working size max side 1024 px; 1 reference shot per pair |
| Seeds | --seed 0; per pair SeedSequence(seed, split code calib=1/test=2, cell, repeat) |
| Wall time | calib 1288 s, test 1854 s with 8 worker processes (OpenCV 1 thread each) |
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
| pbp | `threshold` | 0.5833 | 0.047 | 3.1% | 192 |
| census | `threshold_census` | 0.4091 | 0.026 | 1.6% | 192 |
| ssim | `threshold_ssim` | 0.1475 | 0.021 | 1.6% | 192 |
| pixdiff | `threshold_pixdiff` | 0.2512 | 0.042 | 2.1% | 192 |

## TEST results

### Lighting: none + monotone (the operating domain) — 1152 defect pairs, 192 defect-free pairs

| Method | Defect recall | Escapes (defect → PASS) | False-alarm rate (good → FAIL) | Good parts not passed (FAIL or RETAKE) | False zones / pair | Retake rate (all) | Coverage | Mean ms / inspection |
|---|---|---|---|---|---|---|---|---|
| pbp | 67.8% (65.0–70.4) | 23.6% (21.2–26.1) | 1.0% (0.3–3.7) | 2.1% (0.8–5.2) | 0.039 | 3.1% (2.3–4.1) | 94.3% | 495 |
| census | 26.5% (24.0–29.1) | 62.3% (59.5–65.1) | 2.1% (0.8–5.2) | 3.1% (1.4–6.6) | 0.026 | 3.3% (2.4–4.4) | 94.3% | 248 |
| ssim | 67.3% (64.5–69.9) | 24.9% (22.5–27.5) | 1.0% (0.3–3.7) | 2.1% (0.8–5.2) | 0.021 | 2.8% (2.1–3.9) | 94.3% | 205 |
| pixdiff | 16.1% (14.1–18.3) | 75.0% (72.4–77.4) | 2.1% (0.8–5.2) | 3.1% (1.4–6.6) | 0.053 | 4.7% (3.7–6.0) | 94.3% | 225 |

### Lighting: non-monotone (specular highlight, hand shadow) — expected failure — 384 defect pairs, 64 defect-free pairs

| Method | Defect recall | Escapes (defect → PASS) | False-alarm rate (good → FAIL) | Good parts not passed (FAIL or RETAKE) | False zones / pair | Retake rate (all) | Coverage | Mean ms / inspection |
|---|---|---|---|---|---|---|---|---|
| pbp | 68.8% (63.9–73.2) | 6.2% (4.2–9.1) | 93.8% (85.0–97.5) | 96.9% (89.3–99.1) | 3.342 | 3.1% (1.9–5.2) | 92.1% | 495 |
| census | 33.1% (28.6–37.9) | 10.7% (8.0–14.2) | 89.1% (79.1–94.6) | 92.2% (83.0–96.6) | 4.348 | 4.7% (3.1–7.1) | 92.1% | 248 |
| ssim | 65.6% (60.7–70.2) | 6.8% (4.7–9.7) | 84.4% (73.6–91.3) | 96.9% (89.3–99.1) | 1.426 | 9.6% (7.2–12.7) | 92.1% | 205 |
| pixdiff | 23.4% (19.5–27.9) | 23.4% (19.5–27.9) | 62.5% (50.3–73.3) | 78.1% (66.6–86.5) | 0.732 | 14.3% (11.3–17.8) | 92.1% | 225 |

### All lighting conditions — 1536 defect pairs, 256 defect-free pairs

| Method | Defect recall | Escapes (defect → PASS) | False-alarm rate (good → FAIL) | Good parts not passed (FAIL or RETAKE) | False zones / pair | Retake rate (all) | Coverage | Mean ms / inspection |
|---|---|---|---|---|---|---|---|---|
| pbp | 68.0% (65.7–70.3) | 19.3% (17.4–21.3) | 24.2% (19.4–29.8) | 25.8% (20.8–31.5) | 0.864 | 3.1% (2.4–4.0) | 93.8% | 495 |
| census | 28.1% (25.9–30.4) | 49.4% (46.9–51.9) | 23.8% (19.0–29.4) | 25.4% (20.4–31.1) | 1.107 | 3.6% (2.9–4.6) | 93.8% | 248 |
| ssim | 66.9% (64.5–69.2) | 20.4% (18.4–22.5) | 21.9% (17.2–27.3) | 25.8% (20.8–31.5) | 0.372 | 4.5% (3.7–5.6) | 93.8% | 205 |
| pixdiff | 17.9% (16.1–19.9) | 62.1% (59.7–64.5) | 17.2% (13.1–22.3) | 21.9% (17.2–27.3) | 0.223 | 7.1% (6.0–8.4) | 93.8% | 225 |

### By lighting kind — defect recall / false-alarm rate

| Lighting | Group | pbp | census | ssim | pixdiff |
|---|---|---|---|---|---|
| none | — | 66.1% (59.2–72.5) / 0.0% (0.0–10.7) | 26.0% (20.3–32.7) / 3.1% (0.6–15.7) | 67.7% (60.8–73.9) / 0.0% (0.0–10.7) | 10.9% (7.3–16.1) / 0.0% (0.0–10.7) |
| drift_gain | monotone | 66.7% (59.7–73.0) / 0.0% (0.0–10.7) | 28.6% (22.7–35.4) / 3.1% (0.6–15.7) | 65.6% (58.7–72.0) / 0.0% (0.0–10.7) | 19.3% (14.3–25.4) / 0.0% (0.0–10.7) |
| gamma | monotone | 68.8% (61.9–74.9) / 3.1% (0.6–15.7) | 28.1% (22.2–34.9) / 0.0% (0.0–10.7) | 67.2% (60.3–73.4) / 3.1% (0.6–15.7) | 17.7% (13.0–23.7) / 0.0% (0.0–10.7) |
| window_gradient | monotone | 73.4% (66.8–79.2) / 0.0% (0.0–10.7) | 27.6% (21.8–34.3) / 6.2% (1.7–20.1) | 69.8% (63.0–75.8) / 0.0% (0.0–10.7) | 16.7% (12.1–22.6) / 0.0% (0.0–10.7) |
| lamp_aging | monotone | 71.4% (64.6–77.3) / 3.1% (0.6–15.7) | 26.0% (20.3–32.7) / 0.0% (0.0–10.7) | 71.4% (64.6–77.3) / 3.1% (0.6–15.7) | 16.1% (11.6–22.0) / 0.0% (0.0–10.7) |
| auto_exposure | monotone | 60.4% (53.4–67.1) / 0.0% (0.0–10.7) | 22.4% (17.1–28.8) / 0.0% (0.0–10.7) | 62.0% (54.9–68.5) / 0.0% (0.0–10.7) | 15.6% (11.2–21.4) / 12.5% (5.0–28.1) |
| **monotone (all)** |  | 68.1% (65.1–71.0) / 1.2% (0.3–4.4) | 26.6% (23.9–29.4) / 1.9% (0.6–5.4) | 67.2% (64.2–70.1) / 1.2% (0.3–4.4) | 17.1% (14.8–19.6) / 2.5% (1.0–6.3) |
| specular | non-monotone | 65.1% (58.1–71.5) / 90.6% (75.8–96.8) | 29.7% (23.7–36.5) / 90.6% (75.8–96.8) | 64.1% (57.1–70.5) / 75.0% (57.9–86.7) | 15.6% (11.2–21.4) / 65.6% (48.3–79.6) |
| hand_shadow | non-monotone | 72.4% (65.7–78.2) / 96.9% (84.3–99.4) | 36.5% (30.0–43.5) / 87.5% (71.9–95.0) | 67.2% (60.3–73.4) / 93.8% (79.9–98.3) | 31.2% (25.1–38.1) / 59.4% (42.3–74.5) |
| **non-monotone (all)** |  | 68.8% (63.9–73.2) / 93.8% (85.0–97.5) | 33.1% (28.6–37.9) / 89.1% (79.1–94.6) | 65.6% (60.7–70.2) / 84.4% (73.6–91.3) | 23.4% (19.5–27.9) / 62.5% (50.3–73.3) |

### Per-defect recall (none + monotone lighting)

| Defect | pbp | census | ssim | pixdiff |
|---|---|---|---|---|
| scratch | 69.8% (63.0–75.8) | 1.6% (0.5–4.5) | 60.9% (53.9–67.6) | 2.6% (1.1–6.0) |
| tape | 75.5% (69.0–81.1) | 27.1% (21.3–33.8) | 71.4% (64.6–77.3) | 29.2% (23.2–36.0) |
| missing_hole | 74.0% (67.3–79.7) | 59.4% (52.3–66.1) | 72.9% (66.2–78.7) | 25.0% (19.4–31.6) |
| burr | 25.5% (19.9–32.1) | 2.6% (1.1–6.0) | 43.2% (36.4–50.3) | 1.6% (0.5–4.5) |
| dent | 83.9% (78.0–88.4) | 59.9% (52.8–66.6) | 80.2% (74.0–85.2) | 14.1% (9.8–19.7) |
| stain | 78.1% (71.8–83.4) | 8.3% (5.2–13.1) | 75.0% (68.4–80.6) | 24.0% (18.5–30.5) |

### By motion and part kind (none + monotone lighting) — recall / false-alarm rate

| Subset | pbp | census | ssim | pixdiff |
|---|---|---|---|---|
| motion:jitter | 85.6% / 0.0% | 38.9% / 0.0% | 86.6% / 0.0% | 22.2% / 4.2% |
| motion:shift | 50.0% / 2.1% | 14.1% / 4.2% | 47.9% / 2.1% | 9.9% / 0.0% |
| part:stamped | 68.4% / 0.0% | 19.4% / 0.0% | 68.4% / 0.0% | 21.2% / 0.0% |
| part:painted | 67.4% / 0.0% | 21.9% / 4.2% | 64.9% / 0.0% | 12.5% / 0.0% |
| part:molded | 76.7% / 2.1% | 34.0% / 4.2% | 74.7% / 2.1% | 12.2% / 8.3% |
| part:machined | 58.7% / 2.1% | 30.6% / 0.0% | 61.1% / 2.1% | 18.4% / 0.0% |

### Retake reasons (all TEST pairs)

| Method | Reasons |
|---|---|
| pbp | glare: 46, blurry: 8, mismatch: 1 |
| census | glare: 57, blurry: 8 |
| ssim | glare: 60, mismatch: 13, blurry: 8 |
| pixdiff | glare: 95, mismatch: 24, blurry: 8 |

### Threshold-free separability (AUROC of the max smoothed cell score, defect vs defect-free)

| Method | none + monotone | non-monotone | all |
|---|---|---|---|
| pbp | 0.931 | 0.474 | 0.790 |
| census | 0.887 | 0.468 | 0.737 |
| ssim | 0.934 | 0.584 | 0.821 |
| pixdiff | 0.703 | 0.531 | 0.638 |

Pairs rejected before scoring (alignment / glare / blur / coverage) have no score and are left out of the AUROC. The full recall-vs-false-alarm curves are in `eval/results/test_sweep.json` (and `calib_sweep.json`).

### Paired comparison: PBP vs each method on the same TEST pairs

| Lighting | vs | Recall PBP vs it | Defects found only by PBP / only by it | McNemar p | Recall | False alarms PBP vs it | False alarms only PBP / only it | McNemar p | False alarms |
|---|---|---|---|---|---|---|---|---|---|
| none+monotone | census | 67.8% vs 26.5% | 479 / 3 | 3e-138 | PBP better | 1.0% vs 2.1% | 2 / 4 | 0.69 | no significant difference |
| none+monotone | ssim | 67.8% vs 67.3% | 68 / 62 | 0.66 | no significant difference | 1.0% vs 1.0% | 0 / 0 | 1 | no significant difference |
| none+monotone | pixdiff | 67.8% vs 16.1% | 605 / 9 | 9.6e-166 | PBP better | 1.0% vs 2.1% | 2 / 4 | 0.69 | no significant difference |
| non_monotone | census | 68.8% vs 33.1% | 138 / 1 | 4e-40 | PBP better | 93.8% vs 89.1% | 3 / 0 | 0.25 | no significant difference |
| non_monotone | ssim | 68.8% vs 65.6% | 23 / 11 | 0.058 | no significant difference | 93.8% vs 84.4% | 6 / 0 | 0.031 | ssim better |
| non_monotone | pixdiff | 68.8% vs 23.4% | 178 / 4 | 1.5e-47 | PBP better | 93.8% vs 62.5% | 20 / 0 | 1.9e-06 | pixdiff better |

Exact two-sided McNemar tests on the discordant pairs (each pair is seen by both methods), at the frozen thresholds; p-values are not corrected for the number of comparisons.

## Speed (sequential pass, this machine)

| Method | Mean ms | Median ms | Min–max ms | Inspections |
|---|---|---|---|---|
| pbp | 495 | 443 | 96–716 | 24 |
| census | 248 | 218 | 117–366 | 24 |
| ssim | 205 | 192 | 102–296 | 24 |
| pixdiff | 225 | 221 | 110–351 | 24 |

PBP stage medians (ms): resize 1, quality 4, align 55, mask 39, features 96, change 159, zones 1, render 72, total 453, extra_views 253. `prepare_reference` (1 shot, PBP): median 132 ms. Means include inspections that stopped early at a RETAKE gate; stage medians are over the inspections that reached the stage (`extra_views`: moved part only).

CPU: 13th Gen Intel(R) Core(TM) i5-13450HX. Frames 1280x720 resized to max side 1024 px. Measured 2026-10-11 02:47 UTC with `python -m eval.benchmark --split timing --timing-pairs 24`. Other processes on the machine were not controlled; treat as indicative.

## Caveats (read before quoting any number)

- **Synthetic data only.** Parts, defects and lighting are procedural (`ava/synth.py`). Real scratches, dents
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
- **Small N.** 4 pair(s) per grid cell; per-cell and per-class rates rest on few pairs, see the Wilson
  95 % intervals. Differences inside overlapping intervals are not evidence.
- **One calibration criterion for all methods** (false events ≤ 0.05 per good pair under none + monotone
  lighting). A different operating point changes the trade-off; the full sweeps are in `eval/results/`.
- **The ground truth is a bbox.** Zones live on a 4 px window grid; recall uses the bbox IoU / coverage rule
  above, false zones use "does not touch the (dilated) defect mask".
- Defect classes are not equally hard: burrs are small blobs on an edge, and a bright burr under strong
  exposure can be masked as glare (unchecked; the frame is then answered RETAKE rather than PASS) — see the
  per-defect table.

## Real data

VisA (Amazon, CC BY 4.0) evaluation was **not run** in this benchmark (`--visa` not given). A download of VisA_20220922.tar (about 1.8 GB) was attempted on 2026-10-10; measured speed was 40-50 KB/s (about 10 hours), so it was abandoned. Real-data evaluation is **pending**: the team photo set in `data/pairs/` (protocol in `data/pairs/README.md`). No real-data metric is claimed here.
