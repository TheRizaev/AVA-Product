# Synthetic benchmark — procedurally generated uniform surfaces, NOT real photos

Every number in this file was produced by `eval/benchmark_surface.py` (raw data in `eval/results/surface_*.json`). The photos come from `dalil/synth.py`: procedural close-ups of painted, plastic, brushed-metal and dark glossy surfaces that fill the whole frame, procedural defects and procedural shop-floor lighting. They compare the methods under controlled conditions; they are **not** evidence of accuracy on real parts.

## Run

```
python -m eval.benchmark_surface --n 4 --clean-mult 3 --seed 0 --workers 10
```

|  |  |
|---|---|
| Date (calib / test) | 2026-10-10 17:58 UTC / 2026-10-10 18:02 UTC |
| CPU | 13th Gen Intel(R) Core(TM) i5-13450HX (16 logical cores) |
| Software | Windows-10-10.0.26200-SP0; Python 3.11.9, numpy 2.2.6, OpenCV 4.12.0 |
| Code | 26c853e + uncommitted changes in dalil/eval |
| Grid | 4 surface kinds x 8 lightings x 7 defect classes; 4 photo(s) per defect cell, 12 defect-free photo(s) per (kind, lighting) = 1152 TEST photos (1152 CALIB) |
| Surface kinds | painted, plastic, brushed, glass_dark |
| Lightings | none, drift_gain, gamma, window_gradient, lamp_aging, auto_exposure, specular, hand_shadow |
| Defects | none, scratch, chip, stain, dent, burr, sticker |
| Image size | 1280x720 px generated, working size max side 1024 px |
| Seeds | --seed 0; per photo SeedSequence(seed, split code calib=101/test=102, kind, lighting, defect, repeat) |
| Wall time | calib 285 s, test 241 s with 10 worker processes (1 thread each) |
| Photos that raised an exception | calib 0, test 0 (excluded; listed in the JSON) |
| Sweep vs scan_surface consistency | 0 + 0 mismatches at the operating point |

## Protocol

1. **Photos.** `synth.make_uniform_surface` renders a surface filling the whole frame (with a soft
   illumination fall-off and sensor noise); a defect is drawn on it (scratch, chip, stain, dent, burr, sticker) or not; then
   one shop-floor lighting change is applied to the whole photo, or none. Monotone: drift_gain, gamma, window_gradient, lamp_aging, auto_exposure;
   non-monotone: specular, hand_shadow.
2. **Same pipeline for every method.** `surface.scan_surface(photo, params, method)`: identical capture
   gates (dark / blurry / glare / coverage), glare mask, smoothing and zone extraction; only the per-window
   anomaly score differs — PBP self-reference kNN + contrast term (AVA), the same kNN on Census codes, or a
   robust z-score of the Sobel gradient magnitude.
3. **Calibration (CALIB split only).** Each method's smoothed score map is re-thresholded over a geometric
   grid (0.02–2.5). Per method we take the smallest grid threshold at which the **false-alarm rate** (defect-free
   photos answered FAIL) under **none + monotone** lighting is ≤ 5 %, at that threshold and every
   larger one. The thresholds are frozen and written to `dalil/config.py`.
4. **Test (TEST split, disjoint seeds).** *Recall*: a defect photo is detected if the verdict is FAIL and
   some zone has IoU > 0.1 with the true defect bbox or covers ≥ 30 % of it (RETAKE counts as
   a miss). *False-alarm rate*: defect-free photos answered FAIL. *False zones / photo*: zones that do not
   touch the defect (dilated by one window) — on a defect-free photo every zone — averaged over all photos.
   *Retake rate*: photos answered RETAKE (glare, blurry, dark, low coverage), all photos. Brackets: Wilson
   95 % intervals.
5. **Timing.** A separate sequential pass in one process (OpenCV / BLAS threads on),
   12 TEST photos after one warm-up photo; times include resize, capture checks, glare
   mask, scoring, zones and the overlay JPEG. The "ms (parallel)" columns are measured inside the
   10 benchmark workers (1 thread each, CPU shared) and are only comparable with each other.

## Calibrated thresholds (frozen before TEST)

| Method | Param | Threshold | CALIB false-alarm rate | CALIB recall (none+monotone) |
|---|---|---|---|---|
| PBP (AVA) | `surface_threshold` | 0.2376 | 3.5% (10/288) | 97.2% |
| Census kNN | `threshold_surface_census` | 0.146 | 4.2% (12/288) | 97.4% |
| Gradient z-score | `threshold_surface_gradient` | 0.535 | 2.8% (8/288) | 97.2% |

## TEST results

### Lighting: none + monotone (the operating domain) — 576 defect photos, 288 defect-free photos

| Method | Defect recall | False-alarm rate (good → FAIL) | False zones / photo | Retake rate | Coverage | Mean ms (sequential) | Mean ms (parallel) |
|---|---|---|---|---|---|---|---|
| PBP (AVA) | 97.9% (96.4–98.8) | 5.2% (3.2–8.4) | 0.057 | 0.0% (0.0–0.4) | 93.5% | 187 | 343 |
| Census kNN | 97.9% (96.4–98.8) | 5.9% (3.7–9.2) | 0.046 | 0.0% (0.0–0.4) | 93.5% | 106 | 202 |
| Gradient z-score | 97.9% (96.4–98.8) | 3.1% (1.7–5.8) | 0.035 | 0.0% (0.0–0.4) | 93.5% | 65 | 119 |

### Lighting: none — 96 defect photos, 48 defect-free photos

| Method | Defect recall | False-alarm rate (good → FAIL) | False zones / photo | Retake rate | Coverage | Mean ms (sequential) | Mean ms (parallel) |
|---|---|---|---|---|---|---|---|
| PBP (AVA) | 99.0% (94.3–99.8) | 0.0% (0.0–7.4) | 0.021 | 0.0% (0.0–2.6) | 93.5% | 187 | 362 |
| Census kNN | 99.0% (94.3–99.8) | 4.2% (1.2–14.0) | 0.028 | 0.0% (0.0–2.6) | 93.5% | 106 | 206 |
| Gradient z-score | 99.0% (94.3–99.8) | 0.0% (0.0–7.4) | 0.014 | 0.0% (0.0–2.6) | 93.5% | 65 | 124 |

### Lighting: monotone changes only — 480 defect photos, 240 defect-free photos

| Method | Defect recall | False-alarm rate (good → FAIL) | False zones / photo | Retake rate | Coverage | Mean ms (sequential) | Mean ms (parallel) |
|---|---|---|---|---|---|---|---|
| PBP (AVA) | 97.7% (95.9–98.7) | 6.2% (3.8–10.1) | 0.064 | 0.0% (0.0–0.5) | 93.5% | 187 | 339 |
| Census kNN | 97.7% (95.9–98.7) | 6.2% (3.8–10.1) | 0.050 | 0.0% (0.0–0.5) | 93.5% | 106 | 202 |
| Gradient z-score | 97.7% (95.9–98.7) | 3.8% (2.0–7.0) | 0.039 | 0.0% (0.0–0.5) | 93.5% | 65 | 118 |

### Lighting: non-monotone (specular highlight, hand shadow) — expected failure — 192 defect photos, 96 defect-free photos

| Method | Defect recall | False-alarm rate (good → FAIL) | False zones / photo | Retake rate | Coverage | Mean ms (sequential) | Mean ms (parallel) |
|---|---|---|---|---|---|---|---|
| PBP (AVA) | 90.1% (85.1–93.6) | 52.1% (42.2–61.8) | 0.368 | 0.0% (0.0–1.3) | 86.4% | 187 | 379 |
| Census kNN | 90.1% (85.1–93.6) | 52.1% (42.2–61.8) | 0.372 | 0.0% (0.0–1.3) | 86.4% | 106 | 238 |
| Gradient z-score | 90.1% (85.1–93.6) | 53.1% (43.2–62.8) | 0.368 | 0.0% (0.0–1.3) | 86.4% | 65 | 131 |

### All lighting conditions — 768 defect photos, 384 defect-free photos

| Method | Defect recall | False-alarm rate (good → FAIL) | False zones / photo | Retake rate | Coverage | Mean ms (sequential) | Mean ms (parallel) |
|---|---|---|---|---|---|---|---|
| PBP (AVA) | 96.0% (94.3–97.1) | 16.9% (13.5–21.0) | 0.135 | 0.0% (0.0–0.3) | 91.8% | 187 | 352 |
| Census kNN | 96.0% (94.3–97.1) | 17.4% (14.0–21.6) | 0.128 | 0.0% (0.0–0.3) | 91.8% | 106 | 211 |
| Gradient z-score | 96.0% (94.3–97.1) | 15.6% (12.3–19.6) | 0.118 | 0.0% (0.0–0.3) | 91.8% | 65 | 122 |

### By lighting kind — recall / false-alarm rate / false zones per photo / retake rate

| Lighting | Group | PBP (AVA) | Census kNN | Gradient z-score |
|---|---|---|---|---|
| none | — | 99.0% (94.3–99.8) / 0.0% (0.0–7.4) / 0.02 / 0.0% | 99.0% (94.3–99.8) / 4.2% (1.2–14.0) / 0.03 / 0.0% | 99.0% (94.3–99.8) / 0.0% (0.0–7.4) / 0.01 / 0.0% |
| drift_gain | monotone | 94.8% (88.4–97.8) / 10.4% (4.5–22.2) / 0.04 / 0.0% | 94.8% (88.4–97.8) / 8.3% (3.3–19.6) / 0.05 / 0.0% | 94.8% (88.4–97.8) / 6.2% (2.1–16.8) / 0.03 / 0.0% |
| gamma | monotone | 100.0% (96.2–100.0) / 4.2% (1.2–14.0) / 0.05 / 0.0% | 100.0% (96.2–100.0) / 4.2% (1.2–14.0) / 0.03 / 0.0% | 100.0% (96.2–100.0) / 2.1% (0.4–10.9) / 0.04 / 0.0% |
| window_gradient | monotone | 100.0% (96.2–100.0) / 8.3% (3.3–19.6) / 0.15 / 0.0% | 100.0% (96.2–100.0) / 14.6% (7.2–27.2) / 0.13 / 0.0% | 100.0% (96.2–100.0) / 10.4% (4.5–22.2) / 0.09 / 0.0% |
| lamp_aging | monotone | 100.0% (96.2–100.0) / 6.2% (2.1–16.8) / 0.04 / 0.0% | 100.0% (96.2–100.0) / 4.2% (1.2–14.0) / 0.01 / 0.0% | 100.0% (96.2–100.0) / 0.0% (0.0–7.4) / 0.01 / 0.0% |
| auto_exposure | monotone | 93.8% (87.0–97.1) / 2.1% (0.4–10.9) / 0.04 / 0.0% | 93.8% (87.0–97.1) / 0.0% (0.0–7.4) / 0.03 / 0.0% | 93.8% (87.0–97.1) / 0.0% (0.0–7.4) / 0.02 / 0.0% |
| specular | non-monotone | 80.2% (71.1–86.9) / 4.2% (1.2–14.0) / 0.05 / 0.0% | 80.2% (71.1–86.9) / 4.2% (1.2–14.0) / 0.06 / 0.0% | 80.2% (71.1–86.9) / 6.2% (2.1–16.8) / 0.04 / 0.0% |
| hand_shadow | non-monotone | 100.0% (96.2–100.0) / 100.0% (92.6–100.0) / 0.69 / 0.0% | 100.0% (96.2–100.0) / 100.0% (92.6–100.0) / 0.69 / 0.0% | 100.0% (96.2–100.0) / 100.0% (92.6–100.0) / 0.69 / 0.0% |

### Per-defect recall (none + monotone lighting)

| Defect | Photos | PBP (AVA) | Census kNN | Gradient z-score |
|---|---|---|---|---|
| scratch | 96 | 100.0% (96.2–100.0) | 100.0% (96.2–100.0) | 100.0% (96.2–100.0) |
| chip | 96 | 96.9% (91.2–98.9) | 96.9% (91.2–98.9) | 96.9% (91.2–98.9) |
| stain | 96 | 100.0% (96.2–100.0) | 100.0% (96.2–100.0) | 100.0% (96.2–100.0) |
| dent | 96 | 97.9% (92.7–99.4) | 97.9% (92.7–99.4) | 97.9% (92.7–99.4) |
| burr | 96 | 93.8% (87.0–97.1) | 93.8% (87.0–97.1) | 93.8% (87.0–97.1) |
| sticker | 96 | 99.0% (94.3–99.8) | 99.0% (94.3–99.8) | 99.0% (94.3–99.8) |

### Per surface kind (none + monotone lighting) — recall / false-alarm rate

| Surface | PBP (AVA) | Census kNN | Gradient z-score |
|---|---|---|---|
| painted | 97.2% (93.1–98.9) / 0.0% (0.0–5.1) | 97.2% (93.1–98.9) / 1.4% (0.2–7.5) | 97.2% (93.1–98.9) / 0.0% (0.0–5.1) |
| plastic | 97.2% (93.1–98.9) / 0.0% (0.0–5.1) | 97.2% (93.1–98.9) / 6.9% (3.0–15.2) | 97.2% (93.1–98.9) / 0.0% (0.0–5.1) |
| brushed | 98.6% (95.1–99.6) / 20.8% (13.1–31.6) | 98.6% (95.1–99.6) / 15.3% (8.8–25.3) | 98.6% (95.1–99.6) / 12.5% (6.7–22.1) |
| glass_dark | 98.6% (95.1–99.6) / 0.0% (0.0–5.1) | 98.6% (95.1–99.6) / 0.0% (0.0–5.1) | 98.6% (95.1–99.6) / 0.0% (0.0–5.1) |

### PBP (AVA): detected / photos per surface kind x defect (none + monotone lighting)

| Surface | scratch | chip | stain | dent | burr | sticker |
|---|---|---|---|---|---|---|
| painted | 24/24 | 22/24 | 24/24 | 23/24 | 23/24 | 24/24 |
| plastic | 24/24 | 24/24 | 24/24 | 24/24 | 21/24 | 23/24 |
| brushed | 24/24 | 24/24 | 24/24 | 23/24 | 23/24 | 24/24 |
| glass_dark | 24/24 | 23/24 | 24/24 | 24/24 | 23/24 | 24/24 |

### Why defects were missed (TEST)

| Lighting | Method | Missed | RETAKE | Defect < 50 % checked (glare / border mask) | Checked, no zone at threshold |
|---|---|---|---|---|---|
| none+monotone | PBP (AVA) | 12 | 0 | 12 | 0 |
| none+monotone | Census kNN | 12 | 0 | 12 | 0 |
| none+monotone | Gradient z-score | 12 | 0 | 12 | 0 |
| non_monotone | PBP (AVA) | 19 | 0 | 18 | 1 |
| non_monotone | Census kNN | 19 | 0 | 18 | 1 |
| non_monotone | Gradient z-score | 19 | 0 | 18 | 1 |

On 768 of 768 TEST defect photos all 3 methods give the same detected / missed outcome; on 361 of 384 defect-free photos they give the same verdict. The capture gates, the glare mask and the zone rules are shared, and the synthetic defects are high-contrast, so on this set the methods differ mainly in false alarms and speed, not in recall.

### Retake reasons (all TEST photos)

| Method | Reasons |
|---|---|
| PBP (AVA) | — |
| Census kNN | — |
| Gradient z-score | — |

### Threshold-free separability (AUROC of the max smoothed cell score, defect vs defect-free)

| Method | none + monotone | non-monotone | all |
|---|---|---|---|
| PBP (AVA) | 0.990 | 0.757 | 0.934 |
| Census kNN | 0.991 | 0.763 | 0.939 |
| Gradient z-score | 0.991 | 0.784 | 0.942 |

Photos rejected before scoring (glare / blur / dark / coverage) have no score and are left out of the AUROC. Recall / false-alarm / false-zone / retake curves over the whole threshold grid, per lighting group, are in `eval/results/surface_test_sweep.json` (and `surface_calib_sweep.json`).

## Speed (sequential pass, this machine)

| Method | Mean ms | Median ms | Min–max ms | Photos |
|---|---|---|---|---|
| PBP (AVA) | 187 | 185 | 169–227 | 12 |
| Census kNN | 106 | 99 | 90–138 | 12 |
| Gradient z-score | 65 | 64 | 58–76 | 12 |

PBP stage medians (ms): resize 1, quality 8, mask 28, score 123, zones 1, render 19, total 185.

CPU: 13th Gen Intel(R) Core(TM) i5-13450HX. Photos 1280x720 resized to max side 1024 px. Other processes on the machine were not controlled; treat the times as indicative. The deployed server CPU will differ.

## Caveats (read before quoting any number)

- **Synthetic data only.** Surfaces, defects and lighting are procedural (`dalil/synth.py`). Real
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
- **Small N.** 4 photo(s) per defect cell and 12 defect-free photo(s) per
  (kind, lighting). Check the Wilson 95 % intervals; differences inside overlapping intervals are not evidence.
- **One calibration criterion for all methods** (false-alarm rate ≤ 5 % on defect-free CALIB photos
  under none + monotone lighting). A different operating point changes the trade-off; see the sweeps.
- **The ground truth is a bbox.** Zones live on a 4 px window grid; recall uses the bbox IoU /
  coverage rule above, false zones use "does not touch the (dilated) defect mask".

## Real data

No real-photo evaluation of Mode 1 is included in this file. No real-data metric is claimed here.
