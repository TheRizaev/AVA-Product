# AVA — technical contract (authoritative)

The product name in all user-facing text (UI, README, documents) is **AVA**. The Python package is `ava`.

**Product.** A visual defect check of parts from photos, in two modes chosen by the user:
1. **Surface check (1 photo)**: a uniform surface or a regular texture. The surface is its own reference: rare windows are defects. Implemented in `ava/surface.py`, served at `POST /api/scan`.
2. **Compare with reference (2 photos)**: complex parts. The Check photo is aligned to a Reference photo of a good part and compared with PBP window signatures. Implemented in `ava/pipeline.py`, served at `POST /api/compare`.

**Verdict:** PASS, FAIL (+ defect zones) or RETAKE (+ reason). There are no stations, journals or accounts: one screen, one result. In Mode 2 the result is shown next to plain reference subtraction.

**Differentiator to prove with numbers:** the comparison stays below the threshold under monotone lighting drift without a light box, measured against plain reference subtraction (`pixdiff`), SSIM and Census on the same pipeline (eval/RESULTS.md).

Every module implements exactly the public API below; internal helpers are free.

**Conventions:**
- Images are `H×W` gray float32 0..255 or `H×W×3` BGR uint8 (OpenCV order).
- All tunables live in `ava/config.py::Params`. Fields may be added with defaults; never rename or remove one.
- Terminology: `reference` (golden) and `frame` (inspected). Older modules keep their parameter names `before` / `after`.

## Repository layout
```
ava/
  config.py      Params (incl. calibrated thresholds)
  pbp.py         pbp_reference, format_polynomial, poly_degree, Signatures, window_signatures,
                 pair_distance, serialize_signatures, grid_shape, cell_mask_from_pixel_mask
  change.py      tolerant_change_map, multi_reference_change_map, pbp_change_map
  zones.py       Zone, extract_zones, coverage (hints: water_or_dirt, something_removed, something_added)
  align.py       AlignResult, align (identity / ORB+RANSAC / ECC + part motion), corner_shift, resize_max_side
  quality.py     glare_mask, blur_score, is_too_dark, capture_report, outline_png
  baselines.py   ssim / census / pixdiff features + distances, get_methods()
  synth.py       procedural parts, uniform surfaces, defects, factory lighting, motion
  pipeline.py    Mode 2: Reference, InspectResult, prepare_reference, inspect_compare, inspect_frame,
                 decide, smooth_scores, render_overlay, part_roi, contrast_gain, match_scale,
                 bright_spot, result_to_json
  surface.py     Mode 1: scan_surface, surface_score_map, glare_region, method_threshold
server/
  app.py         FastAPI: API under /api, static web/ at /
web/             vanilla JS mobile-first UI (index.html, app.js, style.css, i18n.js, qr.js, qrcode.js,
                 icon.svg, examples/)
eval/            benchmark.py (Mode 2) -> eval/RESULTS.md; benchmark_surface.py (Mode 1) ->
                 eval/RESULTS_SURFACE.md; make_examples.py -> web/examples/; raw data in eval/results/
tests/           pytest suite (python -m pytest -q must pass)
Dockerfile       HF Spaces (port 7860)
```

## Removed

These are deleted and must not come back:
- the station / journal / report product: `report.py`, `server/store.py`, `act.py`;
- all `/api/stations/...` and `/api/files/...` routes;
- the station, viewer, journal and report screens, and their tests;
- the UI language switch and all Russian UI strings.

## Window grid (shared)

- Windows are `win×win` (8) at `y=i*step`, `x=j*step` (`step` = 4).
- `gh=(H-win)//step+1`, `gw=(W-win)//step+1`, and per-window arrays are `[gh,gw]`.
- `cell_mask_from_pixel_mask(mask, win, step, frac=0.25)` marks a cell True if ≥ `frac` of its window pixels are True.

## ava/pbp.py

**PBP of an m×n patch.** Per column, for each distinct value `t > min`, add `(t - prev(t)) * Π_{i: C[i,j] < t} y_i`. The constant is `Σ` of the column minima. Like monomials are merged.
- `pbp_reference(patch) -> dict[tuple, float]`: the exact PBP.
- `format_polynomial(poly)`: renders the paper's form, e.g. `"11 + 11y3 + 3y4 + 2y1y3 + 4y2y3 + 4y1y4 + 12y1y2y3 + 6y1y2y4"`.
- `poly_degree(poly)`.

**`window_signatures(gray, params) -> Signatures`.**
- `win` = 8 only.
- Two orientations: the rows are the variables; then the transposed window.
- ε-ties: `C[r] < t - eps*contrast`.
- Constant dropped; 256-bin coefficient vectors normalised to sum 1 per orientation.
- `flat = contrast < flat_floor`, with all-zero vectors.
- Fields: `coeffs float32[gh,gw,2,256]`, `contrast float32[gh,gw]`, `degree int8[gh,gw,2]`, `flat bool[gh,gw]`, `win`, `step`, `shape`, `eps`, `flat_floor`.

**`pair_distance(a, b, dy=0, dx=0) -> float32[gh,gw]`.**
- The mean over orientations of `0.5·L1`, in [0,1].
- Out-of-range partner → +inf; flat-vs-flat 0; flat-vs-nonflat 1.
- Symmetric: `D(a,b,dy,dx)[i,j] == D(b,a,-dy,-dx)[i+dy,j+dx]`.

**`serialize_signatures(sig) -> bytes`**: deterministic, used for SHA-256 fingerprints.

**Invariance (tested; the README states it):**
- the monomial *set* is invariant to strictly monotone brightness changes;
- the normalised signature is invariant to positive affine `a·x+b`, except windows whose contrast crosses the absolute `flat_floor` (Mode 2 compensates, see `_change_map`);
- under gamma-like changes the coefficients shift slightly.

## ava/change.py

- `tolerant_change_map(dist_fn, a, b, tol, symmetric=None) -> float32[gh,gw]`:
  - `d_ab = min_{|dy|,|dx|≤tol} dist_fn(a,b,dy,dx)` and `d_ba = min dist_fn(b,a,dy,dx)`;
  - the result is `max(d_ab, d_ba)`, with no partner in range → 1.0;
  - symmetric distances (PBP and all baselines) reuse the `d_ab` maps.
- `multi_reference_change_map(dist_fn, refs, frame, tol)`: element-wise MIN over the references.
- `pbp_change_map(sig_ref | [sig_ref...], sig_frame, params)`: a convenience with `params.tolerance`.

## ava/align.py

`align(before_bgr, after_bgr, params) -> AlignResult`, where H maps AFTER (frame) → BEFORE (reference) coordinates.

**`AlignResult` fields:**
- `ok`, `reason` (`too_few_features` / `too_few_inliers` / `degenerate_homography`);
- `H`, `n_matches`, `n_inliers`, `inlier_ratio`;
- `after_warped`: BGR, warped with a replicated border;
- `valid`: bool; validity is defined by this mask only;
- `method`: `"identity" | "orb" | "ecc" | "none"`;
- `score`: identity robust NCC, ORB inlier ratio, or ECC correlation;
- `alt_H`;
- `inlier_pts`: reference-frame points of H's inliers.

**Order of the paths:**
1. **identity.** Lighting-normalised robust NCC ≥ `fixed_cam_ncc`, and a quick ECC check finds a visible motion ≤ `fixed_cam_max_shift` px.
2. **orb.** ORB + ratio test + RANSAC homography with a least-squares refit. Accepted with ≥ `min_inliers` inliers, an inlier ratio ≥ `min_inlier_ratio`, and a non-degenerate H.
3. **ecc.** Only after an ORB `too_few_features` / `too_few_inliers`. Coarse-to-fine Euclidean ECC on lighting-normalised images, accepted if the correlation is ≥ `ecc_min_corr`.

**Part motion** (fixed camera, part moved over a static background):
- A second homography is fitted on ORB's outliers. If it has ≥ `min_inliers` inliers and its corners move by more than `2·ransac_thresh` relative to H, it is kept.
- If none was found and H is ≈ identity, a similarity is fitted on the outliers instead. It needs ≥ `part_min_inliers` inliers and is accepted only if it beats identity by `part_ncc_margin` in robust NCC on the hull of its inliers.
- If the dominant motion is ≈ identity and the second is not, they are swapped: `H` = part, `alt_H` = background.

`corner_shift(H, shape)` gives the largest corner displacement in px.

## ava/pipeline.py (Mode 2)

**`@dataclass Reference`:**
- `images` (BGR at work size, all registered to `images[0]`);
- `grays` (blurred float32);
- `features` (method → per-image features);
- `glare` (bool[H,W]);
- `work_size` (w,h);
- `scale`, `outline_png`, `capture`.

**`prepare_reference(images_bgr: list, params, methods=("pbp","pixdiff")) -> Reference`.**
- Resizes to `work_max_side` (extra shots then `match_scale` to shot 0) and registers extra shots to shot 0, rejecting non-matching shots with `ValueError`.
- Computes features once. At most `max_references` shots.

**`@dataclass InspectResult`:**
- `verdict` ∈ {"pass","fail","retake"};
- `reason` ∈ {None,"alignment","glare","blurry","dark","low_coverage","mismatch"};
- `zones: list[Zone]` (ranked), `coverage`, `changed_frac`;
- `align: dict(method, n_inliers, inlier_ratio, score)`;
- `timing_ms` (stages: resize, quality, align, mask, features, change, [extra_views], zones, render, total);
- `method`, `threshold`;
- `images: {"frame", "overlay"}` (JPEG bytes);
- `signature_sha256`, `hints`, `work_size (w,h)`;
- `score: float32[gh,gw] | None` and `checked: bool[gh,gw] | None`: the smoothed change map and the checked cells. They are not serialised; eval uses them to re-threshold. They are `None` when the frame was rejected before scoring.
- `bright_spot: bool`: a large glare-like blob was masked (see the bright spot rule); not serialised (the hints carry it); eval passes it to `decide`.

**`inspect_compare(ref, frame_bgr, params, methods=("pbp","pixdiff")) -> dict[str, InspectResult]`.** Aligns ONCE and evaluates every method on the same aligned frame and checked cells:
1. resize;
2. quality: `blurry` if the blur is `< min_blur` AND `< blur_ratio ×` the reference blur; `dark` if the mean is `< dark_mean`;
3. `match_scale(frame, ref.work_size)` (same long side as the reference; exactly its size if the aspect ratio matches within 1 %; upscaling ≤ 2×), then align to `ref.images[0]` (failure → `alignment`);
4. checked = valid ∧ region ∧ ¬border (`border_margin_px`) ∧ no glare in the reference or frame. If `corner_shift(H) > static_bg_min_shift` and `static_background` is on, region = `part_roi(inlier_pts, part_roi_margin_px)` and coverage is relative to it;
   - **bright spot rule:** `bright_spot(core, ...)` is True if the frame's undilated glare (outside the reference's glare mask, inside the region) has a blob of ≥ `glare_blob_min_px` whose reference brightness × the global exposure ratio is below 0.9 × `glare_v` (a white label / tape / clipped bare metal looks exactly like glare and must not end in a silent PASS);
5. glare frac > `max_glare_frac` → retake `glare` + hint `change_angle`; coverage < `min_coverage` → retake `low_coverage`;
6. per method:
   - features of the blurred gray;
   - `multi_reference_change_map` with `tolerance`;
   - PBP only: the low-contrast gate. A cell is set to 0 if `max(ref_contrast, frame_contrast / contrast_gain) < contrast_floor`;
   - PBP only: the exposure flat rule. A cell whose `flat` differs between reference and frame is set to 0 if the larger gain-corrected contrast is `< 2 × contrast_floor` and the smaller is `≥ 0.5 ×` the larger;
   - moved part over a static background (only when align returned `alt_H`): per-cell MIN with the maps of the `alt_H` view and, if `alt_H` itself moves the corners by more than `static_bg_min_shift`, the unregistered frame. A single (camera) motion gets no extra view;
   - `smooth_scores(score, checked, score_smooth)`;
   - `decide(..., bright=bright_spot)`; a resulting `("retake","glare")` carries the hints `change_angle` and `bright_spot`;
   - overlay.

**Other functions:**
- `inspect_frame(ref, frame_bgr, params, method="pbp")`: a single method.
- `decide(score, checked, params, threshold, sig_ref=None, sig_frame=None, bright=False) -> (verdict, reason, zones, changed_frac)`:
  - `changed_frac` = fraction of checked cells with score ≥ threshold;
  - `> mismatch_frac` → `("retake","mismatch",[],changed_frac)`;
  - else `extract_zones` → "fail" if there are zones; else `("retake","glare",[],changed_frac)` if `bright`; else "pass".
  - The PBP signatures, if given, enable the removed/added hints. The benchmark uses `decide` to re-threshold.
- `method_threshold(method, params)`: `threshold` for pbp, `threshold_<method>` for baselines.
- `result_to_json(res) -> dict`: no bytes. See the HTTP API for the shape.

## ava/zones.py

**`extract_zones(score, checked, params, sig_before=None, sig_after=None, win=8, step=4) -> list[Zone]`:**
1. changed = checked ∧ score ≥ `params.threshold`;
2. 8-connected blobs;
3. merge blobs closer than `merge_gap_px`;
4. keep a zone if a fragment has ≥ `min_zone_cells` cells or it is a dense cluster (≥ `blob_min_count` fragments);
5. rank by `n_cells·mean_score`, at most `max_zones`.

**`Zone`:** `id`, `bbox (x,y,w,h)`, `n_cells`, `area_px`, `mean_score`, `max_score`, `rank_score`, `hint`, `contour`.

**Hints:** `water_or_dirt`, `something_removed`, `something_added`.

`coverage(checked)` gives the fraction of grid cells checked.

## ava/surface.py (Mode 1)

**`scan_surface(image_bgr, params, method="pbp") -> InspectResult`.**
- `method` ∈ `SURFACE_METHODS = ("pbp","census","gradient")`. Result `method` = `"pbp_surface"`, `"census_surface"` or `"gradient_surface"`. Deterministic.
- **Stages:**
  1. resize;
  2. `dark` (mean < `dark_mean`), then `blurry` (blur score of the high-pass image < `min_blur`);
  3. checked = ¬border ∧ ¬`glare_region`;
  4. glare frac > `max_glare_frac` → retake `glare` + `change_angle`; coverage < `min_coverage` → retake `low_coverage`;
  5. score;
  6. `smooth_scores(..., surface_smooth)`;
  7. `extract_zones` with the method's threshold;
  8. "fail" if there are zones; else, if a glare core (after the `_GLARE_OPEN_PX` opening, undilated) of ≥ `glare_blob_min_px` was masked, "retake" / "glare" with the hints `change_angle` and `bright_spot` (`bright_spot = True` on the result); else "pass".
- There is no `mismatch` gate.
- `align = {}`, `images = {"frame","overlay"}`, `signature_sha256` = sha256 of the descriptors (PBP: `serialize_signatures`), and `score` / `checked` as in Mode 2.
- Timing stages: resize, quality, mask, score, zones, render, total.

**`surface_score_map(gray, params, method="pbp", checked=None) -> float32[gh,gw]`.** The raw, unsmoothed anomaly score; exposed for eval.

**PBP score:** `clip(knn + surface_contrast_weight · cterm, 0, 1)`.
- `knn` is the mean of the `surface_k` smallest `max(r_a, r_b) · d(a,b)`, where `d` is the `pair_distance` metric.
- The sample is a seeded (`surface_seed`) set of `surface_samples` checked windows, skipping those within `surface_exclude_cells`.
- The candidates are pre-selected by Hellinger affinity (`surface_candidates`) and re-ranked exactly.
- `r` is the noise ramp: `surface_noise_floor..surface_noise_full`, raised to `surface_rel_floor..surface_rel_full` × the 25th-percentile window contrast.
- `cterm` is the robust z of log contrast, ramped from `surface_contrast_z` over `surface_contrast_span`.

**Baselines:**
- `census`: the same kNN, ramp and contrast term on census codes;
- `gradient`: the robust z of the mean Sobel magnitude / 10.

**Other functions:**
- `glare_region(img, gray, params) -> bool[H,W]`: glare cores (thin structures removed) + halo (`surface_glare_elev`, `surface_glare_halo_px`), dilated by `surface_glare_dilate_px`.
- `method_threshold(method, params)`: `surface_threshold` or `threshold_surface_<method>`.

**Known limitation:** the photo must show only the surface. Edges, holes, logos and background are rare too and become zones.

## ava/synth.py

Deterministic given an `np.random.Generator`; no external images, faces, text or plates.

**Parts:**
- `make_part(rng, kind=None, h=720, w=1280)`, with `PART_KINDS = ("stamped","painted","molded","machined")` on `BACKGROUND_KINDS = ("belt","steel_table","esd_mat")`;
- `make_part_layers`, `compose_layers`.

**Defects** each return `(img, mask)`:
- `add_scratch`, `add_tape`, `add_sticker`, `add_dent`, `add_missing_hole`, `add_burr`, `add_stain`, `add_chip`;
- `STATION_DEFECTS` = scratch, tape, sticker, dent, missing_hole, burr, stain;
- `SURFACE_CHECK_DEFECTS` = scratch, chip, stain, sticker, dent, burr.

**Lighting:**
- `apply_factory_lighting(img, rng, kind, strength=1.0)` over `FACTORY_LIGHTING`;
- monotone: drift_gain, gamma, window_gradient, lamp_aging, auto_exposure;
- non-monotone (expected failures): specular, hand_shadow.

**Pairs:**
- `make_station_pair(rng, part_kind=None, defect=None, lighting=None, motion="none"|"jitter"|"shift", h=720, w=1280, strength=1.0) -> dict(reference, frame, mask_ref_frame, meta)`;
- jitter = ±2 px / 0.3°; shift = the part moved by up to 40 px / 3° with the camera fixed;
- independent sensor noise on both images.
- `make_reference_set(rng, base, n, ...)`.

**Surfaces:** `make_uniform_surface(rng, kind="painted", h=720, w=1280)`, with `SURFACE_KINDS = ("painted","plastic","brushed","glass_dark")` filling the whole frame.

**Legacy** car-panel generators (`make_surface`, `apply_lighting`, `make_pair`) are kept for older tests.

## HTTP API (server/app.py) — the frontend relies on this exactly

**General:**
- Responses are JSON. Errors are `{"error": code, "message": text}` with 4xx/5xx.
- **Uploads:** multipart/form-data, at most 15 MB per file, at most 40 MP. Only JPEG (incl. camera MPO), PNG and WebP are decoded (PIL `formats=`); anything else is `bad_image`. JPEGs with a long side above 2048 px are decoded at reduced scale (the draft box keeps the aspect ratio). Decoded with `ImageOps.exif_transpose`; 16-bit gray images are scaled to 8 bits; then converted to BGR. Minimum side 64 px.
- **Body limit:** Content-Length up front, else a streaming count; the 413 is returned also when a middleware wraps the error in an exception group.
- **CPU work** runs in a threadpool, at most 2 jobs at once.
- **No server-side storage of user photos:** images live only in memory for the request.
- **Reference cache:** an in-memory LRU (4 entries) of prepared references, keyed by `(sha256(reference bytes), methods)`.

**Endpoints:**
- `GET /api/health` → `{"ok": true, "version": str, "params": Params.to_dict() + {"surface_threshold"}}`.
- `POST /api/scan`, multipart field `image` → `{"result": result_json, "overlay": base64 JPEG, "image_sha256": str}`.
- `POST /api/compare?methods=pbp,pixdiff`, multipart fields `reference` and `frame` → `{"results": {method: result_json}, "overlays": {method: base64 JPEG}, "reference_sha256": str, "frame_sha256": str}`. `methods` ⊆ {pbp, ssim, census, pixdiff}.

**`result_json`** = `pipeline.result_to_json`:
```
verdict, reason,
zones [{id, bbox [x,y,w,h], bbox_norm [0..1], n_cells, area_px, mean_score, max_score, hint, contour [[x,y],...]}],
coverage, changed_frac, align, timing_ms, method, threshold, signature_sha256, hints, work_size [w,h]
```

**Error codes:**
- `empty_file`, `too_large` (413), `too_many_pixels` (413), `bad_image`, `image_too_small`;
- `multipart_required`, `bad_form`, `file_required`, `bad_methods`, `invalid_request`;
- `not_found`, `method_not_allowed`, `internal` (500).

## Frontend (web/) — vanilla JS, no CDNs, no build

- Mobile-first: iOS Safari and Android Chrome over HTTPS.
- **The UI is in English only**, with no language toggle. The product name is AVA.

**One screen:**
1. **Header.** The AVA wordmark and a QR button (the link to this site, drawn locally by `web/qrcode.js` and `web/qr.js`). Below it, the tagline "Visual defect check without defect training data and without a light box. Photos are not stored." On plain HTTP a note says the live camera needs HTTPS.
2. **Mode switch.** "Surface check" (uniform texture · 1 photo) | "Compare with reference" (complex parts · 2 photos). Each mode shows a one-line "When to use" hint. The **user** chooses the mode; it is remembered in `localStorage` (`ava.mode`).
3. **Surface mode.**
   - One slot, "Surface photo" ("Fill the frame with the surface only"), with Camera and Gallery.
   - Button "Check surface" → `POST /api/scan`.
4. **Compare mode.**
   - Two slots: **Reference** ("a good part") and **Check** ("the part to inspect, same angle").
   - Camera: a full-screen live camera (`getUserMedia`, rear camera, `playsinline muted`). For Check, the reference is overlaid with an opacity slider, stored as `ava.ghostOpacity`.
   - Gallery: `<input type=file accept=image/*>`.
   - Button "Compare" → `POST /api/compare?methods=pbp,pixdiff`.
5. **Result.**
   - Verdict banner: green "PASS — no changes found" (compare) or "PASS — no defects found" (surface), red "FAIL — N defect zones", or grey "RETAKE — <reason>".
   - Reasons: alignment "could not align — shoot from the same angle"; glare "glare — change the angle by 15–30°" (with the hint `bright_spot`: the advice says the spot may be a white label, tape or bare metal and should be checked by eye); blurry; dark; mismatch "different part or part missing"; low_coverage.
   - Zone hints: `water_or_dirt` "looks like dirt or a drop"; `something_removed` / `something_added` are texture-activity hints and are shown neutrally as "surface looks smoother here" / "surface looks rougher here".
   - The overlay image, and a zone list with hints. Tapping a zone highlights it.
   - ms, coverage %, and a short SHA-256 (full on tap).
6. **Plain-subtraction panel** (compare mode only): "Plain reference subtraction", with the `pixdiff` result and overlay.
   - For the baseline only, reason `mismatch` (more than `mismatch_frac` of the checked cells changed) is shown as red "FAIL — a large part of the frame changed", with the measured `changed_frac` and the note that the API reports it as RETAKE: mismatch. Only if AVA passed the same pair: "AVA found no change on the same aligned photo." The baseline's threshold is explained as calibrated by the same rule as AVA's.
7. **After a result.**
   - Buttons: "New check" (keeps the reference), "New reference" or "New photo".
   - Session history of the last 10 results as chips (in-page only, no server).

**Design:**
- industrial, high contrast, touch targets ≥ 48 px;
- light and dark via `prefers-color-scheme`, system fonts;
- no horizontal scroll at 360 px;
- toasts with server error messages.

## Demo scenarios and built-in examples (must always work)

| Mode | Case | Expected |
|---|---|---|
| Surface check | uniform texture WITH a defect | FAIL, defect zone highlighted |
| Surface check | the same texture WITHOUT a defect | PASS, no zones |
| Compare with reference | reference + same part WITH a defect | FAIL, zone on the defect |
| Compare with reference | reference + same part, strong global brightness/lighting change, NO defect | AVA PASS; plain subtraction panel shows FAIL — a large part of the frame changed (API: retake / mismatch) |

**`web/examples/`:**
- `surface_defect.jpg`, `surface_ok.jpg`, `compare_reference.jpg`, `compare_defect.jpg`, `compare_light.jpg`;
- a `README.md` stating the source and the expected results;
- all are synthetic, generated by `eval/make_examples.py` from `ava/synth.py`. The team may replace them with its own photos under the same names.

**`eval/make_examples.py`:**
- regenerates the examples deterministically from a fixed seed list;
- accepts a candidate only if the real pipeline gives the expected verdicts with margins: defect score ≥ 1.35 × threshold, clean score ≤ 0.75 × threshold;
- in the brightness case, plain subtraction must flag ≥ 30 % of the checked area. That case uses exposure ×0.65, gamma 1.3 and a warm cast.

**UI.** Under each mode there is a row "Try an example:" with the buttons "Defect" / "No defect" (surface) or "Defect" / "Brightness change" (compare).
- The buttons only load the example images into the slots, then run the real API.
- Results are NEVER hard-coded.

**Test.** `tests/test_demo_examples.py` runs the real pipeline on the five files and asserts the four verdicts, including the plain-subtraction false FAIL / mismatch in the brightness case. **Do not deploy if it fails.**

## Evaluation (eval/)

**Shared rules:**
- The CALIB and TEST splits use disjoint seeds.
- Thresholds are calibrated on CALIB only, frozen in `ava/config.py`, then reported on TEST.
- Intervals are Wilson 95 %.
- Every generated report is labelled synthetic. No number may appear in the README or the pitch unless it comes from these files.

**Per mode:**
- **Mode 2.** `eval/benchmark.py` compares pbp, census, ssim and pixdiff through `inspect_compare` and `decide`. It writes `eval/results/*.json` and `eval/RESULTS.md`.
- **Mode 1.** `eval/benchmark_surface.py` compares pbp, census and gradient through `scan_surface`. It writes `eval/results/surface_*.json` and `eval/RESULTS_SURFACE.md`.
