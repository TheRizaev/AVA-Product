# AVA — technical contract (authoritative)

Product name in all user-facing text (UI, reports, README): **AVA**. The Python package is still `dalil` (internal).

Product: compare TWO photos — a reference photo of a good part and a photo of the part to check. The second photo is
aligned to the reference and compared with PBP window signatures → verdict PASS / FAIL (+ defect zones) / RETAKE.
No stations, no journals, no accounts: one screen, two photos, one result (shown next to plain reference subtraction).
Differentiator to prove with numbers: robustness to (monotone) lighting drift without a light box, compared with
plain reference subtraction (pixdiff), SSIM and Census on the same pipeline.

Every module owner implements exactly the public API below. Internal helpers are free.
Images: `H×W` gray float32 0..255 or `H×W×3` BGR uint8 (OpenCV order). All tunables live in `dalil/config.py::Params`
(add fields with defaults if needed; never rename/remove). Terminology in code: `reference` (golden, was "before")
and `frame` (inspected, was "after"). Existing modules keep their parameter names (before/after) — that's fine.

## Repository layout
```
dalil/
  config.py      Params (exists)
  pbp.py         EXISTS: pbp_reference, format_polynomial, poly_degree, Signatures, window_signatures,
                 pair_distance, serialize_signatures, grid_shape, cell_mask_from_pixel_mask
  change.py      NEW: tolerant change map (generic over dist_fn), multi-reference min
  zones.py       EXISTS: Zone, extract_zones, coverage (hint codes: water_or_dirt, something_removed, something_added)
  align.py       EXISTS: AlignResult, align (ORB+RANSAC); ADD: fallback for low-texture parts + fast path for a fixed camera
  quality.py     EXISTS: glare_mask, blur_score, is_too_dark, capture_report, outline_png
  baselines.py   EXISTS: ssim/census/pixdiff features+distances, get_methods()
  synth.py       EXISTS (car-panel surfaces, defects, lighting); ADD: industrial parts + factory lighting drift
  pipeline.py    NEW: Reference (prepared golden set), prepare_reference(), inspect_frame(), result_to_json()
  (report.py removed)
server/
  app.py         FastAPI: API under /api, static web/ at /
  (store.py removed)
web/             vanilla JS mobile-first UI (index.html, app.js, style.css, i18n.js, qr.js)
eval/benchmark.py synthetic (+ VisA real, if downloaded) evaluation → eval/results/, eval/RESULTS.md
tests/
Dockerfile       HF Spaces (port 7860)
```
`dalil/act.py` and `tests/test_act.py` are DELETED (superseded by report.py).

## Window grid (shared)
Windows `win×win` (8) at `y=i*step, x=j*step`, `gh=(H-win)//step+1`, `gw=(W-win)//step+1`. Per-window arrays are `[gh,gw]`.

## dalil/pbp.py (exists — verify, test, keep API)
PBP of an m×n patch: per column, for each distinct value t > min, add `(t - prev(t)) * Π_{i: C[i,j] < t} y_i`;
constant = Σ column minima; merge like monomials. `window_signatures` computes 2 orientations (rows; transposed),
ε-ties (`C[r] < C[i] - eps*contrast`), normalised 256-bin coefficient vectors, `flat` windows. `pair_distance(a,b,dy,dx)`
= mean over orientations of 0.5·L1 (∈[0,1]), out-of-range → +inf, flat-vs-flat 0, flat-vs-nonflat 1.

## dalil/change.py (new)
- `tolerant_change_map(dist_fn, a, b, tol) -> float32[gh,gw]`: `d_ab = min_{|dy|,|dx|≤tol} dist_fn(a,b,dy,dx)`,
  `d_ba = min dist_fn(b,a,dy,dx)`, result `max(d_ab, d_ba)` (inf → 1.0 clamp for PBP-type distances; keep finite).
- `multi_reference_change_map(dist_fn, refs: list, frame, tol) -> float32[gh,gw]`: element-wise MIN over references of
  `tolerant_change_map(dist_fn, ref, frame, tol)` (natural part-to-part variation is absorbed by having several good shots).
- `pbp_change_map(sig_ref, sig_frame, params)` convenience.

## dalil/align.py (exists; extend)
`align(before_bgr, after_bgr, params) -> AlignResult` (H maps AFTER(frame) → BEFORE(reference) coords).
Add: (1) fast path — if the frame is already registered (fixed camera, part not moved: e.g. normalised cross-correlation of
downscaled CLAHE-gray images ≥ params.fixed_cam_ncc), return identity with ok=True and `method="identity"`;
(2) ORB+RANSAC as now (`method="orb"`); (3) fallback for low-texture parts: `cv2.findTransformECC` (MOTION_EUCLIDEAN, on
pyramid-downscaled CLAHE-gray, gaussFiltSize 5) and/or phase correlation for pure translation (`method="ecc"`), accept only
if the final ECC correlation ≥ params.ecc_min_corr. Add field `method: str` to AlignResult (default "orb").
Also (integration): `score` (identity NCC / ORB inlier ratio / ECC corr); `inlier_pts` (ORB: reference-frame points of H's
inliers); `alt_H` — a second motion fitted on ORB's outliers (part moved over a static belt: one motion is the part, one
the background). If the dominant motion is ≈ identity and the second is not, the second is taken as `H` (fixed camera:
the static thing is the background). `corner_shift(H, shape)` helper. The warp uses a replicated border (validity is
defined by `valid` only).

## dalil/synth.py (exists; extend, keep existing API and tests)
Add, deterministic with rng, no external images:
- `make_part(rng, kind=None, h=720, w=1280) -> BGR`: kinds `"stamped"` (metal plate, brushed texture, holes, bends/edges),
  `"painted"` (painted panel piece with seams, orange-peel texture), `"molded"` (dark plastic with ribs, bosses, grain),
  `"machined"` (round flange with concentric grooves, bolt holes). Background = conveyor belt / table texture.
  Part occupies 40–80 % of the frame.
- Defects: reuse add_scratch/add_tape/add_sticker; add `add_dent(img, rng)` (soft shading ring), `add_missing_hole(img, rng)`
  (fill a hole / remove a feature), `add_burr(img, rng)` (small extra bright/dark blob at an edge), `add_stain(img, rng)` (oil/
  water stain). Each returns `(img, mask)`.
- `FACTORY_LIGHTING = ("drift_gain","gamma","window_gradient","lamp_aging","auto_exposure","specular","hand_shadow")`;
  `apply_factory_lighting(img, rng, kind, strength=1.0)`. Monotone ones: drift_gain, gamma, auto_exposure, lamp_aging (global
  colour/gain), window_gradient (smooth spatial gain — locally monotone); non-monotone: specular (saturating moving highlight),
  hand_shadow (hard shadow) — these are expected failure cases and must be reported as such.
- `make_station_pair(rng, part_kind=None, defect=None, lighting=None, motion="none"|"jitter"|"shift") -> dict(reference,
  frame, mask_ref_frame, meta)`: jitter = ±2 px / 0.3°, shift = part moved up to 40 px / 3° on the belt (camera fixed).
  Independent sensor noise on both. `make_reference_set(rng, base, n)` → n good shots with tiny jitter/noise.

## dalil/pipeline.py (new)
- `@dataclass Reference`: `images: list[BGR]` (work size, all registered to images[0]), `grays: list[float32]`,
  `features: dict[str, list]` (method → per-image features), `glare: bool[H,W]`, `work_size: (w,h)`, `scale: float`,
  `outline_png: bytes`, `capture: dict`.
- `prepare_reference(images_bgr: list, params, methods=("pbp","pixdiff")) -> Reference` (resize to params.work_max_side;
  align extra shots to the first; compute features for the requested methods once — cached by the server).
- `@dataclass InspectResult`: `verdict` ∈ {"pass","fail","retake"}, `reason` ∈ {None,"alignment","glare","blurry","dark",
  "low_coverage","mismatch"} ("mismatch" = changed area > params.mismatch_frac → wrong/absent part), `zones: list[Zone]`,
  `coverage`, `changed_frac`, `align: dict(method, n_inliers, inlier_ratio)`, `timing_ms: dict`, `method: str`,
  `threshold: float`, `images: dict[str, bytes]` ("frame" JPEG aligned frame, "overlay" JPEG: zones red outline + rank,
  unchecked cells grey-hatched; for baselines the overlay is the same style), `signature_sha256`, `hints: list[str]`.
- `inspect_frame(ref: Reference, frame_bgr, params, method="pbp") -> InspectResult`: resize → quality(blur/dark → retake)
  → align to ref.images[0] → gray+blur → features → `multi_reference_change_map` → checked = valid & ~glare(ref|frame)
  → glare frac > params.max_glare_frac → retake "glare" + hint "change_angle"; coverage < params.min_coverage → retake
  → zones with the method's threshold (`params.threshold` for pbp, `params.threshold_<method>` for baselines) →
  mismatch check → verdict fail if any zone else pass.
- `inspect_compare(ref, frame_bgr, params, methods=("pbp","pixdiff")) -> dict[str, InspectResult]` — runs alignment ONCE
  and evaluates several methods on the same aligned frame (used by the UI side-by-side and by eval).
- `result_to_json(res) -> dict` (no bytes; zones with `bbox`, `bbox_norm` 0..1).
- Integration details (all tunables in Params): blur gate is relative too (`blur < min_blur` AND `< blur_ratio ×`
  reference blur — a smooth part is "soft" in the reference as well); PBP low-contrast gate (cells whose reference and
  gain-normalised frame contrast are both `< contrast_floor` count as unchanged — sensor noise sets the pixel order there);
  `score_smooth`×`score_smooth` normalised box mean of the change map over checked cells before zones (all methods);
  `border_margin_px` of the image border is not checked. When the registration moved the image by more than
  `static_bg_min_shift` px (`static_background`): only the hull of H's ORB inliers (+`part_roi_margin_px`) is checked
  (coverage is relative to it), and the change map is the per-cell MIN over the part-registered view, the
  background-registered view (`alt_H`) and/or the frame as shot (camera-fixed view). `InspectResult.work_size` (w, h)
  is added for `bbox_norm`. Timing stages: resize, quality, align, mask, features, change, [extra_views], zones, render, total.

## Removed (product simplified to "compare two photos")
`dalil/report.py`, `server/store.py`, all `/api/stations/...` and `/api/files/...` routes, the station/viewer/journal
screens and their tests are deleted. `pipeline.prepare_reference` keeps accepting a list but the UI sends one reference.

## HTTP API (server/app.py) — the frontend relies on this exactly
JSON; errors `{"error": code, "message": text}` with 4xx/5xx. Upload limit 15 MB per file; decode via PIL with
`ImageOps.exif_transpose`, then BGR. CPU work in a threadpool. No server-side storage of user photos (privacy): images
live only in memory for the request. An in-memory LRU (≤ 4 entries) caches `prepare_reference` results keyed by
sha256(reference bytes) so repeated checks against the same reference are faster.
- `GET  /api/health` → `{"ok": true, "version": str, "params": {"threshold", "threshold_pixdiff", ...}}`
- `POST /api/compare?methods=pbp,pixdiff` multipart fields `reference`, `frame` →
  `{"results": {method: result_json}, "overlays": {method: base64 JPEG}, "reference_sha256": str, "frame_sha256": str}`
  result_json = `pipeline.result_to_json` (verdict, reason, zones[{id,bbox,bbox_norm,n_cells,area_px,mean_score,
  max_score,rank_score,hint}], coverage, changed_frac, timing_ms, align, method, threshold, signature_sha256, hints,
  work_size).

## Frontend (web/) — vanilla JS, no CDNs, no build
Mobile-first; iOS Safari + Android Chrome over HTTPS. **UI language: ENGLISH ONLY** — no Russian strings, no language toggle (the Russian labels quoted below are the meaning; render them in English: Reference / Check / Compare / PASS — no changes / FAIL — N zones / RETAKE: <reason> / Plain reference subtraction / New check / New reference).
Product name everywhere: **AVA**. ONE main screen:
1. Header: wordmark "AVA", one line: "Сравнение с эталоном: показываем только то, что изменилось. Без обучения на
   дефектах, без светового бокса." Language toggle. Small privacy note: "Фото не сохраняются на сервере."
2. Two slots side by side (stacked on narrow screens): **«Эталон»** (good part) and **«Проверка»** (part to check).
   Each slot: "Камера" (opens full-screen live camera: getUserMedia rear, playsinline muted, object-fit contain; for
   the «Проверка» slot the reference photo is overlaid at ~40 % opacity with a slider so the user matches the angle;
   shutter grabs full-res frame → JPEG 0.9) and "Из галереи" (`<input type=file accept=image/*>`). Thumbnail preview,
   "Заменить". Stop camera tracks when closing.
3. Big button **«Сравнить»** (enabled when both photos exist) → POST /api/compare (methods=pbp,pixdiff) with spinner.
4. Result: verdict banner — green «ГОДНА — изменений нет», red «БРАК — найдено зон: N», grey «ПЕРЕСНЯТЬ: <reason>»
   (alignment → «не удалось совместить: снимите с того же ракурса»; glare → «блик: смените угол на 15–30°»;
   blurry/dark → «размыто/темно»; mismatch → «не та деталь или деталь отсутствует»; low_coverage → «мало
   проверенной площади»). Overlay image (pbp) with zones; tapping a zone in the list highlights it. Zone hints:
   water_or_dirt → «похоже на загрязнение/каплю», something_removed → «отсутствует элемент?», something_added →
   «лишний элемент/наплыв?». Small: processing ms, coverage %, short SHA-256 fingerprint (signature_sha256, full on tap).
5. Side panel/collapsible «Простое вычитание из эталона» (pixdiff result + overlay) — the key demo moment. Display
   rule for the BASELINE only: reason "mismatch" means plain subtraction saw almost the whole frame as changed — show it
   as red «БРАК — изменился почти весь кадр» with a one-line explanation («так простое вычитание реагирует на смену
   света»). Never call this panel "Cognex".
6. After a result: buttons «Новая проверка» (keeps the reference, clears «Проверка» and reopens its camera) and
   «Новый эталон». Keep a small in-page history (this session only, no server) of the last 10 results as chips
   (verdict + time) — tap to view again.
Design: industrial, high contrast, large touch targets (≥ 48 px), readable in bright light, light + dark via
prefers-color-scheme, system fonts, no horizontal scroll at 360 px, toasts with server error messages.

## Mode 1 — single-photo surface check: dalil/surface.py (new)
The site offers TWO modes: (1) **Surface check** — one photo of a uniform surface (painted panel, plastic, sheet metal,
glass), find local defects with no reference; (2) **Compare with reference** — two photos (sections above).
Method (self-referenced PBP anomaly): the surface itself is the reference. Compute `window_signatures` of the photo;
for every window, the anomaly score is the mean of the k smallest `pair`-type distances (same metric as
`pbp.pair_distance`: 0.5·L1 of normalised coefficients, averaged over orientations; flat-vs-flat 0, flat-vs-nonflat 1)
to a deterministic random sample of `params.surface_samples` windows of the SAME image, excluding samples within
`params.surface_exclude_cells` cells of the window. Rationale: on a uniform surface normal windows have many look-alikes;
a scratch/chip/stain/dent edge is rare. Signatures are invariant to monotone brightness changes, so uneven lighting does
not create look-alike failures. Add a contrast term: windows whose contrast is far above the image's robust (median/MAD)
contrast distribution score higher (gain-normalised, so global exposure does not matter). Smooth, glare-mask, border
margin and zones exactly like the reference pipeline (reuse zones.extract_zones, quality.glare_mask, etc.).
- `scan_surface(image_bgr, params) -> InspectResult` (reuse `dalil.pipeline.InspectResult`; `method="pbp_surface"`;
  verdict pass / fail / retake with reasons glare / blurry / dark / low_coverage; `images["overlay"]` JPEG with zones;
  `align` = {} ; `signature_sha256` = sha256(serialize_signatures(sig))). Must be deterministic (seeded sampling).
- `surface_score_map(gray, params, method="pbp") -> float32[gh,gw]` (exposed for eval); baselines for eval:
  `method="gradient"` (Sobel gradient magnitude per window, robust z-score vs image median/MAD) and `method="census"`
  (same kNN self-reference on census features).
- Params (add): `surface_threshold`, `surface_samples` (≈768), `surface_k` (≈5), `surface_exclude_cells` (≈6),
  `surface_contrast_weight`, `threshold_surface_gradient`, `threshold_surface_census`.
- Target: ≤ 1.5 s per 1024-px photo on the laptop CPU.
- Synthetic data (dalil/synth.py): `make_uniform_surface(rng, kind, h=720, w=1280)` kinds "painted" (smooth colour +
  orange peel), "plastic" (fine grain), "brushed" (directional brushed metal), "glass_dark" (dark glossy with faint
  reflections) filling the WHOLE frame; defects reuse add_scratch / add_dent / add_stain / add_burr / add_sticker + a
  new `add_chip` (small paint chip exposing primer); lighting via apply_factory_lighting.
- Eval: eval/benchmark_surface.py — calib/test split on disjoint seeds; per method (pbp, census, gradient) and per
  lighting kind: recall, false-alarm rate on defect-free photos, false zones per photo, ms per photo; writes
  eval/results/surface_*.json and eval/RESULTS_SURFACE.md (generated, labelled synthetic, Wilson 95 % CIs).
- Known limitations to state: the photo must show only the surface (part edges, holes, logos and background are
  "rare" too → false zones); strong regular patterns are fine, unique features are not.

## HTTP API addition for Mode 1
- `POST /api/scan` multipart field `image` → `{"result": result_json, "overlay": base64 JPEG, "image_sha256": str}`.
UI: a mode switch at the top — "Surface check (1 photo)" | "Compare with reference (2 photos)". Surface mode: one
slot "Photo of the surface" (camera/gallery) + hint "Fill the frame with the surface only", button "Check surface",
same verdict banner / overlay / zone list / timing / SHA as compare mode (no subtraction panel).
