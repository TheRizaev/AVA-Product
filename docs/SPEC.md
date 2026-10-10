# Dalil QA — technical contract (authoritative)

Product: a reference-based visual QA station. A camera looks at a part; the operator stores 1–5 shots of a good part
(the "golden reference"); every new frame is aligned to the reference and compared with PBP window signatures →
verdict PASS / FAIL (+ defect zones) / RETAKE, logged in a hash-chained shift journal.
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
  report.py      NEW (replaces act.py): hash-chained journal records, CSV export, shift report HTML
server/
  app.py         FastAPI: API under /api, static web/ at /
  store.py       on-disk station storage (DALIL_DATA_DIR, default ./_data) + in-memory reference cache
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

## dalil/report.py (new)
- `canonical_json(obj) -> str` (sort_keys, separators=(',',':'), ensure_ascii=False).
- `record_hash(prev_hash: str, record: dict, blobs: list[bytes]) -> str` = sha256(prev_hash + canonical_json(record without
  "hash"/"prev_hash"/"review") + sha256 of each blob). Genesis prev_hash = "0"*64. Reviews are appended as separate events
  (their own chained records), so editing history is detectable. Docstring states exactly what is hashed.
- `verify_chain(records: list[dict], blobs_for: callable) -> (ok: bool, first_bad_seq: int|None)`.
- `journal_csv(station: dict, records: list[dict]) -> str` (seq, time_local, verdict, reason, n_zones, top_zone_bbox,
  review, method, timing_ms_total, hash).
- `render_shift_report(station: dict, records: list[dict], images: dict[str,str]) -> str` — standalone printable HTML (RU),
  A4 print CSS, no external resources: header (station name, SKU, shift start/end in Asia/Tashkent + UTC), summary (total,
  pass, fail, retake, operator-confirmed defects, operator-marked false alarms), table of records (thumbnail of overlay for
  fail rows), chain head hash + "цепочка проверена/нарушена", method note: "Сравнение с эталоном методом PBP (псевдобулевы
  полиномы, Chikake & Goldengorin). Решение подтверждает оператор." Escape all user text.

## HTTP API (server/app.py) — the frontend relies on this exactly
JSON; errors `{"error": code, "message": text}` 4xx. Upload limit 15 MB; decode via PIL with `ImageOps.exif_transpose`
then to BGR. CPU work in `run_in_threadpool`. Station ids: 6 chars `[A-Z2-9]` (easy to type; validate strictly).
- `GET  /api/health` → `{"ok":true,"version":str,"params":{threshold, threshold_pixdiff, ...}}`
- `POST /api/stations` body `{"name": str, "sku": str?}` → station JSON
- `GET  /api/stations/{sid}` → station JSON
- `POST /api/stations/{sid}/reference` multipart `file` → station JSON (adds a golden shot; max 5; returns capture report
  for it in `last_capture`); `DELETE /api/stations/{sid}/reference` → station JSON (clears golden set and cache)
- `POST /api/stations/{sid}/inspect?compare=1` multipart `file` → inspection record JSON (when compare=1, the record has
  `baseline` = {method:"pixdiff", verdict, n_zones, overlay_url})
- `GET  /api/stations/{sid}/inspections?after=<seq>&limit=50` → `{"items":[record...], "last_seq": int}` (newest last;
  used by viewers polling every ~1.5 s)
- `POST /api/stations/{sid}/inspections/{seq}/review` body `{"label":"defect"|"false_alarm"|"ok"}` → record JSON
- `GET  /api/stations/{sid}/journal.csv` → text/csv; `GET /api/stations/{sid}/report` → text/html shift report
- `GET  /api/files/{sid}/{name}` → stored images (strict regex whitelist, no traversal)
- `POST /api/compare?methods=pbp,pixdiff` multipart `reference`, `frame` (stateless) → `{"results": {method: result_json},
  "overlays": {method: base64 jpeg}}`
Station JSON: `{"id","name","sku","created_at","references":[{"idx","url","at"}],"outline_url":str|null,
"stats":{"total","pass","fail","retake","confirmed_defects","false_alarms"},"chain_head":str,"last_seq":int}`.
Inspection record JSON: `{"seq","at","verdict","reason","zones":[...],"coverage","changed_frac","timing_ms","align",
"method","frame_url","overlay_url","baseline":{...}|null,"review":null|label,"hash","prev_hash"}`.

## Frontend (web/) — vanilla JS, no CDNs, no build
Mobile-first; iOS Safari + Android Chrome over HTTPS; RU default + EN toggle (i18n.js). Hash routing.
1. Home: wordmark "Dalil QA", value line ("Контроль деталей по эталону — без обучения на дефектах, без светового бокса"),
   buttons: "Новая станция", "Подключиться к станции" (enter 6-char code), "Сравнить два фото".
2. Station setup (#/s/<sid>/setup): name/SKU; camera preview; "Снять эталон" (1–5 shots, thumbnails, clear); hint
   "снимите годную деталь; можно 2–3 кадра с небольшим сдвигом"; button "Начать контроль".
3. Operator (#/s/<sid>/run): live camera (getUserMedia rear, playsinline, object-fit contain) with the reference outline
   overlay (outline_url, opacity slider) for positioning; big "Проверить" button + "Авто" toggle (capture every N s,
   N selectable 2/3/5; skip if previous request still running); latest verdict banner: green "ГОДНА", red "БРАК — N зон",
   grey "ПЕРЕСНЯТЬ: <reason>" (+ hint "смените угол на 15–30°" for glare); overlay image with zones; a "Сравнение с простым
   вычитанием" toggle showing the baseline verdict + overlay side by side (this is the key demo moment); QR code (generated
   locally by web/qr.js — implement a compact QR encoder (byte mode, ECC M, versions 1–10) or port a MIT one WITH attribution
   header) linking to the viewer URL; station code shown large. Fallback without camera: file input (capture=environment).
4. Viewer (#/s/<sid>/view): for judges' phones — polls inspections, shows the latest verdict big + overlay + baseline
   comparison, the journal list, and review buttons "Дефект подтверждён" / "Ложная тревога".
5. Journal (#/s/<sid>/journal): stats tiles, list with thumbnails, review labels, "Скачать CSV", "Отчёт смены" (opens report).
6. Quick compare (#/compare): two file inputs → /api/compare → both methods' verdicts and overlays.
Design: industrial, high contrast, large touch targets (≥48 px), readable in a bright shop; CSS variables, light + dark;
system fonts; no horizontal scroll at 360 px; spinners/disabled states; toasts with server error messages.
