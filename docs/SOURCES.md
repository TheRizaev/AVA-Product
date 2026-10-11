# Sources, licences and verified facts

**How the sources were checked:**
- Titles, authors and pages of every DOI below were checked against the Crossref API (`api.crossref.org/works/<doi>`) on 2026-10-10.
- A web fact is marked **VERIFIED** only when the quoted text was read on the linked page.
- Search-engine snippets that could not be opened are marked **UNVERIFIED** and must not go on slides.

## 1. The PBP method (Chikake, Goldengorin et al.)

| # | Reference | DOI | What we take from it |
|---|---|---|---|
| P1 | T. M. Chikake, B. I. Goldengorin, P. M. Pardalos. *Pseudo-Boolean Polynomial Method for Interpretable Dimensionality Reduction: A Paradigm Shift from Abstract to Meaningful Feature Extraction.* **Computer Optics** 49(6), 2025, pp. 1191–1201. | [10.18287/COJ1815](https://doi.org/10.18287/COJ1815) | • The construction of the PBP: sort, consecutive differences, cumulative OR, merging of like monomials.<br>• Training-free and deterministic.<br>• The example polynomial `11 + 11y3 + 3y4 + 2y1y3 + 4y2y3 + 4y1y4 + 12y1y2y3 + 6y1y2y4`, which `tests/test_pbp.py` reproduces. |
| P2 | T. M. Chikake, B. I. Goldengorin. *Image edge detection using pseudo-Boolean polynomials.* Sixteenth International Conference on Machine Vision (**ICMV 2023**), Proc. SPIE 13072, 130720O (published 2024). | [10.1117/12.3023452](https://doi.org/10.1117/12.3023452) | PBP on image patches. |
| P3 | T. M. Chikake, B. Goldengorin, A. Samosyuk. *Pseudo-Boolean Polynomials Approach to Edge Detection and Image Segmentation.* In: *Data Analysis and Optimization*, Springer Optimization and Its Applications, **Springer 2023**, pp. 73–87. | [10.1007/978-3-031-31654-8_5](https://doi.org/10.1007/978-3-031-31654-8_5) | PBP edge detection and segmentation on images. |
| P4 | T. M. Chikake, B. I. Goldengorin, P. M. Pardalos. *Applications of Pseudo-Boolean Polynomials in Medical Imaging.* In: *Trends in Biomathematics* (**BIOMAT** 2024 lectures), **Springer 2025**, pp. 39–60. | [10.1007/978-3-031-97461-8_3](https://doi.org/10.1007/978-3-031-97461-8_3) | A worked 4×4 example in which near-uniform patches (values 136–142) already reach the maximum degree (book p. 53). This is why AVA does **not** use the degree as the defect signal. |
| P5 | T. M. Chikake, B. Goldengorin. *Dimensionality Reduction Using Pseudo-Boolean Polynomials for Cluster Analysis.* In: *Data Analysis and Optimization*, Springer 2023, pp. 59–72. | [10.1007/978-3-031-31654-8_4](https://doi.org/10.1007/978-3-031-31654-8_4) | Background: coefficient vectors as features, and the equivalence property. |
| P6 | T. M. Chikake. *Pseudo-Boolean Polynomial approach to solving Computer Vision tasks.* PhD dissertation, MIPT, 2025; supervisor B. I. Goldengorin. | — | Context only. The full Python code of the method is printed in its Appendix B. Timings reported there are the authors' measurements of a different task and are never quoted as ours. |

### Software by the authors

[`tmc-pbp`](https://github.com/Tenfleques/tmc-pbp) 0.1.0 on PyPI, by Tendai Mapungwana Chikake, MIT licence (checked in the wheel's METADATA).
- It is **not** a dependency. `ava/pbp.py` is an independent implementation.
- **Our own parts:** the 8×8 sorting network, ε-ties relative to window contrast, 256-bin normalised signatures in two orientations, the distance, the tolerant multi-reference comparison and the self-referenced surface score (`ava/surface.py`).

### Our own derivations

None of the papers states these. They are tested in `tests/test_pbp.py` and measured in the benchmarks:
- The monomial *set* is invariant to strictly monotone intensity maps.
- AVA's *normalised* signature is invariant to a positive affine `a·x + b`, except for windows whose contrast crosses the absolute flat floor (6 gray levels): they become "flat". Mode 2 ignores such a flat/non-flat disagreement when the two windows' gain-corrected contrasts agree (`pipeline._change_map`). Under gamma-like changes the coefficients shift slightly.
- PBP signatures can be used for reference-based change detection and for a single-photo surface anomaly score.

## 2. Algorithms used in the pipeline

| Algorithm | Reference | DOI | Where |
|---|---|---|---|
| ORB features | E. Rublee, V. Rabaud, K. Konolige, G. Bradski. *ORB: An efficient alternative to SIFT or SURF.* ICCV 2011, pp. 2564–2571. | [10.1109/ICCV.2011.6126544](https://doi.org/10.1109/ICCV.2011.6126544) | `ava/align.py` |
| Ratio test | D. G. Lowe. *Distinctive Image Features from Scale-Invariant Keypoints.* IJCV 60, 2004, pp. 91–110. | [10.1023/B:VISI.0000029664.99615.94](https://doi.org/10.1023/B:VISI.0000029664.99615.94) | `ava/align.py::_match` |
| RANSAC | M. A. Fischler, R. C. Bolles. *Random sample consensus.* Communications of the ACM 24(6), 1981, pp. 381–395. | [10.1145/358669.358692](https://doi.org/10.1145/358669.358692) | homography and similarity fits (`cv2.findHomography`, `cv2.estimateAffinePartial2D`) |
| ECC alignment | G. D. Evangelidis, E. Z. Psarakis. *Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization.* IEEE TPAMI 30(10), 2008, pp. 1858–1865. | [10.1109/TPAMI.2008.113](https://doi.org/10.1109/TPAMI.2008.113) | the low-texture fallback and the fixed-camera check (`cv2.findTransformECC`) |
| SSIM (baseline) | Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli. *Image quality assessment: from error visibility to structural similarity.* IEEE TIP 13(4), 2004, pp. 600–612. | [10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861) | `ava/baselines.py` |
| Census transform (baseline) | R. Zabih, J. Woodfill. *Non-parametric local transforms for computing visual correspondence.* ECCV '94, LNCS 801, pp. 151–158. | [10.1007/BFb0028345](https://doi.org/10.1007/BFb0028345) | `ava/baselines.py`, `ava/surface.py` (Census kNN) |

**Not counted as references:**
- Plain reference subtraction (`pixdiff`) is the textbook baseline: the mean absolute pixel difference per window, divided by 255.
- The gradient baseline of Mode 1 is a robust z-score of the mean Sobel magnitude.

**Related work, not used or benchmarked:** K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, P. Gehler. *Towards Total Recall in Industrial Anomaly Detection* (PatchCore). CVPR 2022, pp. 14298–14308. [10.1109/CVPR52688.2022.01392](https://doi.org/10.1109/CVPR52688.2022.01392).

## 3. Software licences

The versions below are the exact pins of `requirements.txt` and `requirements-dev.txt`; the Docker image installs exactly these. The full test suite passes on them. The benchmarks ran with the same numeric stack (numpy 2.2.6, OpenCV 4.12.0; the development machine has the `opencv-python` 4.12.0.88 wheel, the same OpenCV build as `opencv-python-headless` 4.12.0.88 without the GUI modules). Licences were read from `pip show` and the package classifiers on 2026-10-10.

| Package | Pinned version | Licence |
|---|---|---|
| numpy | 2.2.6 | BSD-3-Clause |
| opencv-python-headless | 4.12.0.88 | Apache-2.0. The wheels bundle third-party libraries under their own licences. |
| fastapi | 0.116.2 | MIT |
| starlette (via FastAPI) | 0.47.3 | BSD-3-Clause |
| anyio (via Starlette) | 4.15.1 | MIT |
| uvicorn[standard] | 0.35.0 | BSD-3-Clause |
| Pillow | 12.3.0 | MIT-CMU (HPND) |
| python-multipart | 0.0.20 | Apache-2.0 |
| pytest (dev) | 7.4.4 | MIT |
| httpx (dev) | 0.28.1 | BSD-3-Clause |
| qrcode-generator (vendored as `web/qrcode.js`) | — | MIT, © 2009 Kazuhiko Arase, <https://github.com/kazuhikoarase/qrcode-generator>. "QR Code" is a registered trademark of DENSO WAVE INCORPORATED. |
| tmc-pbp (cited, not used) | 0.1.0 | MIT |

## 4. Datasets

| Dataset | Licence | Status |
|---|---|---|
| **Synthetic**, from `ava/synth.py` | Our code (MIT) | **Used** for calibration, benchmarks, tests and the demo examples (`eval/make_examples.py`). Procedural, seeded, no external images; no faces, text or plates. See the details below the table. |
| **VisA** (Amazon). Y. Zou, J. Jeong, L. Pemula, D. Zhang, O. Dabeer. *SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation.* ECCV 2022, [10.1007/978-3-031-20056-4_23](https://doi.org/10.1007/978-3-031-20056-4_23). 10,821 images, 12 classes. | **CC BY 4.0**: commercial use allowed with attribution. VERIFIED: <https://registry.opendata.aws/visa/> | **Not used.** A download was attempted on 2026-10-10 and abandoned; the measured speed would have taken about 10 hours (eval/RESULTS.md). |
| **Team photos** (`data/pairs/`) | Team AVA, CC BY 4.0 | Planned real test set ([protocol](../data/pairs/README.md)). Own objects only, no faces or plates. No results yet. |
| MVTec AD | CC BY-NC-SA 4.0 (non-commercial). VERIFIED: <https://www.mvtec.com/company/research/datasets/mvtec-ad> | **Not used** |
| MVTec LOCO AD | CC BY-NC-SA 4.0. VERIFIED: <https://www.mvtec.com/company/research/datasets/mvtec-loco> | **Not used** |
| KolektorSDD / KolektorSDD2 | CC BY-NC-SA 4.0. VERIFIED: <https://www.vicos.si/resources/kolektorsdd/>, <https://www.vicos.si/resources/kolektorsdd2/> | **Not used** |
| CarDD (USTC) | Non-commercial research and education only (from snippets of the terms; a licensing form is required): <https://cardd-ustc.github.io/> | **Not used** |

**What the synthetic set contains:**
- **Parts:** stamped, painted, molded and machined, on belt, steel-table and ESD-mat backgrounds.
- **Uniform surfaces:** painted, plastic, brushed and glass_dark.
- **Defects:** scratch, tape, sticker, dent, missing hole, burr, stain and chip.
- **Shop lighting:**
  - monotone: gain drift, gamma, window gradient, lamp ageing, auto-exposure;
  - non-monotone (the expected failures): specular, hand shadow.
- **Motion:** camera jitter, and the part moved on the belt.

## 5. Market and competitor facts

Status as of 2026-10-10. Quote these only with their URL.

### Golden-template and vision-sensor incumbents

**Cognex — Golden Template Comparison (GTC).** VERIFIED.
- Method: a template (the mean of good samples) is registered to each image and subtracted, the result is thresholded, and blobs are checked. In-Sight has a CompareImage tool.
  - <https://support.cognex.com/docs/is_574/web/EN/ise/Content/Reference/CompareImage.htm>
  - <https://www.cognex.com/en/products/machine-vision-software/visionpro-software>
- **For fairness:** Cognex PatInspect normalises run-time images *globally* (histogram equalisation, mean/std, tails matching, or identity).
  - <https://docs.cognex.com/cvl_900/web/EN/cvl_vision_tools/Content/Topics/VisionTools/Normalizing_Run_Time_Ima.htm>
  - Our claim is therefore **not** that incumbents ignore lighting. It is that one global intensity map cannot correct a *spatially varying* change, while per-window signatures normalise each window on its own.

**Keyence IV3 AI vision sensor.** VERIFIED.
- Does "presence and difference checking".
- Is set up by registering "a minimum of one OK and one NG image".
- <https://www.keyence.com/products/vision/vision-sensor/iv3/>

### Vehicle-inspection systems

These are adjacent, not our market, but judges may name them.

- **UVeye.** VERIFIED.
  - Drive-through scanners for exterior, underbody and tyres.
  - Funding of $191M (equity + debt; $41M equity) led by Woven Capital (Toyota), 29 Jan 2025: <https://uveye.com/funding-round-extension/>, <https://www.autoremarketing.com/ar/technology/uveye-says-new-191m-funding-round-will-help-it-meet-surging-global-demand>
- **DeGould.** VERIFIED.
  - Toyota Mississippi checks "all Corolla vehicles leaving the plant for damage, equal to 170,000 vehicles a year".
  - The setup uses a "dent arch-lighting system".
  - <https://www.automotivelogistics.media/digitalisation/toyota-installs-degould-vehicle-inspection-system-at-mississippi-plant/200876>
- **ProovStation.** VERIFIED. "200+ images per scan", 4K cameras with "dynamic LED lighting": <https://proovstation.com/services/body-inspection>
- **Tchek.** VERIFIED. Smartphone-guided photo inspection with real-time checks for "angle, brightness, blur and coverage": <https://www.tchek.ai/press-release/volkswagen-financial-services-rolls-out-digital-end-of-lease-pre-inspection-with-tchek-ai>

### Uzbekistan market

- **427,097** new cars, including light commercial vehicles, were sold in Uzbekistan in 2025 (CERR report). VERIFIED: <https://www.spot.uz/ru/2026/01/17/cars-sales-2025/>
- In the same article, BYD Uzbekistan Factory has a 5.6 % share and ADM Jizzakh 10.8 %. VERIFIED. The UzAuto Motors share appeared only in a snippet: UNVERIFIED.
- **BYD Uzbekistan Factory** (Jizzakh). VERIFIED.
  - A JV of Uzavtosanoat (60 %) and BYD Europe B.V. (40 %), with "$160 million" invested: <https://www.gazeta.uz/en/2024/03/19/byd-uzbekistan-factory/>
  - First phase 50,000 vehicles a year: <https://carnewschina.com/2024/06/28/byd-first-batch-mass-produced-vehicles-rolled-off-assembly-line-in-uzbekistan-factory/>

### Hosting

Hugging Face Spaces plans and hardware: <https://huggingface.co/pricing>, <https://huggingface.co/docs/hub/spaces-gpus>, <https://huggingface.co/docs/hub/spaces-sdks-docker>. VERIFIED on 2026-10-11 (re-check before the deck):
- CPU Basic: 2 vCPU, 16 GB, hourly price "Free!"; CPU Upgrade: 8 vCPU, 32 GB, "$0.03" per hour (spaces-gpus hardware table).
- "CPU Basic has no hourly cost, but creating a new Space that runs on compute (Gradio or Docker) requires a paid plan" (same page).
- PRO account: "$9 /month" (pricing page).
- CPU Basic Spaces go to sleep after 48 hours without use; a visit restarts them (same page, "Set a custom sleep time").
