# Demo examples

The five images behind the "Try an example" buttons. The buttons only load these files into the photo slots; the verdicts are always computed live by the API.

| File | Mode | What it shows | Expected |
|---|---|---|---|
| `surface_defect.jpg` | Surface check | black seat leather with a hairline scratch | FAIL, zone on the scratch |
| `surface_ok.jpg` | Surface check | the same leather, no defect | PASS |
| `compare_reference.jpg` | Compare | reference: drilled brake disc on a garage floor | — |
| `compare_defect.jpg` | Compare | the same disc with a hairline crack from a drilled hole | AVA FAIL (plain subtraction misses it) |
| `compare_light.jpg` | Compare | the same disc, darker and warmer light, no defect | AVA PASS (plain subtraction: false alarm) |

**Source and honesty note:** real texture photos from [Poly Haven](https://polyhaven.com) (CC0 1.0 — public domain) + **synthetic** defects, geometry and lighting changes made by our code. No faces, no licence plates, no logos.

| Texture | Author | Licence |
|---|---|---|
| [Brown Leather](https://polyhaven.com/a/brown_leather) (colour-graded to black) | Rob Tuytel | CC0 1.0 |
| [Rusty Metal Sheet](https://polyhaven.com/a/rusty_metal_sheet) | Amal Kumar | CC0 1.0 |
| [Garage Floor](https://polyhaven.com/a/garage_floor) | Jenelle van Heerden | CC0 1.0 |

Regenerate: `eval/real_texture_examples/` (download the textures with the `fetch_*_assets.py` scripts, then run `make_surface_examples.py` / `make_compare_examples.py`). `tests/test_demo_examples.py` checks all four expected verdicts. You can replace the files with your own photos under the same names — then re-run the tests.
