# Team test set: real photos (`data/pairs/`)

This is the protocol for AVA's own real test set, covering **both modes**. The judges ask for quality "on your own test set". The synthetic benchmarks ([eval/RESULTS.md](../../eval/RESULTS.md), [eval/RESULTS_SURFACE.md](../../eval/RESULTS_SURFACE.md)) cannot show how real phone cameras, real lamps and real materials behave.

**Status:** no real photos have been evaluated yet. No real-data number is claimed anywhere until results on this set are published.

## Rules (both modes)

- **Own objects only.**
- **Nothing private in the frame:** no faces, people, licence plates, documents or screens. If a hand is needed to apply a defect, take it out of the frame before the shot.
- **No editing:** no crop, filter or HDR. Keep the phone's original JPEG.
- **Camera:** the default camera app with automatic exposure, rear main camera, no zoom. Exposure drift is part of what we test.
- **Record the setup:** for each set, write the phone model, the lamp and the material in `set.txt` next to the photos.

## Lighting labels

| Label | Meaning | Group |
|---|---|---|
| `base` | The lighting of the reference shot (Mode 2), or of the normal shots (Mode 1). | — |
| `dimmed` | The lamp dimmed, partly covered, or turned further away. | monotone |
| `lamp_moved` | The same lamp moved about 30–50 cm sideways (soft, uneven light). | light direction: monotone only on flat, matte surfaces. On a relief part (ribs, edges, grooves) it flips the shading order: an expected failure, record it separately |
| `daylight` | Window light on; say in `notes` whether the lamp was on. | monotone |
| `flash_side` | The phone flashlight from the side, held at a distance, diffuse. | light direction (as `lamp_moved`) |
| `specular` | A deliberate highlight on a glossy area. | non-monotone (expected failure) |
| `hand_shadow` | A hard shadow across the part. | non-monotone (expected failure) |

## Defects

Use removable defects so that the same object can be reused:
- a strip of **masking tape** (1–3 cm);
- **marker dots** (washable marker on a protected surface);
- **small stickers** (0.5–2 cm);
- a **removed** screw, clip or cap;
- an **added** washer, nut or piece of wire;
- a drop of **oil or water** (stain).

Optionally, a **real scratch, chip or dent** on scrap parts only.

---

## Mode 2: compare with reference (`data/pairs/compare/`)

**Station setup:**
- A phone on a tripod or clamp, 25–50 cm above a table. Look straight down, or tilt less than 15°. Do not move the phone during a set.
- A plain or lightly textured background, the same for the whole set. The part covers about 40–80 % of the frame.

**Objects** (at least 5): metal brackets, stamped plates, plastic housings and clips with ribs, painted samples, small real auto parts (filter, sensor, fasteners). Include **one** featureless or glossy object on purpose. It is a known weak case and must be reported.

**Shots per object** (one set = one object and one camera position):

| Group | Shots | Notes |
|---|---|---|
| `reference.jpg` | 1 | Good part, `base` lighting. This is what the web UI uses. |
| Extra references `reference_2..5.jpg` | optional | Lift the part and put it back roughly in place between shots. For library experiments with several references. |
| Good frames | ≥ 5 | Same part, untouched, re-placed each time. These measure false alarms. |
| Defect frames | ≥ 5 | One defect per frame. |
| Lighting variants | ≥ 5 good + ≥ 3 with a defect | Labelled with a lighting label. |
| Moved part | ≥ 3 good + ≥ 3 with a defect | The part moved by 1–3 cm or turned by a few degrees, camera fixed. Note it in `notes`. |

**Layout:**
```
data/pairs/compare/<set>/reference.jpg
data/pairs/compare/<set>/frame_001.jpg
data/pairs/compare/<set>/frame_001.json
...
data/pairs/compare/<set>/set.txt
```

**Labels.** Boxes are in pixels of `reference.jpg`, at its original resolution after EXIF rotation, i.e. as an image viewer shows it upright. AVA reports zones in reference coordinates, so draw each box where the defect would be **on the reference**. With a fixed camera and an unmoved part, that is simply where it appears in the frame.

---

## Mode 1: surface check (`data/pairs/surface/`)

**Setup:**
- A phone held or clamped 10–30 cm from the surface, roughly perpendicular.
- **The surface must fill the whole frame.** No part edges, holes, logos, background or fingers, because Mode 1 flags anything rare in the photo.

**Surfaces** (at least 5): painted panel or painted tin, matte plastic, sheet or brushed metal, fabric or mesh, a dark glossy surface. Include **one** brushed-metal surface: it is a known weak case (synthetic false-alarm rate 20.8 %).

**Shots per surface** (one set = one surface):

| Group | Shots | Notes |
|---|---|---|
| Good photos | ≥ 8 | Shift the phone a little between shots. These measure false alarms. |
| Defect photos | ≥ 8 | One defect per photo: tape, marker dot, sticker, stain, or a real scratch or chip. |
| Lighting variants | ≥ 5 good + ≥ 3 with a defect | Labelled with a lighting label. |

**Layout:**
```
data/pairs/surface/<set>/photo_001.jpg
data/pairs/surface/<set>/photo_001.json
...
data/pairs/surface/<set>/set.txt
```

**Labels.** Boxes are in pixels of the photo itself (original resolution, after EXIF rotation).

---

## Label format (`*.json`, both modes)

```json
{
  "defect": true,
  "boxes": [[412, 280, 96, 60]],
  "lighting": "lamp_moved",
  "notes": "masking tape, 2 cm, near the left hole"
}
```

| Field | Type | Meaning |
|---|---|---|
| `defect` | bool | `true` if the photo shows a defect, `false` for a good part or surface. |
| `boxes` | list of `[x, y, w, h]` | One box per defect, with a margin of a few pixels. Empty `[]` when `defect` is `false`. Coordinates are in the pixels of the image named above. |
| `lighting` | str | One of the lighting labels. |
| `notes` | str | Free text: defect kind and size, anything unusual (part moved, reflection, blur). |

**Set names** are short, lower case and without spaces, for example `bracket_steel_01`, `clip_black_02`, `panel_red_01`. Files are numbered `frame_001`, `photo_001`, …, and the order does not matter.

**Drawing boxes:** any viewer that shows pixel coordinates works (GIMP, Paint.NET, or OpenCV `selectROI`).

**Scaling.** AVA works at `work_max_side` = 1024 px and returns `work_size` with every result. An evaluation script must scale the boxes by `work_size / original size`.

## How results will be reported

- Run each photo through the same code as the site: `ava.pipeline.prepare_reference` + `inspect_compare` (methods `pbp,ssim,census,pixdiff`) for Mode 2, `ava.surface.scan_surface` for Mode 1, or the HTTP API.
- Use the **frozen** thresholds from `ava/config.py`. Do not re-tune on this set.
- Use the same rules as the synthetic benchmarks:
  - a defect counts as detected if the verdict is FAIL and a zone has IoU > 0.1 with a box or covers ≥ 30 % of it;
  - the false-alarm rate is the share of good photos answered FAIL;
  - Wilson 95 % intervals.
- Report per mode and per lighting group, **separately from the synthetic results**, with the number of sets, photos and objects, and with the failures listed.

## Checklist per set

- [ ] Every `*.jpg` has a `*.json` with all four fields.
- [ ] No faces, people, plates, documents or screens are visible.
- [ ] Mode 1: the surface fills the whole frame in every photo.
- [ ] Mode 2: the camera did not move during the set (except for the "moved part" frames, where only the part moves).
- [ ] `set.txt` records the object or surface, the material, the phone model and the lamp.

## Publishing and licence

- `.gitignore` excludes `data/pairs/**/*.jpg|jpeg|png`. **The photos are not committed by default**; only the JSON labels, `set.txt` and this protocol are.
- To make the set reproducible for the judges, publish the images as a zip or a Hugging Face dataset, and link it from the results.
- The photos are taken by Team AVA of its own objects and published under **CC BY 4.0**.
