"""Static checks of the web client (web/): files, English only, no CDNs, both modes' API, examples, JS syntax."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
JS_FILES = ["app.js", "i18n.js", "qr.js", "qrcode.js"]
WEB_FILES = ["index.html", "style.css", "icon.svg", *JS_FILES]
EXAMPLE_FILES = ["surface_defect.jpg", "surface_ok.jpg", "compare_reference.jpg",
                 "compare_defect.jpg", "compare_light.jpg"]


def _read(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


def _text_files() -> list[Path]:
    return sorted(p for p in WEB.glob("*") if p.suffix in (".html", ".js", ".css"))


def _string_keys() -> set[str]:
    """Keys of STRINGS in i18n.js (entries are indented by exactly 4 spaces)."""
    src = _read("i18n.js")
    block = src[src.index("const STRINGS = {"):src.index("\n  };")]
    return set(re.findall(r"^ {4}(\w+):", block, flags=re.M))


def test_files_exist():
    for name in WEB_FILES:
        assert (WEB / name).is_file(), name


def test_qr_library_keeps_licence_header():
    head = _read("qrcode.js")[:800]
    assert "Kazuhiko Arase" in head and "MIT" in head


def test_index_loads_local_assets_only():
    html = _read("index.html")
    assert re.search(r"<title>AVA</title>", html)
    assert '<html lang="en">' in html
    for name in ["style.css", "icon.svg"]:
        assert name in html, name
    # Scripts in dependency order: encoder -> wrapper -> strings -> app.
    order = [html.index(f'<script src="{n}"></script>') for n in ["qrcode.js", "qr.js", "i18n.js", "app.js"]]
    assert order == sorted(order)
    for page in ["index.html", "app.js", "style.css"]:
        assert not re.search(r"""(src|href)\s*=\s*["']https?://""", _read(page)), page
        assert "@import" not in _read(page)


def test_english_only_no_cyrillic():
    files = _text_files()
    assert {p.name for p in files} >= {"index.html", "app.js", "i18n.js", "qr.js", "style.css"}
    for path in files:
        found = re.findall(r"[Ѐ-ӿ]+", path.read_text(encoding="utf-8"))
        assert not found, (path.name, found[:5])


def test_no_language_toggle():
    html, app, strings = _read("index.html"), _read("app.js"), _read("i18n.js")
    assert "data-lang" not in html and "lang-toggle" not in _read("style.css")
    for text in (app, strings):
        assert "setLang" not in text and "ava.lang" not in text


def test_product_name_is_ava_everywhere():
    # UI files only (web/examples/README.md is documentation, not UI).
    legacy = "da" + "lil"  # old working name, spelled split so this guard does not match itself
    for path in WEB.rglob("*"):
        if path.is_file() and path.suffix in (".html", ".js", ".css", ".svg"):
            assert legacy not in path.read_text(encoding="utf-8", errors="ignore").lower(), path.name


def test_no_station_journal_or_viewer_leftovers():
    for name in ["index.html", "app.js", "i18n.js", "style.css"]:
        text = _read(name).lower()
        for word in ["station", "journal", "viewer", "hashchange", "cognex"]:
            assert word not in text, (name, word)


def test_every_key_used_is_defined():
    app = _read("app.js")
    used = set(re.findall(r"""\bt\('([a-z0-9_]+)'""", app))
    # Keys built from modes, slot names and server codes (SPEC "reason", zone hints, camera failures).
    modes, slots = ["surface", "compare"], ["image", "reference", "frame"]
    used |= {f"{p}_{m}" for p in ["mode_hint", "run", "busy", "need", "result_title"] for m in modes}
    used |= {f"mode_{m}_short" for m in modes}
    used |= {f"{p}_{s}" for p in ["slot", "cam_title"] for s in slots} | {f"slot_{s}_sub" for s in slots}
    used |= {f"rr_{r}" for r in ["alignment", "glare", "blurry", "dark", "low_coverage", "mismatch", "unknown"]}
    used |= {f"rh_low_coverage_{m}" for m in modes} | {"rh_bright_spot", "v_pass_surface", "base_mismatch_frac"}
    used |= {f"zh_{h}" for h in ["water_or_dirt", "something_removed", "something_added"]}
    used |= {f"cam_fail_{r}" for r in ["denied", "unavailable"]}
    missing = used - _string_keys()
    assert not missing, sorted(missing)


def test_wording():
    html, strings = _read("index.html"), _read("i18n.js")
    assert '<h1 class="wordmark">AVA</h1>' in html
    assert ("Visual defect check without defect training data and without a light box. "
            "Photos are not stored.") in html
    for text in ["Surface check", "uniform texture · 1 photo", "Compare with reference", "complex parts · 2 photos"]:
        assert text in html, text
    for text in ["Surface photo", "Fill the frame with the surface only", "Check surface", "Reference", "Check",
                 "Compare", "Camera", "Gallery", "Try an example:", "Defect", "No defect", "Brightness change",
                 "Example image (synthetic)",
                 "PASS — no changes found", "PASS — no defects found", "FAIL — {n} defect zones",
                 "RETAKE — {reason}",
                 "could not align — shoot from the same angle", "glare — change the angle by 15–30°",
                 "photo is blurry", "photo is too dark", "different part or part missing",
                 "too little of the image could be checked",
                 "looks like dirt or a drop", "surface looks smoother here", "surface looks rougher here",
                 "Plain reference subtraction", "FAIL — a large part of the frame changed",
                 "New check", "New reference"]:
        assert text in strings, text
    # The texture-activity hints cannot tell added from removed material (smooth tape on steel
    # reads as "smoother"); the mismatch rule fires above mismatch_frac (15 %), not "almost all";
    # the user chooses the mode, AVA never picks it; tap-to-focus does not exist in the camera.
    for text in ["missing feature?", "extra material or burr?", "almost the whole frame",
                 "automatically", "tap the screen to focus"]:
        assert text not in strings + html, text


def test_api_matches_spec():
    app = _read("app.js")
    found = set(re.findall(r"""[`'](/api/[^`']*)[`']""", app))
    assert found == {"/api/scan", "/api/compare?methods=pbp,pixdiff"}, found
    assert "form.append('image'" in app
    assert "form.append('reference'" in app and "form.append('frame'" in app
    # Client-side downscale: longest side 1600 px, JPEG 0.9 from a canvas (no EXIF).
    assert re.search(r"MAX_SIDE = 1600\b", app) and re.search(r"JPEG_QUALITY = 0\.9\b", app)
    # Baseline display rule for plain subtraction.
    assert "reason === 'mismatch'" in app
    # The last mode is remembered (best effort).
    assert "ava.mode" in app and "localStorage" in app


def test_examples_are_loaded_not_hard_coded():
    app = _read("app.js")
    assert "'/examples/'" in app
    for name in EXAMPLE_FILES:
        assert f"'{name}'" in app, name
    # Examples feed the same path as a gallery pick and then the real request.
    body = app[app.index("async function runExample"):app.index("// Mode switch")]
    assert "setPhoto(" in body and "await run()" in body
    for verdict in ["'pass'", "'fail'", "verdict:"]:
        assert verdict not in body, verdict


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("name", JS_FILES)
def test_js_syntax(name):
    res = subprocess.run(["node", "--check", str(WEB / name)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
