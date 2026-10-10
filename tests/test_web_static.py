"""Static checks of the web client (web/): files, no CDNs, i18n coverage, API paths vs SPEC, JS syntax."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
JS_FILES = ["app.js", "i18n.js", "qr.js", "qrcode.js"]


def _read(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


def _dict_keys(lang: str) -> set[str]:
    """Keys of DICT.<lang> in i18n.js (entries are indented by exactly 6 spaces)."""
    src = _read("i18n.js")
    ru_start = src.index("    ru: {")
    en_start = src.index("    en: {")
    block = src[ru_start:en_start] if lang == "ru" else src[en_start:src.index("\n  };", en_start)]
    return set(re.findall(r"^ {6}(\w+):", block, flags=re.M))


def test_files_exist():
    for name in ["index.html", "style.css", "icon.svg", *JS_FILES]:
        assert (WEB / name).is_file(), name


def test_qr_library_keeps_licence_header():
    head = _read("qrcode.js")[:800]
    assert "Kazuhiko Arase" in head and "MIT" in head


def test_index_loads_local_assets_only():
    html = _read("index.html")
    for name in ["style.css", "icon.svg", "qrcode.js", "qr.js", "i18n.js", "app.js"]:
        assert name in html, name
    # Scripts in dependency order: encoder -> wrapper -> strings -> app.
    order = [html.index(f'src="{n}"') for n in ["qrcode.js", "qr.js", "i18n.js", "app.js"]]
    assert order == sorted(order)
    for page in ["index.html", "app.js", "style.css"]:
        assert not re.search(r"""(src|href)\s*=\s*["']https?://""", _read(page)), page
        assert "@import" not in _read(page)


def test_i18n_languages_have_same_keys():
    ru, en = _dict_keys("ru"), _dict_keys("en")
    assert ru and ru == en, (ru ^ en)


def test_every_literal_key_used_by_app_is_translated():
    app = _read("app.js")
    used = set(re.findall(r"""\bt\('([a-z0-9_]+)'""", app))
    used |= {"loading", "noscript", "zone_forms"}
    # Keys built dynamically from server codes (SPEC "reason", zone hints, review labels, capture warnings).
    used |= {f"rr_{r}" for r in ["alignment", "glare", "blurry", "dark", "low_coverage", "mismatch", "unknown"]}
    used |= {f"zh_{h}" for h in ["water_or_dirt", "something_removed", "something_added"]}
    used |= {f"rv_{l}" for l in ["defect", "false_alarm", "ok"]}
    used |= {f"cam_fail_{r}" for r in ["insecure", "denied", "unavailable"]}
    used |= {f"warn_{w}" for w in ["blurry", "glare", "dark"]}
    used |= {f"filter_{f}" for f in ["all", "fail", "pass", "retake"]}
    used |= set(re.findall(r"""'((?:review|st)_[a-z_]+)'""", app))
    missing = used - _dict_keys("ru")
    assert not missing, sorted(missing)


def test_verdict_wording_matches_spec():
    ru = _read("i18n.js")
    assert "?compare=1" in _read("app.js")
    for text in ["ГОДНА", "БРАК", "ПЕРЕСНЯТЬ", "Смените угол на 15–30°",
                 "не та деталь или деталь отсутствует", "похоже на загрязнение/каплю",
                 "отсутствует элемент?", "лишний элемент/наплыв?", "Простое вычитание из эталона"]:
        assert text in ru, text
    assert "cognex" not in ru.lower() and "cognex" not in _read("app.js").lower()


def test_api_paths_match_spec():
    app = _read("app.js")
    found = set(re.findall(r"""[`'](/api/[^`']*)[`']""", app))
    allowed = [
        r"/api/health",
        r"/api/stations",
        r"/api/stations/\$\{enc\(sid\)\}",
        r"/api/stations/\$\{enc\(sid\)\}/reference",
        r"/api/stations/\$\{enc\(sid\)\}/inspect",
        r"/api/stations/\$\{enc\(sid\)\}/inspections\?after=\$\{enc\(after\)\}&limit=\$\{enc\(limit\)\}",
        r"/api/stations/\$\{enc\(sid\)\}/inspections/\$\{enc\(seq\)\}/review",
        r"/api/stations/\$\{enc\(sid\)\}/journal\.csv",
        r"/api/stations/\$\{enc\(sid\)\}/report",
        r"/api/compare\?methods=pbp,pixdiff",
    ]
    assert found, "no API paths found"
    for path in found:
        assert any(re.fullmatch(p, path) for p in allowed), path
    # Multipart field names from SPEC.
    assert "form.append('file'" in app
    assert "form.append('reference'" in app and "form.append('frame'" in app


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("name", JS_FILES)
def test_js_syntax(name):
    res = subprocess.run(["node", "--check", str(WEB / name)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
