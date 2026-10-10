"""Static checks of the web client (web/): files, no CDNs, i18n coverage, SPEC wording and API, JS syntax."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
JS_FILES = ["app.js", "i18n.js", "qr.js", "qrcode.js"]
WEB_FILES = ["index.html", "style.css", "icon.svg", *JS_FILES]


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
    for name in WEB_FILES:
        assert (WEB / name).is_file(), name


def test_qr_library_keeps_licence_header():
    head = _read("qrcode.js")[:800]
    assert "Kazuhiko Arase" in head and "MIT" in head


def test_index_loads_local_assets_only():
    html = _read("index.html")
    assert re.search(r"<title>AVA</title>", html)
    for name in ["style.css", "icon.svg"]:
        assert name in html, name
    # Scripts in dependency order: encoder -> wrapper -> strings -> app.
    order = [html.index(f'<script src="{n}"></script>') for n in ["qrcode.js", "qr.js", "i18n.js", "app.js"]]
    assert order == sorted(order)
    for page in ["index.html", "app.js", "style.css"]:
        assert not re.search(r"""(src|href)\s*=\s*["']https?://""", _read(page)), page
        assert "@import" not in _read(page)


def test_product_name_is_ava_everywhere():
    for name in WEB_FILES:
        assert "dalil" not in _read(name).lower(), name


def test_no_station_journal_or_viewer_leftovers():
    texts = {name: _read(name).lower() for name in ["index.html", "app.js", "i18n.js", "style.css"]}
    for name, text in texts.items():
        for word in ["station", "journal", "viewer", "станци", "журнал", "наблюдател", "hashchange"]:
            assert word not in text, (name, word)


def test_i18n_languages_have_same_keys():
    ru, en = _dict_keys("ru"), _dict_keys("en")
    assert ru and ru == en, (ru ^ en)


def test_every_key_used_is_translated():
    app = _read("app.js")
    used = set(re.findall(r"""\bt\('([a-z0-9_]+)'""", app))
    # Keys built from slot names and server codes (SPEC "reason", zone hints, camera failures).
    used |= {f"{p}_{s}" for p in ["slot", "cam_title"] for s in ["reference", "frame"]}
    used |= {f"slot_{s}_sub" for s in ["reference", "frame"]}
    used |= {f"rr_{r}" for r in ["alignment", "glare", "blurry", "dark", "low_coverage", "mismatch", "unknown"]}
    used |= {f"zh_{h}" for h in ["water_or_dirt", "something_removed", "something_added"]}
    used |= {f"cam_fail_{r}" for r in ["insecure", "denied", "unavailable"]}
    # Static markup in index.html.
    used |= set(re.findall(r'data-i18n(?:-aria)?="([a-z0-9_]+)"', _read("index.html")))
    missing = used - _dict_keys("ru")
    assert not missing, sorted(missing)


def test_wording_matches_spec():
    ru = _read("i18n.js")
    for text in ["Эталон", "Проверка", "Камера", "Из галереи", "Заменить", "Сравнить",
                 "ГОДНА — изменений нет", "БРАК — найдено зон: {n}", "ПЕРЕСНЯТЬ: {reason}",
                 "не удалось совместить: снимите с того же ракурса", "блик: смените угол на 15–30°",
                 "не та деталь или деталь отсутствует", "мало проверенной площади",
                 "похоже на загрязнение/каплю", "отсутствует элемент?", "лишний элемент/наплыв?",
                 "Простое вычитание из эталона", "БРАК — изменился почти весь кадр",
                 "Так простое вычитание реагирует на смену света",
                 "Новая проверка", "Новый эталон", "Фото не сохраняются на сервере."]:
        assert text in ru, text
    assert "cognex" not in ru.lower() and "cognex" not in _read("app.js").lower()


def test_api_matches_spec():
    app = _read("app.js")
    found = set(re.findall(r"""[`'](/api/[^`']*)[`']""", app))
    assert found == {"/api/compare?methods=pbp,pixdiff"}, found
    assert "form.append('reference'" in app and "form.append('frame'" in app
    # Client-side downscale: longest side 1600 px, JPEG 0.9 from a canvas (no EXIF).
    assert re.search(r"MAX_SIDE = 1600\b", app) and re.search(r"JPEG_QUALITY = 0\.9\b", app)
    # Baseline display rule for plain subtraction.
    assert "reason === 'mismatch'" in app


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("name", JS_FILES)
def test_js_syntax(name):
    res = subprocess.run(["node", "--check", str(WEB / name)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
