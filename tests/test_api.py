"""HTTP API tests (docs/SPEC.md "HTTP API") with FastAPI's TestClient."""

import base64
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server.app as app_module
from dalil import pipeline, synth

H, W = 360, 640  # small synthetic frames keep the tests fast
METHODS = ("pbp", "pixdiff")


def jpeg(img: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def pairs():
    """A good pair (lighting drift + camera jitter) and a pair with tape on the part."""
    good = synth.make_station_pair(np.random.default_rng(2), part_kind="stamped", lighting="drift_gain",
                                   motion="jitter", h=H, w=W)
    bad = synth.make_station_pair(np.random.default_rng(2), part_kind="stamped", defect="tape",
                                  lighting="gamma", h=H, w=W)
    return {
        "good": {"reference": jpeg(good["reference"]), "frame": jpeg(good["frame"])},
        "bad": {"reference": jpeg(bad["reference"]), "frame": jpeg(bad["frame"]),
                "mask": bad["mask_ref_frame"]},
    }


@pytest.fixture(scope="module")
def client():
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture(autouse=True)
def empty_cache():
    app_module.ref_cache.clear()
    yield
    app_module.ref_cache.clear()


def files_of(reference: bytes | None = None, frame: bytes | None = None) -> dict:
    out = {}
    if reference is not None:
        out["reference"] = ("reference.jpg", reference, "image/jpeg")
    if frame is not None:
        out["frame"] = ("frame.jpg", frame, "image/jpeg")
    return out


def compare(client, pair: dict, methods: str = "pbp,pixdiff"):
    return client.post(f"/api/compare?methods={methods}", files=files_of(pair["reference"], pair["frame"]))


def assert_error(r, status: int, code: str | None = None):
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error", "message"} and body["message"]
    if code is not None:
        assert body["error"] == code, body


# --------------------------------------------------------------------------- happy path

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    body = r.json()
    assert body["ok"] is True and body["version"]
    assert body["params"]["threshold"] == app_module.PARAMS.threshold
    assert "threshold_pixdiff" in body["params"]


def test_static_index(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "<html" in r.text.lower()
    assert client.get("/app.js").status_code == 200


def test_compare_good_pair_passes(client, pairs):
    r = compare(client, pairs["good"])
    assert r.status_code == 200, r.text
    assert r.headers["cache-control"] == "no-store"
    body = r.json()
    assert set(body["results"]) == set(METHODS) == set(body["overlays"])
    pbp = body["results"]["pbp"]
    assert pbp["verdict"] == "pass" and pbp["zones"] == [], pbp
    assert pbp["method"] == "pbp" and pbp["work_size"] == [W, H]
    assert {"reason", "coverage", "changed_frac", "timing_ms", "align", "threshold",
            "signature_sha256", "hints"} <= set(pbp)
    for b64 in body["overlays"].values():
        assert base64.b64decode(b64)[:2] == b"\xff\xd8"


def test_compare_defect_fails_on_the_defect(client, pairs):
    bad = pairs["bad"]
    r = compare(client, bad)
    assert r.status_code == 200, r.text
    body = r.json()
    pbp = body["results"]["pbp"]
    assert pbp["verdict"] == "fail" and pbp["zones"], pbp
    # Some zone overlaps the true defect (dilated by one 8 px window: zones live on the window grid).
    near = cv2.dilate(bad["mask"].astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    hits = []
    for z in pbp["zones"]:
        x0, y0, x1, y1 = (round(v * s) for v, s in zip(
            (z["bbox_norm"][0], z["bbox_norm"][1],
             z["bbox_norm"][0] + z["bbox_norm"][2], z["bbox_norm"][1] + z["bbox_norm"][3]), (W, H, W, H)))
        hits.append(bool(near[y0:y1, x0:x1].any()))
    assert any(hits), pbp["zones"]
    assert set(body["overlays"]) == set(METHODS)
    assert all(base64.b64decode(b64)[:2] == b"\xff\xd8" for b64 in body["overlays"].values())


def test_sha_fields_and_reference_cache(client, pairs, monkeypatch):
    calls = []
    real = pipeline.prepare_reference

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "prepare_reference", counting)
    bad = pairs["bad"]
    first = compare(client, bad).json()
    second = compare(client, bad).json()
    assert len(calls) == 1 and len(app_module.ref_cache) == 1
    assert first["reference_sha256"] == second["reference_sha256"]
    assert len(first["reference_sha256"]) == 64 and len(first["frame_sha256"]) == 64
    assert first["reference_sha256"] != first["frame_sha256"]
    assert first["results"]["pbp"]["verdict"] == second["results"]["pbp"]["verdict"] == "fail"

    # A different method set is a different cache entry.
    assert compare(client, bad, "pbp").status_code == 200
    assert len(calls) == 2 and len(app_module.ref_cache) == 2


def test_reference_cache_is_bounded_lru():
    cache = app_module.ReferenceCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # "a" becomes most recent
    cache.put("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1 and cache.get("c") == 3
    assert len(cache) == 2
    assert app_module.REF_CACHE_SIZE <= 4


# --------------------------------------------------------------------------- errors

def test_missing_field(client, pairs):
    r = client.post("/api/compare", files=files_of(reference=pairs["good"]["reference"]))
    assert_error(r, 400, "file_required")
    r = client.post("/api/compare", data={"reference": "text", "frame": "text"})
    assert_error(r, 400)
    r = client.post("/api/compare", json={"reference": "x"})
    assert_error(r, 400, "multipart_required")


def test_non_image_bytes(client, pairs):
    r = client.post("/api/compare", files=files_of(b"this is not an image", pairs["good"]["frame"]))
    assert_error(r, 400, "bad_image")
    r = client.post("/api/compare", files=files_of(pairs["good"]["reference"], b"\x00" * 100))
    assert_error(r, 400, "bad_image")
    r = client.post("/api/compare", files=files_of(pairs["good"]["reference"], b""))
    assert_error(r, 400)
    assert len(app_module.ref_cache) == 0  # nothing was prepared


def test_oversize_upload(client, pairs):
    big = b"\xff\xd8" + b"\0" * (16 * 1024 * 1024)
    # One file over 15 MB inside a request under the whole-body limit: caught on the actual bytes.
    r = client.post("/api/compare", files=files_of(big, pairs["good"]["frame"]))
    assert_error(r, 413, "too_large")
    # A declared body over the limit is rejected from Content-Length before it is read.
    r = client.post("/api/compare", content=b"x",
                    headers={"content-type": "multipart/form-data; boundary=x",
                             "content-length": str(app_module.MAX_BODY + 1)})
    assert_error(r, 413, "too_large")


@pytest.mark.parametrize("methods", ["pbp,evil", "", ",", "PBP"])
def test_bad_methods(client, pairs, methods):
    assert_error(compare(client, pairs["good"], methods), 400, "bad_methods")


def test_unknown_api_route_is_json_404(client):
    assert_error(client.get("/api/stations/ABCDEF"), 404, "not_found")


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (400, 200), (40, 90, 160))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes())
    bgr = app_module.decode_image(buf.getvalue())
    assert bgr.shape == (400, 200, 3)
    assert abs(int(bgr[100, 100, 0]) - 160) < 6  # BGR order
