"""HTTP API tests (docs/SPEC.md "HTTP API") with FastAPI's TestClient on a temp data dir."""

import base64
import io
import re

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server.app as app_module
from dalil import report, synth
from server.store import Store

H, W = 576, 1024


def jpeg(img: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def images():
    """Two golden shots, a good frame and a frame with tape, all of one synthetic scene."""
    good = synth.make_station_pair(np.random.default_rng(7), part_kind="stamped", lighting="gamma",
                                   motion="jitter", h=H, w=W)
    bad = synth.make_station_pair(np.random.default_rng(7), part_kind="stamped", defect="tape",
                                  lighting="drift_gain", h=H, w=W)
    clean = good["meta"]["reference_clean"]
    assert np.array_equal(clean, bad["meta"]["reference_clean"])
    shots = synth.make_reference_set(np.random.default_rng(1), clean, 2)
    return {"refs": [jpeg(s) for s in shots], "good": jpeg(good["frame"]), "bad": jpeg(bad["frame"])}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app_module._store = Store(root=tmp_path_factory.mktemp("dalil_data"), params=app_module.PARAMS)
    with TestClient(app_module.app) as c:
        yield c
    app_module._store = None


def upload(name: str, data: bytes) -> dict:
    return {name: (f"{name}.jpg", data, "image/jpeg")}


def test_health_and_static(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    body = r.json()
    assert body["ok"] is True and body["version"]
    assert body["params"]["threshold"] == app_module.PARAMS.threshold
    assert "threshold_pixdiff" in body["params"]
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()
    assert client.get("/qrcode.js").status_code == 200
    assert client.get("/app.js").status_code == 200


def test_station_flow(client, images):
    # create + read
    r = client.post("/api/stations", json={"name": "Линия 1 <b>", "sku": "BR-17"})
    assert r.status_code == 200, r.text
    st = r.json()
    sid = st["id"]
    assert re.fullmatch(r"[A-Z2-9]{6}", sid)
    assert st["name"] == "Линия 1 <b>" and st["sku"] == "BR-17"
    assert st["references"] == [] and st["outline_url"] is None and st["last_seq"] == 0
    assert st["chain_head"] == "0" * 64
    assert st["stats"] == {"total": 0, "pass": 0, "fail": 0, "retake": 0,
                           "confirmed_defects": 0, "false_alarms": 0}
    assert client.get(f"/api/stations/{sid}").json()["id"] == sid

    # no reference yet
    r = client.post(f"/api/stations/{sid}/inspect", files=upload("file", images["good"]))
    assert r.status_code == 409 and r.json()["error"] == "no_reference"

    # two golden shots
    for i, data in enumerate(images["refs"]):
        r = client.post(f"/api/stations/{sid}/reference", files=upload("file", data))
        assert r.status_code == 200, r.text
        st = r.json()
        assert len(st["references"]) == i + 1
        assert isinstance(st["last_capture"]["warnings"], list)
    assert [ref["idx"] for ref in st["references"]] == [0, 1]
    assert st["outline_url"] == f"/api/files/{sid}/outline.png"
    r = client.get(st["outline_url"] + "?v=2")  # the client adds a cache-buster
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert client.get(st["references"][0]["url"]).content[:2] == b"\xff\xd8"

    # good part with the baseline comparison
    r = client.post(f"/api/stations/{sid}/inspect?compare=1", files=upload("file", images["good"]))
    assert r.status_code == 200, r.text
    rec1 = r.json()
    assert rec1["seq"] == 1 and rec1["verdict"] == "pass", rec1
    assert rec1["zones"] == [] and rec1["review"] is None and rec1["method"] == "pbp"
    assert rec1["prev_hash"] == "0" * 64 and len(rec1["hash"]) == 64
    assert rec1["baseline"]["method"] == "pixdiff"
    assert rec1["baseline"]["verdict"] in ("pass", "fail", "retake")
    for url in (rec1["frame_url"], rec1["overlay_url"], rec1["baseline"]["overlay_url"]):
        resp = client.get(url)
        assert resp.status_code == 200 and resp.headers["content-type"] == "image/jpeg"
    assert {"align", "coverage", "changed_frac", "timing_ms", "at"} <= set(rec1)

    # defect
    r = client.post(f"/api/stations/{sid}/inspect", files=upload("file", images["bad"]))
    rec2 = r.json()
    assert rec2["seq"] == 2 and rec2["verdict"] == "fail" and rec2["zones"], rec2
    assert rec2["prev_hash"] == rec1["hash"] and rec2["baseline"] is None
    z = rec2["zones"][0]
    assert len(z["bbox"]) == 4 and len(z["bbox_norm"]) == 4

    # review
    r = client.post(f"/api/stations/{sid}/inspections/2/review", json={"label": "defect"})
    assert r.status_code == 200 and r.json()["review"] == "defect" and r.json()["seq"] == 2
    assert client.post(f"/api/stations/{sid}/inspections/2/review", json={"label": "maybe"}).status_code == 400
    r = client.post(f"/api/stations/{sid}/inspections/9/review", json={"label": "ok"})
    assert r.status_code == 404 and r.json()["error"] == "inspection_not_found"

    # polling
    r = client.get(f"/api/stations/{sid}/inspections?after=0&limit=50")
    data = r.json()
    assert r.headers["cache-control"] == "no-store"
    assert [i["seq"] for i in data["items"]] == [1, 2] and data["last_seq"] == 2
    data = client.get(f"/api/stations/{sid}/inspections?after=1&limit=50").json()
    assert [i["seq"] for i in data["items"]] == [2] and data["items"][0]["review"] == "defect"
    assert client.get(f"/api/stations/{sid}/inspections?after=2").json()["items"] == []
    assert client.get(f"/api/stations/{sid}/inspections?after=0&limit=1").json()["items"][0]["seq"] == 1

    st = client.get(f"/api/stations/{sid}").json()
    assert st["last_seq"] == 2
    assert st["stats"] == {"total": 2, "pass": 1, "fail": 1, "retake": 0,
                           "confirmed_defects": 1, "false_alarms": 0}

    # the on-disk journal is a valid chain (inspections + the review event)
    store = app_module.get_store()
    station, records = store.records(sid)
    assert [report.is_review(rec) for rec in records] == [False, False, True]
    assert report.verify_chain(records, store.blobs_for(sid)) == (True, None)
    assert station["chain_head"] == records[-1]["hash"]

    # CSV + report
    r = client.get(f"/api/stations/{sid}/journal.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    text = r.content.decode("utf-8")
    lines = text.lstrip("﻿").strip().splitlines()
    assert lines[0].startswith("seq,time_local,verdict") and len(lines) == 3
    assert ",fail," in lines[2] and ",defect," in lines[2]
    r = client.get(f"/api/stations/{sid}/report")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert station["chain_head"] in r.text and "цепочка проверена" in r.text
    assert "Линия 1 &lt;b&gt;" in r.text and "<b>" not in r.text.split("Линия 1", 1)[1][:10]
    assert "data:image/jpeg;base64," in r.text  # thumbnail of the fail row

    # clearing the golden set
    st = client.delete(f"/api/stations/{sid}/reference").json()
    assert st["references"] == [] and st["outline_url"] is None
    r = client.post(f"/api/stations/{sid}/inspect", files=upload("file", images["good"]))
    assert r.status_code == 409


def test_compare_endpoint(client, images):
    files = {**upload("reference", images["refs"][0]), **upload("frame", images["bad"])}
    r = client.post("/api/compare?methods=pbp,pixdiff", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["results"]) == {"pbp", "pixdiff"} == set(body["overlays"])
    assert body["results"]["pbp"]["verdict"] == "fail"
    for b64 in body["overlays"].values():
        assert base64.b64decode(b64)[:2] == b"\xff\xd8"
    r = client.post("/api/compare?methods=pbp,evil", files=files)
    assert r.status_code == 400 and r.json()["error"] == "bad_methods"
    r = client.post("/api/compare", files=upload("reference", images["refs"][0]))
    assert r.status_code == 400 and r.json()["error"] == "file_required"


def test_files_traversal_rejected(client):
    sid = client.post("/api/stations", json={"name": "T"}).json()["id"]
    for path in (f"/api/files/{sid}/station.json", f"/api/files/{sid}/journal.jsonl",
                 f"/api/files/{sid}/..%2Fstation.json", f"/api/files/{sid}/%2e%2e%2f%2e%2e%2fCLAUDE.md",
                 f"/api/files/{sid}/....%5C..%5Cstation.json", "/api/files/../../server/app.py",
                 f"/api/files/{sid}/f_0.jpg", f"/api/files/{sid}/f_1.jpg.png",
                 f"/api/files/{sid.lower()}/outline.png", "/api/files/..%2F..%2Fdalil/config.py"):
        r = client.get(path)
        assert r.status_code in (400, 404), path
        assert b"name" not in r.content.lower() or b"error" in r.content, path
        assert b"import" not in r.content, path


def test_bad_station_ids(client):
    for bad in ("abcdef", "ABCDE1", "ABCDE", "ABCDEFG", "ABC%2FEF", "ABCDE0"):
        r = client.get(f"/api/stations/{bad}")
        assert r.status_code in (400, 404), bad
        if r.status_code == 400:
            assert r.json()["error"] == "bad_station_id"
    r = client.get("/api/stations/ABCDE1")
    assert r.status_code == 400 and r.json()["error"] == "bad_station_id"
    r = client.get("/api/stations/ZZZZZZ")
    assert r.status_code == 404 and r.json()["error"] == "station_not_found"
    assert client.post("/api/stations", json={"name": "  "}).json()["error"] == "name_required"
    assert client.post("/api/stations", content=b"not json",
                       headers={"content-type": "application/json"}).status_code == 400


def test_oversize_and_bad_uploads(client, images):
    sid = client.post("/api/stations", json={"name": "Big"}).json()["id"]
    big = b"\xff\xd8" + b"\0" * (16 * 1024 * 1024)
    r = client.post(f"/api/stations/{sid}/reference", files=upload("file", big))
    assert r.status_code == 413 and r.json()["error"] == "too_large"
    # one file over 15 MB inside a request that is under the whole-body limit
    files = {**upload("reference", big), **upload("frame", images["good"])}
    r = client.post("/api/compare", files=files)
    assert r.status_code == 413
    r = client.post(f"/api/stations/{sid}/reference", files=upload("file", b"this is not an image"))
    assert r.status_code == 400 and r.json()["error"] == "bad_image"
    r = client.post(f"/api/stations/{sid}/reference", data={"file": "text"})
    assert r.status_code == 400
    assert client.get(f"/api/stations/{sid}").json()["references"] == []


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (400, 200), (40, 90, 160))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes())
    bgr = app_module.decode_image(buf.getvalue())
    assert bgr.shape == (400, 200, 3)
    assert abs(int(bgr[100, 100, 0]) - 160) < 6  # BGR order
