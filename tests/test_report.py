"""Tests for dalil.report: hash chain, CSV export and the shift report HTML."""

import copy
import csv
import io
import json

import pytest

from dalil.report import (GENESIS_HASH, CSV_COLUMNS, FOOTNOTE, canonical_json, check_links,
                          journal_csv, make_review_event, record_hash, render_shift_report,
                          seal_record, shift_stats, verify_chain)

BLOBS = {
    1: [b"frame-1", b"overlay-1"],
    2: [b"frame-2", b"overlay-2"],
    3: [b"frame-3", b"overlay-3"],
    4: [],
}


def inspection(seq, verdict="pass", zones=(), reason=None, at=None):
    return {
        "seq": seq, "at": at or f"2026-10-10T0{seq}:00:00Z", "verdict": verdict, "reason": reason,
        "zones": [dict(z) for z in zones], "coverage": 0.93, "changed_frac": 0.01,
        "timing_ms": {"align": 40.5, "features": 120.25, "total": 180.0},
        "align": {"method": "identity", "n_inliers": 0, "inlier_ratio": 1.0},
        "method": "pbp", "frame_url": f"/api/files/ABC234/f{seq}.jpg",
        "overlay_url": f"/api/files/ABC234/o{seq}.jpg", "baseline": None, "review": None,
    }


def build_journal():
    """3 inspections (pass, fail, retake) + a review of the fail, all chained."""
    raw = [
        inspection(1),
        inspection(2, "fail", zones=[{"id": 1, "bbox": [10, 20, 30, 40], "bbox_norm": [0.1, 0.2, 0.3, 0.4]}]),
        inspection(3, "retake", reason="glare"),
        make_review_event(4, "2026-10-10T05:00:00Z", 2, "defect"),
    ]
    out, prev = [], GENESIS_HASH
    for rec in raw:
        sealed = seal_record(prev, rec, BLOBS[rec["seq"]])
        out.append(sealed)
        prev = sealed["hash"]
    return out


def blobs_for(rec):
    return BLOBS[rec["seq"]]


def test_canonical_json_is_key_order_invariant_and_compact():
    a = {"b": 1, "a": {"y": [1, 2], "x": "Цех"}}
    b = {"a": {"x": "Цех", "y": [1, 2]}, "b": 1}
    assert canonical_json(a) == canonical_json(b) == '{"a":{"x":"Цех","y":[1,2]},"b":1}'


def test_record_hash_deterministic_and_key_order_invariant():
    rec = inspection(1)
    h1 = record_hash(GENESIS_HASH, rec, [b"x"])
    h2 = record_hash(GENESIS_HASH, dict(reversed(list(rec.items()))), [b"x"])
    assert h1 == h2 and len(h1) == 64
    assert h1 == record_hash(GENESIS_HASH, json.loads(json.dumps(rec)), [b"x"])  # survives a JSON round trip


def test_record_hash_ignores_bookkeeping_fields_but_not_content():
    rec = inspection(1)
    base = record_hash(GENESIS_HASH, rec, [])
    assert record_hash(GENESIS_HASH, {**rec, "hash": "x", "prev_hash": "y", "review": "defect"}, []) == base
    assert record_hash(GENESIS_HASH, {**rec, "verdict": "fail"}, []) != base
    assert record_hash("1" * 64, rec, []) != base
    assert record_hash(GENESIS_HASH, rec, [b"a"]) != base
    assert record_hash(GENESIS_HASH, rec, [b"a", b"b"]) != record_hash(GENESIS_HASH, rec, [b"b", b"a"])


def test_verify_chain_intact():
    recs = build_journal()
    assert verify_chain(recs, blobs_for) == (True, None)
    assert verify_chain([], blobs_for) == (True, None)
    assert check_links(recs, recs[-1]["hash"]) == (True, None)


@pytest.mark.parametrize("idx", [0, 1, 2, 3])
def test_any_record_edit_breaks_chain_at_that_seq(idx):
    recs = build_journal()
    tampered = copy.deepcopy(recs)
    key = "label" if tampered[idx].get("type") == "review" else "verdict"
    tampered[idx][key] = "ok" if key == "label" else "tampered"
    assert verify_chain(tampered, blobs_for) == (False, recs[idx]["seq"])


def test_nested_edit_breaks_chain():
    recs = build_journal()
    recs[1]["zones"][0]["bbox"][0] = 11
    assert verify_chain(recs, blobs_for) == (False, 2)


def test_review_field_edit_is_not_a_chain_break_but_events_are():
    recs = build_journal()
    recs[0]["review"] = "false_alarm"  # convenience field, not hashed
    assert verify_chain(recs, blobs_for) == (True, None)


def test_blob_change_breaks_chain_at_that_seq():
    recs = build_journal()
    changed = dict(BLOBS)
    changed[2] = [b"frame-2", b"overlay-2-edited"]
    assert verify_chain(recs, lambda r: changed[r["seq"]]) == (False, 2)

    def missing(r):
        if r["seq"] == 3:
            raise FileNotFoundError("gone")
        return BLOBS[r["seq"]]

    assert verify_chain(recs, missing) == (False, 3)


def test_reorder_and_deletion_break_chain():
    recs = build_journal()
    swapped = [recs[0], recs[2], recs[1], recs[3]]
    assert verify_chain(swapped, blobs_for) == (False, 3)
    assert verify_chain([recs[0], recs[2], recs[3]], blobs_for) == (False, 3)
    assert verify_chain(recs[1:], blobs_for) == (False, 2)
    # truncating the tail keeps a valid prefix, but the stored head no longer matches
    assert verify_chain(recs[:3], blobs_for) == (True, None)
    assert check_links(recs[:3], recs[-1]["hash"]) == (False, 3)


def test_review_of_unknown_inspection_is_rejected():
    recs = build_journal()
    bogus = seal_record(recs[-1]["hash"], make_review_event(5, "2026-10-10T06:00:00Z", 99, "defect"), [])
    assert verify_chain(recs + [bogus], lambda r: BLOBS.get(r["seq"], [])) == (False, 5)
    with pytest.raises(ValueError):
        make_review_event(5, "2026-10-10T06:00:00Z", 1, "maybe")


def test_shift_stats_apply_review_events():
    stats = shift_stats(build_journal())
    assert stats == {"total": 3, "pass": 1, "fail": 1, "retake": 1, "confirmed_defects": 1, "false_alarms": 0}


def test_journal_csv_header_rows_and_bom():
    recs = build_journal()
    text = journal_csv({"name": "Линия 1"}, recs)
    assert text.startswith("﻿")
    rows = list(csv.reader(io.StringIO(text[1:])))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert len(rows) == 4  # header + 3 inspections; review events are not rows
    by = {r[0]: dict(zip(CSV_COLUMNS, r)) for r in rows[1:]}
    assert by["1"]["time_local"] == "2026-10-10 06:00:00"  # 01:00 UTC -> UTC+5
    assert by["2"]["verdict"] == "fail" and by["2"]["n_zones"] == "1"
    assert by["2"]["top_zone_bbox"] == "10 20 30 40"
    assert by["2"]["review"] == "defect"
    assert by["3"]["reason"] == "glare" and by["3"]["review"] == ""
    assert by["1"]["timing_ms_total"] == "180.0" and by["1"]["method"] == "pbp"
    assert by["3"]["hash"] == recs[2]["hash"]
    text.encode("utf-8")  # must be encodable as UTF-8


def test_report_escapes_user_text_and_shows_chain_head():
    recs = build_journal()
    station = {"id": "ABC234", "name": "<script>alert(1)</script>", "sku": "A&B \"x\"",
               "created_at": "2026-10-10T00:00:00Z", "chain_head": recs[-1]["hash"]}
    images = {"2": "data:image/jpeg;base64,AAAA", "1": "data:image/jpeg;base64,BBBB"}
    page = render_shift_report(station, recs, images)
    assert "<script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "A&amp;B &quot;x&quot;" in page
    assert recs[-1]["hash"] in page
    assert "цепочка проверена" in page
    assert FOOTNOTE in page and "Chikake &amp; Goldengorin" in page
    assert "@page" in page and "A4" in page
    assert "data:image/jpeg;base64,AAAA" in page      # fail row gets a thumbnail
    assert "data:image/jpeg;base64,BBBB" not in page  # pass rows do not
    assert "06:00:00 (Ташкент)" in page and "01:00:00 UTC" in page
    assert "<link" not in page and "<script" not in page


def test_report_full_verification_and_broken_chain():
    recs = build_journal()
    page = render_shift_report({"name": "L1"}, recs, {}, blobs_for=blobs_for)
    assert "цепочка проверена" in page
    recs[1]["verdict"] = "pass"
    page = render_shift_report({"name": "L1"}, recs, {}, blobs_for=blobs_for)
    assert "цепочка нарушена" in page and "№ 2" in page


def test_report_rejects_unsafe_image_urls_and_handles_empty_journal():
    recs = build_journal()
    page = render_shift_report({"name": "L1"}, recs, {"2": "javascript:alert(1)"})
    assert "javascript:" not in page
    empty = render_shift_report({"name": "L1"}, [], {})
    assert "Проверок нет" in empty and GENESIS_HASH in empty


def test_local_tz_falls_back_to_fixed_utc_plus_5(monkeypatch):
    import sys
    from datetime import datetime, timedelta, timezone

    from dalil import report

    monkeypatch.setitem(sys.modules, "zoneinfo", None)  # import fails like missing tz data
    tz = report._local_tz()
    assert tz.utcoffset(datetime(2026, 7, 1, tzinfo=timezone.utc)) == timedelta(hours=5)
