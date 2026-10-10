"""Shift journal: hash-chained records, CSV export and a printable shift report.

Journal model
-------------
The journal is an append-only list of records. Two kinds share one hash chain:

* inspection records (no "type" key, or any type other than "review") - the
  inspection record JSON from docs/SPEC.md (seq, at, verdict, reason, zones, ...);
* review events ``{"seq", "at", "type": "review", "target_seq", "label"}`` - an
  operator's verdict on an earlier inspection ("defect" | "false_alarm" | "ok").

A review never edits the hashed content of the inspection it targets: the
inspection's convenience "review" field is excluded from hashing, and the label
is recorded as a separate chained event. Rewriting history therefore breaks the
chain at the first edited record.

What is hashed (see `record_hash`)::

    sha256( prev_hash
          + canonical_json(record minus "hash", "prev_hash", "review")
          + sha256_hex(blob_1) + sha256_hex(blob_2) + ... )

as UTF-8 text, genesis prev_hash = "0" * 64.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Callable, Iterable

GENESIS_HASH = "0" * 64
UNHASHED_KEYS = ("hash", "prev_hash", "review")
REVIEW_LABELS = ("defect", "false_alarm", "ok")

CSV_COLUMNS = ("seq", "time_local", "verdict", "reason", "n_zones", "top_zone_bbox",
               "review", "method", "timing_ms_total", "hash")

METHOD_NOTE = ("Сравнение с эталоном методом PBP (псевдобулевы полиномы, Chikake & Goldengorin). "
               "Решение подтверждает оператор.")
FOOTNOTE = "Результат — подсказка; решение о браке подтверждает оператор"


def _local_tz() -> tzinfo:
    """Asia/Tashkent; fixed UTC+5 if tz data is missing (Windows without tzdata). No DST there."""
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Asia/Tashkent")
    except Exception:  # ZoneInfoNotFoundError, missing module/data
        return timezone(timedelta(hours=5), "UTC+5")


LOCAL_TZ = _local_tz()


# --------------------------------------------------------------------------- hashing

def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8 text kept as is."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def record_hash(prev_hash: str, record: dict, blobs: Iterable[bytes] | None = None) -> str:
    """Chain hash of one journal record.

    Hashes, as UTF-8 text: ``prev_hash`` + ``canonical_json(record)`` with the keys
    "hash", "prev_hash" and "review" removed + the hex sha256 of each blob in order
    (e.g. the stored frame and overlay JPEGs). Returns the hex sha256 digest.
    """
    body = {k: v for k, v in record.items() if k not in UNHASHED_KEYS}
    h = hashlib.sha256()
    h.update(prev_hash.encode("utf-8"))
    h.update(canonical_json(body).encode("utf-8"))
    for blob in blobs or ():
        h.update(hashlib.sha256(blob).hexdigest().encode("ascii"))
    return h.hexdigest()


def seal_record(prev_hash: str, record: dict, blobs: Iterable[bytes] | None = None) -> dict:
    """Return a copy of `record` with "prev_hash" and "hash" set (for appending to the chain)."""
    out = dict(record)
    out["prev_hash"] = prev_hash
    out["hash"] = record_hash(prev_hash, out, blobs)
    return out


def make_review_event(seq: int, at: str, target_seq: int, label: str) -> dict:
    """Unsealed review event; pass it to `seal_record` to chain it."""
    if label not in REVIEW_LABELS:
        raise ValueError(f"unknown review label: {label!r}")
    return {"seq": seq, "at": at, "type": "review", "target_seq": target_seq, "label": label}


def is_review(record: dict) -> bool:
    return record.get("type") == "review"


def is_inspection(record: dict) -> bool:
    return "type" not in record or record.get("type") in (None, "inspection")


def verify_chain(records: list[dict],
                 blobs_for: Callable[[dict], Iterable[bytes] | None] | None = None,
                 ) -> tuple[bool, int | None]:
    """Recompute the chain in list order.

    `blobs_for(record)` returns the blobs that were hashed with that record (or None).
    Returns (True, None) if intact, else (False, seq of the first bad record). A record
    is bad if its prev_hash does not link to the previous hash, its stored hash does not
    match, its blobs cannot be loaded, or it is a review of an unknown earlier inspection.
    """
    prev = GENESIS_HASH
    seen_inspections: set = set()
    for rec in records:
        seq = rec.get("seq")
        if rec.get("prev_hash") != prev:
            return False, seq
        try:
            blobs = list(blobs_for(rec) or ()) if blobs_for is not None else []
        except (OSError, KeyError, ValueError):
            return False, seq
        if rec.get("hash") != record_hash(prev, rec, blobs):
            return False, seq
        if is_review(rec):
            if rec.get("target_seq") not in seen_inspections or rec.get("label") not in REVIEW_LABELS:
                return False, seq
        elif is_inspection(rec):
            seen_inspections.add(seq)
        prev = rec["hash"]
    return True, None


def check_links(records: list[dict], expected_head: str | None = None) -> tuple[bool, int | None]:
    """Cheap structural check without blobs: prev_hash links and (optionally) the head hash."""
    prev = GENESIS_HASH
    for rec in records:
        if rec.get("prev_hash") != prev or not isinstance(rec.get("hash"), str):
            return False, rec.get("seq")
        prev = rec["hash"]
    if expected_head and expected_head != prev:
        return False, records[-1].get("seq") if records else None
    return True, None


def chain_head(records: list[dict]) -> str:
    """Hash of the last record (genesis hash for an empty journal)."""
    return records[-1].get("hash", GENESIS_HASH) if records else GENESIS_HASH


# --------------------------------------------------------------------------- stats

def effective_reviews(records: list[dict]) -> dict:
    """seq -> latest operator label. Review events win over an inspection's own "review" field."""
    out: dict = {}
    for rec in records:
        if is_inspection(rec) and rec.get("review"):
            out[rec.get("seq")] = rec["review"]
    for rec in records:
        if is_review(rec):
            out[rec.get("target_seq")] = rec.get("label")
    return out


def shift_stats(records: list[dict]) -> dict:
    """Counts in the station JSON "stats" shape, computed from the journal."""
    inspections = [r for r in records if is_inspection(r)]
    reviews = effective_reviews(records)
    seqs = {r.get("seq") for r in inspections}
    labels = [lab for s, lab in reviews.items() if s in seqs]
    return {
        "total": len(inspections),
        "pass": sum(r.get("verdict") == "pass" for r in inspections),
        "fail": sum(r.get("verdict") == "fail" for r in inspections),
        "retake": sum(r.get("verdict") == "retake" for r in inspections),
        "confirmed_defects": labels.count("defect"),
        "false_alarms": labels.count("false_alarm"),
    }


# --------------------------------------------------------------------------- time / formatting

def parse_time(at: Any) -> datetime | None:
    """ISO 8601 string (naive = UTC, 'Z' accepted) or epoch seconds -> aware UTC datetime."""
    if at is None or at == "":
        return None
    try:
        if isinstance(at, (int, float)):
            return datetime.fromtimestamp(float(at), tz=timezone.utc)
        dt = datetime.fromisoformat(str(at).strip().replace("Z", "+00:00"))
    except (ValueError, OverflowError, OSError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_local(at: Any) -> str:
    dt = parse_time(at)
    return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S") if dt else ""


def format_utc(at: Any) -> str:
    dt = parse_time(at)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC") if dt else ""


def _timing_total(timing: Any) -> float | None:
    if not isinstance(timing, dict) or not timing:
        return None
    if isinstance(timing.get("total"), (int, float)):
        return float(timing["total"])
    vals = [float(v) for v in timing.values() if isinstance(v, (int, float))]
    return sum(vals) if vals else None


def _top_bbox(rec: dict) -> list | None:
    zones = rec.get("zones") or []
    if zones and isinstance(zones[0], dict) and zones[0].get("bbox") is not None:
        return list(zones[0]["bbox"])
    return None


# --------------------------------------------------------------------------- CSV

def journal_csv(station: dict, records: list[dict]) -> str:
    """Inspection rows as CSV text, UTF-8 BOM prefixed so Excel reads Cyrillic correctly.

    `review` is the latest operator label (review events applied). `time_local` is
    Asia/Tashkent. Review events themselves are not rows (their effect is the column).
    """
    reviews = effective_reviews(records)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(CSV_COLUMNS)
    for rec in records:
        if not is_inspection(rec):
            continue
        bbox = _top_bbox(rec)
        total = _timing_total(rec.get("timing_ms"))
        w.writerow([
            rec.get("seq", ""),
            format_local(rec.get("at")),
            rec.get("verdict") or "",
            rec.get("reason") or "",
            len(rec.get("zones") or []),
            " ".join(str(int(v)) for v in bbox) if bbox else "",
            reviews.get(rec.get("seq")) or "",
            rec.get("method") or "",
            f"{total:.1f}" if total is not None else "",
            rec.get("hash") or "",
        ])
    return "﻿" + buf.getvalue()


# --------------------------------------------------------------------------- HTML report

VERDICT_RU = {"pass": "ГОДНА", "fail": "БРАК", "retake": "ПЕРЕСНЯТЬ"}
REASON_RU = {
    "alignment": "не совмещается с эталоном",
    "glare": "блик",
    "blurry": "размыто",
    "dark": "слишком темно",
    "low_coverage": "мало проверенной площади",
    "mismatch": "не та деталь или деталь отсутствует",
}
REVIEW_RU = {"defect": "дефект подтверждён", "false_alarm": "ложная тревога", "ok": "годна (оператор)"}

_CSS = """
:root{--ink:#111;--muted:#555;--line:#bbb;--pass:#0a7a32;--fail:#b3001b;--retake:#666;}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:#fff;color:var(--ink);
 font:12px/1.4 -apple-system,"Segoe UI",Roboto,Arial,sans-serif}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:14px;margin:16px 0 6px}
.meta{color:var(--muted);margin:0 0 2px}
.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin:10px 0}
.stat{border:1px solid var(--line);border-radius:4px;padding:6px;text-align:center}
.stat b{display:block;font-size:18px}
table{width:100%;border-collapse:collapse}
th,td{border:1px solid var(--line);padding:3px 5px;text-align:left;vertical-align:top}
th{background:#eee}
tr{page-break-inside:avoid;break-inside:avoid}
thead{display:table-header-group}
.v-pass{color:var(--pass);font-weight:700}
.v-fail{color:var(--fail);font-weight:700}
.v-retake{color:var(--retake);font-weight:700}
.hash{font-family:Consolas,"Courier New",monospace;font-size:10px;word-break:break-all}
.thumb{max-width:42mm;max-height:30mm;display:block}
.chain-ok{color:var(--pass);font-weight:700}
.chain-bad{color:var(--fail);font-weight:700}
.note{margin-top:14px;color:var(--muted)}
.foot{margin-top:6px;font-size:11px;color:var(--muted)}
@page{size:A4;margin:12mm}
@media print{body{padding:0}}
"""


def _e(value: Any) -> str:
    """HTML-escape any value (None -> empty)."""
    return html.escape("" if value is None else str(value), quote=True)


def _safe_src(src: Any) -> str | None:
    """Accept only image data URIs and http(s)/relative URLs (no javascript: etc.)."""
    if not isinstance(src, str) or not src:
        return None
    s = src.strip()
    low = s.lower()
    if low.startswith("data:image/"):
        return s
    if low.startswith(("http://", "https://", "/")) or ":" not in s.split("/", 1)[0]:
        return s
    return None


def _thumb_for(rec: dict, images: dict) -> str | None:
    """Look up a thumbnail by seq (str or int), then by the record's overlay_url."""
    images = images or {}
    for key in (str(rec.get("seq")), rec.get("seq"), rec.get("overlay_url")):
        if key is not None and key in images:
            return _safe_src(images[key])
    return None


def render_shift_report(station: dict, records: list[dict], images: dict[str, str] | None = None,
                        blobs_for: Callable[[dict], Iterable[bytes] | None] | None = None) -> str:
    """Standalone printable shift report (Russian, A4 print CSS, no external resources).

    `images` maps an inspection's seq (as str) or its overlay_url to an image source
    (data URI or URL) used as a thumbnail for fail rows. With `blobs_for` the chain is
    fully verified (hashes + blobs); without it only links and the head hash are checked.
    All station/record text is HTML-escaped.
    """
    station = station or {}
    images = images or {}
    inspections = [r for r in records if is_inspection(r)]
    reviews = effective_reviews(records)
    stats = shift_stats(records)
    head = chain_head(records)

    if blobs_for is not None:
        ok, bad = verify_chain(records, blobs_for)
        if ok and station.get("chain_head") and records and station["chain_head"] != head:
            ok, bad = False, records[-1].get("seq")
        scope = "хэши и файлы пересчитаны"
    else:
        ok, bad = check_links(records, station.get("chain_head") if records else None)
        scope = "проверены связи записей, без пересчёта файлов"
    chain_txt = (f'<span class="chain-ok">цепочка проверена</span> ({scope})' if ok else
                 f'<span class="chain-bad">цепочка нарушена</span> — первая неверная запись № {_e(bad)}')

    times = [r.get("at") for r in records if parse_time(r.get("at"))]
    start = times[0] if times else station.get("created_at")
    end = times[-1] if times else None

    def when(at: Any) -> str:
        return f"{_e(format_local(at))} (Ташкент) / {_e(format_utc(at))}" if parse_time(at) else "—"

    has_baseline = any(isinstance(r.get("baseline"), dict) for r in inspections)

    rows = []
    for rec in inspections:
        verdict = rec.get("verdict")
        v_cls = f"v-{verdict}" if verdict in VERDICT_RU else ""
        v_txt = VERDICT_RU.get(verdict, verdict or "—")
        n_zones = len(rec.get("zones") or [])
        if verdict == "fail" and n_zones:
            v_txt = f"{v_txt} — {n_zones} зон"
        reason = rec.get("reason")
        reason_txt = REASON_RU.get(reason, reason) if reason else ""
        label = reviews.get(rec.get("seq"))
        total = _timing_total(rec.get("timing_ms"))
        thumb = ""
        if verdict == "fail":
            src = _thumb_for(rec, images)
            if src:
                thumb = f'<img class="thumb" src="{_e(src)}" alt="зоны дефекта № {_e(rec.get("seq"))}">'
        cells = [
            _e(rec.get("seq")),
            _e(format_local(rec.get("at"))),
            f'<span class="{v_cls}">{_e(v_txt)}</span>',
            _e(reason_txt),
            _e(REVIEW_RU.get(label, label) if label else ""),
            _e(rec.get("method")),
            _e(f"{total:.0f}" if total is not None else ""),
        ]
        if has_baseline:
            b = rec.get("baseline") if isinstance(rec.get("baseline"), dict) else None
            cells.append(_e(VERDICT_RU.get(b.get("verdict"), b.get("verdict")) if b else ""))
        cells.append(f'<span class="hash" title="{_e(rec.get("hash"))}">{_e((rec.get("hash") or "")[:12])}</span>')
        cells.append(thumb)
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    headers = ["№", "Время (Ташкент)", "Вердикт", "Причина", "Оператор", "Метод", "мс"]
    if has_baseline:
        headers.append("Простое вычитание")
    headers += ["Хэш", "Снимок"]
    table = ('<table><thead><tr>' + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"
             + ("".join(rows) or f'<tr><td colspan="{len(headers)}">Проверок нет</td></tr>')
             + "</tbody></table>")

    review_rows = [
        f"<tr><td>{_e(r.get('seq'))}</td><td>{_e(format_local(r.get('at')))}</td>"
        f"<td>{_e(r.get('target_seq'))}</td><td>{_e(REVIEW_RU.get(r.get('label'), r.get('label')))}</td></tr>"
        for r in records if is_review(r)
    ]
    review_block = ""
    if review_rows:
        review_block = ("<h2>Отметки оператора</h2><table><thead><tr><th>№ события</th><th>Время (Ташкент)</th>"
                        "<th>Проверка №</th><th>Отметка</th></tr></thead><tbody>"
                        + "".join(review_rows) + "</tbody></table>")

    tiles = [("Всего", stats["total"]), ("Годна", stats["pass"]), ("Брак", stats["fail"]),
             ("Переснять", stats["retake"]), ("Дефект подтверждён", stats["confirmed_defects"]),
             ("Ложная тревога", stats["false_alarms"])]
    stats_html = '<div class="stats">' + "".join(
        f'<div class="stat"><b>{_e(v)}</b>{_e(k)}</div>' for k, v in tiles) + "</div>"

    name = station.get("name") or "Станция"
    title = f"Отчёт смены — {name}"
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Dalil QA — отчёт смены</h1>
<p class="meta">Станция: <b>{_e(name)}</b> (код {_e(station.get("id") or "—")}) · Артикул (SKU): {_e(station.get("sku") or "—")}</p>
<p class="meta">Начало смены: {when(start)}</p>
<p class="meta">Конец смены: {when(end)}</p>
<p class="meta">Отчёт сформирован: {when(datetime.now(timezone.utc).isoformat())}</p>
<h2>Итоги</h2>
{stats_html}
<h2>Проверки</h2>
{table}
{review_block}
<h2>Целостность журнала</h2>
<p>Хэш-цепочка SHA-256: {chain_txt}.</p>
<p>Голова цепочки: <span class="hash">{_e(head)}</span></p>
<p class="note">{_e(METHOD_NOTE)}</p>
<p class="foot">* {_e(FOOTNOTE)}.</p>
</body>
</html>
"""
