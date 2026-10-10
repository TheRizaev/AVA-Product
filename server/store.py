"""On-disk station storage + in-memory cache of prepared references.

Layout under ``DALIL_DATA_DIR`` (default ``./_data``)::

    <sid>/station.json    name, sku, created_at, golden shots (idx, file, at), ref_version
    <sid>/journal.jsonl   append-only, hash-chained journal (dalil.report): inspections + review events
    <sid>/ref_<i>.jpg     golden shots (i = 0..4)
    <sid>/outline.png     ghost outline of shot 0 for camera positioning
    <sid>/f_<seq>.jpg     aligned frame of inspection <seq>
    <sid>/o_<seq>.jpg     PBP overlay of inspection <seq>
    <sid>/b_<seq>.jpg     baseline (pixdiff) overlay of inspection <seq>, if compared

Every mutation of a station happens under that station's lock; the CPU-heavy
inspection itself runs outside the lock and only the journal append is serialised.
Inspections and review events share one hash chain; inspection `seq` and review
event `seq` are numbered independently (1, 2, ...).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from dalil import pipeline, report
from dalil.config import Params
from dalil.quality import capture_report

SID_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # [A-Z2-9] without the look-alikes I and O
SID_RE = re.compile(r"^[A-Z2-9]{6}$")
FILE_RE = re.compile(r"^(?:ref_[0-9]\.jpg|outline\.png|[fob]_[1-9][0-9]{0,8}\.jpg)$")
REF_JPEG_QUALITY = 92
MAX_NAME = 80
MAX_SKU = 40


class StoreError(Exception):
    """Client-visible error: HTTP status, machine code and a human message."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _jpeg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _clean_text(value, limit: int) -> str:
    """Single-line, trimmed, length-limited user text (HTML escaping happens on output)."""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or "")).strip()
    return text[:limit]


class Store:
    """Station storage. One instance per process (see server.app)."""

    def __init__(self, root: str | os.PathLike | None = None, params: Params | None = None,
                 cache_size: int = 3):
        self.root = Path(root or os.environ.get("DALIL_DATA_DIR") or "./_data").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.params = params or Params()
        self.cache_size = max(1, int(cache_size))
        self._global = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}
        self._journals: dict[str, list] = {}
        self._refs: OrderedDict[str, tuple[int, pipeline.Reference]] = OrderedDict()

    # ------------------------------------------------------------------ helpers

    def _lock(self, sid: str) -> threading.RLock:
        with self._global:
            return self._locks.setdefault(sid, threading.RLock())

    def _dir(self, sid: str) -> Path:
        if not SID_RE.match(sid or ""):
            raise StoreError(400, "bad_station_id", "Station code must be 6 characters A-Z / 2-9.")
        return self.root / sid

    def _load_station(self, sid: str) -> dict:
        path = self._dir(sid) / "station.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise StoreError(404, "station_not_found", f"Station {sid} not found.") from None

    def _save_station(self, st: dict) -> None:
        data = json.dumps(st, ensure_ascii=False, indent=1).encode("utf-8")
        _write_atomic(self._dir(st["id"]) / "station.json", data)

    def _journal(self, sid: str) -> list:
        """All journal records in chain order (cached in memory; caller holds the lock)."""
        records = self._journals.get(sid)
        if records is None:
            records = []
            path = self._dir(sid) / "journal.jsonl"
            if path.exists():
                with path.open(encoding="utf-8") as fh:
                    records = [json.loads(line) for line in fh if line.strip()]
            self._journals[sid] = records
        return records

    def _append(self, sid: str, record: dict, blobs: list[bytes]) -> dict:
        """Seal `record` onto the chain and append it to journal.jsonl (caller holds the lock)."""
        records = self._journal(sid)
        sealed = report.seal_record(report.chain_head(records), record, blobs)
        with (self._dir(sid) / "journal.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(report.canonical_json(sealed) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        records.append(sealed)
        return sealed

    @staticmethod
    def file_url(sid: str, name: str) -> str:
        return f"/api/files/{sid}/{name}"

    # ------------------------------------------------------------------ stations

    def create_station(self, name, sku=None) -> dict:
        name = _clean_text(name, MAX_NAME)
        if not name:
            raise StoreError(400, "name_required", "Station name is required.")
        sku = _clean_text(sku, MAX_SKU) or None
        for _ in range(50):
            sid = "".join(secrets.choice(SID_ALPHABET) for _ in range(6))
            try:
                self._dir(sid).mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                continue
            st = {"id": sid, "name": name, "sku": sku, "created_at": now_iso(),
                  "references": [], "ref_version": 0}
            with self._lock(sid):
                self._save_station(st)
            return self.station_json(sid)
        raise StoreError(503, "no_station_id", "Could not allocate a station code, try again.")

    def station_json(self, sid: str, **extra) -> dict:
        """Station JSON of docs/SPEC.md (plus `extra` keys, e.g. last_capture)."""
        with self._lock(sid):
            st = self._load_station(sid)
            records = self._journal(sid)
            refs = [{"idx": r["idx"], "url": self.file_url(sid, r["file"]), "at": r["at"]}
                    for r in st["references"]]
            outline = (self._dir(sid) / "outline.png").exists() and bool(refs)
            out = {
                "id": sid, "name": st["name"], "sku": st.get("sku"), "created_at": st["created_at"],
                "references": refs,
                "outline_url": self.file_url(sid, "outline.png") if outline else None,
                "stats": report.shift_stats(records),
                "chain_head": report.chain_head(records),
                "last_seq": self._last_seq(records),
            }
        out.update(extra)
        return out

    @staticmethod
    def _last_seq(records: list) -> int:
        seqs = [r["seq"] for r in records if report.is_inspection(r)]
        return max(seqs) if seqs else 0

    # ------------------------------------------------------------------ references

    def _ref_images(self, sid: str, st: dict) -> list:
        imgs = []
        for r in st["references"]:
            img = cv2.imread(str(self._dir(sid) / r["file"]), cv2.IMREAD_COLOR)
            if img is None:
                raise StoreError(500, "reference_unreadable", f"Reference file {r['file']} is unreadable.")
            imgs.append(img)
        return imgs

    def add_reference(self, sid: str, bgr: np.ndarray) -> dict:
        """Add a golden shot (max params.max_references); returns station JSON + last_capture."""
        with self._lock(sid):
            st = self._load_station(sid)
            n = len(st["references"])
            if n >= self.params.max_references:
                raise StoreError(409, "too_many_references",
                                 f"At most {self.params.max_references} reference shots; clear them first.")
            # Store exactly what will be reloaded later (JPEG round trip), then prepare.
            jpeg = _jpeg(bgr, REF_JPEG_QUALITY)
            shot = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            try:
                ref = pipeline.prepare_reference(self._ref_images(sid, st) + [shot], self.params)
            except ValueError as exc:
                raise StoreError(422, "reference_mismatch",
                                 "This shot does not match the first reference shot "
                                 f"(same part, same camera position): {exc}") from None
            name = f"ref_{n}.jpg"
            _write_atomic(self._dir(sid) / name, jpeg)
            if n == 0:
                _write_atomic(self._dir(sid) / "outline.png", ref.outline_png)
            st["references"].append({"idx": n, "file": name, "at": now_iso()})
            st["ref_version"] = int(st.get("ref_version", 0)) + 1
            self._save_station(st)
            self._cache_put(sid, st["ref_version"], ref)
            capture = capture_report(shot, self.params)
        return self.station_json(sid, last_capture=_json_safe(capture))

    def clear_references(self, sid: str) -> dict:
        with self._lock(sid):
            st = self._load_station(sid)
            for r in st["references"]:
                (self._dir(sid) / r["file"]).unlink(missing_ok=True)
            (self._dir(sid) / "outline.png").unlink(missing_ok=True)
            st["references"] = []
            st["ref_version"] = int(st.get("ref_version", 0)) + 1
            self._save_station(st)
            with self._global:
                self._refs.pop(sid, None)
        return self.station_json(sid)

    def _cache_put(self, sid: str, version: int, ref) -> None:
        with self._global:
            self._refs[sid] = (version, ref)
            self._refs.move_to_end(sid)
            while len(self._refs) > self.cache_size:
                self._refs.popitem(last=False)

    def reference(self, sid: str) -> pipeline.Reference:
        """Prepared golden set (LRU cache, invalidated when the shots change)."""
        with self._lock(sid):
            st = self._load_station(sid)
            if not st["references"]:
                raise StoreError(409, "no_reference", "Shoot the reference (golden part) first.")
            version = int(st.get("ref_version", 0))
            with self._global:
                hit = self._refs.get(sid)
                if hit is not None and hit[0] == version:
                    self._refs.move_to_end(sid)
                    return hit[1]
            try:
                ref = pipeline.prepare_reference(self._ref_images(sid, st), self.params)
            except ValueError as exc:
                raise StoreError(409, "reference_invalid", str(exc)) from None
            self._cache_put(sid, version, ref)
            return ref

    # ------------------------------------------------------------------ inspections

    def inspect(self, sid: str, bgr: np.ndarray, compare: bool = False) -> dict:
        """Inspect a frame, store its images and append the sealed record; returns record JSON."""
        ref = self.reference(sid)
        methods = ("pbp", "pixdiff") if compare else ("pbp",)
        results = pipeline.inspect_compare(ref, bgr, self.params, methods)
        res = results["pbp"]
        base = results.get("pixdiff")
        frame_jpeg, overlay_jpeg = res.images["frame"], res.images["overlay"]
        base_jpeg = base.images["overlay"] if base is not None else None

        with self._lock(sid):
            self._load_station(sid)  # still exists?
            records = self._journal(sid)
            seq = self._last_seq(records) + 1
            d = self._dir(sid)
            _write_atomic(d / f"f_{seq}.jpg", frame_jpeg)
            _write_atomic(d / f"o_{seq}.jpg", overlay_jpeg)
            blobs = [frame_jpeg, overlay_jpeg]
            baseline = None
            if base is not None:
                _write_atomic(d / f"b_{seq}.jpg", base_jpeg)
                blobs.append(base_jpeg)
                baseline = {"method": "pixdiff", "verdict": base.verdict, "reason": base.reason,
                            "n_zones": len(base.zones), "threshold": base.threshold,
                            "overlay_url": self.file_url(sid, f"b_{seq}.jpg")}
            body = pipeline.result_to_json(res)
            record = {
                "seq": seq, "at": now_iso(),
                "verdict": body["verdict"], "reason": body["reason"], "zones": body["zones"],
                "coverage": body["coverage"], "changed_frac": body["changed_frac"],
                "timing_ms": body["timing_ms"], "align": body["align"], "method": body["method"],
                "threshold": body["threshold"], "hints": body["hints"],
                "signature_sha256": body["signature_sha256"], "work_size": body["work_size"],
                "frame_url": self.file_url(sid, f"f_{seq}.jpg"),
                "overlay_url": self.file_url(sid, f"o_{seq}.jpg"),
                "baseline": baseline, "review": None,
            }
            sealed = self._append(sid, record, blobs)
        return dict(sealed)

    def review(self, sid: str, seq: int, label: str) -> dict:
        """Append an operator review event; returns the reviewed inspection record JSON."""
        if label not in report.REVIEW_LABELS:
            raise StoreError(400, "bad_label", "label must be one of: " + ", ".join(report.REVIEW_LABELS))
        with self._lock(sid):
            self._load_station(sid)
            records = self._journal(sid)
            target = next((r for r in records if report.is_inspection(r) and r["seq"] == seq), None)
            if target is None:
                raise StoreError(404, "inspection_not_found", f"Inspection {seq} not found.")
            n_reviews = sum(1 for r in records if report.is_review(r))
            event = report.make_review_event(n_reviews + 1, now_iso(), seq, label)
            self._append(sid, event, [])
            return self._with_review(target, report.effective_reviews(records))

    @staticmethod
    def _with_review(rec: dict, reviews: dict) -> dict:
        out = dict(rec)
        out["review"] = reviews.get(rec["seq"])
        return out

    def inspections(self, sid: str, after: int = 0, limit: int = 50) -> dict:
        """Inspection records with seq > after (oldest first, at most `limit`) + last_seq."""
        with self._lock(sid):
            self._load_station(sid)
            records = self._journal(sid)
            reviews = report.effective_reviews(records)
            items = [self._with_review(r, reviews) for r in records
                     if report.is_inspection(r) and r["seq"] > after][:limit]
            return {"items": items, "last_seq": self._last_seq(records)}

    def records(self, sid: str) -> tuple[dict, list]:
        """(station JSON, full journal in chain order) for CSV / report."""
        with self._lock(sid):
            station = self.station_json(sid)
            return station, list(self._journal(sid))

    def blobs_for(self, sid: str):
        """Loader of the blobs hashed with each record (for report.verify_chain)."""
        d = self._dir(sid)

        def load(rec: dict) -> list[bytes]:
            if not report.is_inspection(rec):
                return []
            seq = int(rec["seq"])
            names = [f"f_{seq}.jpg", f"o_{seq}.jpg"] + ([f"b_{seq}.jpg"] if rec.get("baseline") else [])
            return [(d / n).read_bytes() for n in names]

        return load

    def file_path(self, sid: str, name: str) -> Path:
        """Path of a stored image; strict whitelist (no traversal possible)."""
        if not FILE_RE.match(name or ""):
            raise StoreError(404, "file_not_found", "File not found.")
        path = self._dir(sid) / name
        if not path.is_file():
            raise StoreError(404, "file_not_found", "File not found.")
        return path


def _json_safe(obj):
    """numpy scalars -> Python (capture reports contain np.float/np.bool_)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
