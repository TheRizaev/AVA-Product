"""Dalil QA HTTP API (docs/SPEC.md "HTTP API") + the static web UI at "/".

Run: ``uvicorn server.app:app --host 0.0.0.0 --port 7860``. Data lives in
``DALIL_DATA_DIR`` (default ``./_data``). Errors are JSON ``{"error": code, "message": text}``.
"""

from __future__ import annotations

import base64
import io
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from dalil import __version__, baselines, pipeline, report
from dalil.align import align
from dalil.config import Params

from .store import SID_RE, Store, StoreError

MAX_UPLOAD = 15 * 1024 * 1024          # bytes per uploaded image
MAX_BODY = 2 * MAX_UPLOAD + 64 * 1024  # whole request (compare carries two images)
MAX_PIXELS = 40_000_000                # decoded image size guard (decompression bombs)
DECODE_SIDE = 2048                     # JPEGs are decoded at reduced scale down to about this
CPU_SLOTS = 2                          # concurrent CPU-heavy requests (PBP itself is multi-threaded)
THUMB_SIDE = 360                       # report thumbnails
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

PARAMS = Params()
_cpu = threading.BoundedSemaphore(CPU_SLOTS)
_store: Store | None = None


def get_store() -> Store:
    """The process-wide store (created lazily so DALIL_DATA_DIR can be set before first use)."""
    global _store
    if _store is None:
        _store = Store(params=PARAMS)
    return _store


def _warm_up() -> None:
    """Pay OpenCV's one-time ORB/ECC initialisation before the first real request."""
    rng = np.random.default_rng(0)
    a = cv2.GaussianBlur(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), (0, 0), 2)
    align(a, np.roll(a, 3, axis=1), PARAMS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_store()
    await run_in_threadpool(_warm_up)
    yield


app = FastAPI(title="Dalil QA", version=__version__, lifespan=lifespan,
              docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")


# --------------------------------------------------------------------------- errors

class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class _BodyTooLarge(Exception):
    pass


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": code, "message": message}, status_code=status,
                        headers={"Cache-Control": "no-store"})


@app.exception_handler(ApiError)
@app.exception_handler(StoreError)
async def _api_error(_req: Request, exc) -> JSONResponse:
    return _error(exc.status, exc.code, exc.message)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_req: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = {404: "not_found", 405: "method_not_allowed", 413: "too_large"}.get(exc.status_code, "http_error")
    return _error(exc.status_code, code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def _validation_error(_req: Request, exc: RequestValidationError) -> JSONResponse:
    msg = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg')}" for e in exc.errors())
    return _error(400, "invalid_request", msg or "Invalid request.")


@app.exception_handler(Exception)
async def _internal_error(_req: Request, exc: Exception) -> JSONResponse:
    return _error(500, "internal", f"Internal error: {type(exc).__name__}")


# --------------------------------------------------------------------------- middleware

class BodyLimit:
    """Reject request bodies above MAX_BODY: by Content-Length up front, else while streaming."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        length = headers.get(b"content-length")
        if length is not None and (not length.isdigit() or int(length) > MAX_BODY):
            return await _error(413, "too_large", "Upload is larger than 15 MB.")(scope, receive, send)

        received = 0
        started = False

        async def limited_receive():
            nonlocal received
            msg = await receive()
            if msg["type"] == "http.request":
                received += len(msg.get("body", b""))
                if received > MAX_BODY:
                    raise _BodyTooLarge
            return msg

        async def tracking_send(msg):
            nonlocal started
            if msg["type"] == "http.response.start":
                started = True
            await send(msg)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await _error(413, "too_large", "Upload is larger than 15 MB.")(scope, receive, send)


@app.middleware("http")
async def _cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    else:  # static UI: always revalidate so a redeploy is picked up
        response.headers.setdefault("Cache-Control", "no-cache")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    return response


app.add_middleware(BodyLimit)


# --------------------------------------------------------------------------- helpers

def _sid(sid: str) -> str:
    if not SID_RE.match(sid or ""):
        raise ApiError(400, "bad_station_id", "Station code must be 6 characters A-Z / 2-9.")
    return sid


def decode_image(data: bytes) -> np.ndarray:
    """Uploaded bytes -> BGR uint8, EXIF orientation applied, very large JPEGs decoded downscaled."""
    if not data:
        raise ApiError(400, "empty_file", "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD:
        raise ApiError(413, "too_large", "Upload is larger than 15 MB.")
    try:
        im = Image.open(io.BytesIO(data))
        if im.width * im.height > MAX_PIXELS:
            raise ApiError(413, "too_many_pixels", "Image resolution is too large.")
        im.draft("RGB", (DECODE_SIDE, DECODE_SIDE))  # JPEG: decode at 1/2, 1/4, 1/8 scale
        im = ImageOps.exif_transpose(im)
        rgb = np.asarray(im.convert("RGB"))
    except ApiError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ApiError(400, "bad_image", "Could not read the image (JPEG or PNG expected).") from None
    if min(rgb.shape[:2]) < 64:
        raise ApiError(400, "image_too_small", "Image is too small (min 64 px).")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


async def _form_files(request: Request, names: tuple[str, ...]) -> dict[str, bytes]:
    """Read the named multipart file fields (each at most MAX_UPLOAD bytes)."""
    ctype = request.headers.get("content-type", "")
    if not ctype.startswith("multipart/form-data"):
        raise ApiError(400, "multipart_required", "Send the image as multipart/form-data.")
    try:
        form = await request.form(max_files=len(names) + 1, max_fields=8)
    except StarletteHTTPException as exc:
        raise ApiError(400, "bad_form", str(exc.detail)) from None
    out = {}
    try:
        for name in names:
            item = form.get(name)
            if item is None or isinstance(item, str):
                raise ApiError(400, "file_required", f"Multipart field '{name}' (an image file) is required.")
            data = await item.read(MAX_UPLOAD + 1)
            if len(data) > MAX_UPLOAD:
                raise ApiError(413, "too_large", "Upload is larger than 15 MB.")
            out[name] = data
    finally:
        await form.close()
    return out


async def _json_body(request: Request) -> dict:
    try:
        body = json.loads(await request.body() or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ApiError(400, "bad_json", "Request body must be JSON.") from None
    if not isinstance(body, dict):
        raise ApiError(400, "bad_json", "Request body must be a JSON object.")
    return body


def _cpu_bound(fn, *args):
    """Run CPU-heavy work in the thread pool, at most CPU_SLOTS at a time."""
    def run():
        with _cpu:
            return fn(*args)
    return run_in_threadpool(run)


def _thumb_data_uri(path: Path) -> str | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = min(1.0, THUMB_SIDE / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


# --------------------------------------------------------------------------- routes

@app.get("/api/health")
async def health():
    return {"ok": True, "version": __version__, "params": PARAMS.to_dict()}


@app.post("/api/stations")
async def create_station(request: Request):
    body = await _json_body(request)
    name, sku = body.get("name"), body.get("sku")
    if not isinstance(name, str) or (sku is not None and not isinstance(sku, str)):
        raise ApiError(400, "invalid_request", "Body must be {\"name\": str, \"sku\": str?}.")
    return await run_in_threadpool(get_store().create_station, name, sku)


@app.get("/api/stations/{sid}")
async def get_station(sid: str):
    return await run_in_threadpool(get_store().station_json, _sid(sid))


@app.post("/api/stations/{sid}/reference")
async def add_reference(sid: str, request: Request):
    _sid(sid)
    data = (await _form_files(request, ("file",)))["file"]
    bgr = await run_in_threadpool(decode_image, data)
    return await _cpu_bound(get_store().add_reference, sid, bgr)


@app.delete("/api/stations/{sid}/reference")
async def clear_reference(sid: str):
    return await run_in_threadpool(get_store().clear_references, _sid(sid))


@app.post("/api/stations/{sid}/inspect")
async def inspect(sid: str, request: Request, compare: int = 0):
    _sid(sid)
    data = (await _form_files(request, ("file",)))["file"]
    bgr = await run_in_threadpool(decode_image, data)
    return await _cpu_bound(get_store().inspect, sid, bgr, bool(compare))


@app.get("/api/stations/{sid}/inspections")
async def inspections(sid: str, after: int = 0, limit: int = 50):
    if after < 0 or not 1 <= limit <= 200:
        raise ApiError(400, "invalid_request", "after must be >= 0 and limit in 1..200.")
    return await run_in_threadpool(get_store().inspections, _sid(sid), after, limit)


@app.post("/api/stations/{sid}/inspections/{seq}/review")
async def review(sid: str, seq: int, request: Request):
    body = await _json_body(request)
    label = body.get("label")
    if label not in report.REVIEW_LABELS:
        raise ApiError(400, "bad_label", "label must be one of: " + ", ".join(report.REVIEW_LABELS))
    return await run_in_threadpool(get_store().review, _sid(sid), seq, label)


@app.get("/api/stations/{sid}/journal.csv")
async def journal_csv(sid: str):
    station, records = await run_in_threadpool(get_store().records, _sid(sid))
    text = report.journal_csv(station, records)
    return Response(text.encode("utf-8"), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="dalil_{sid}_journal.csv"'})


def _render_report(sid: str) -> str:
    store = get_store()
    station, records = store.records(sid)
    images = {}
    for rec in records:
        if report.is_inspection(rec) and rec.get("verdict") == "fail":
            uri = _thumb_data_uri(store.root / sid / f"o_{int(rec['seq'])}.jpg")
            if uri:
                images[str(rec["seq"])] = uri
    return report.render_shift_report(station, records, images, blobs_for=store.blobs_for(sid))


@app.get("/api/stations/{sid}/report")
async def shift_report(sid: str):
    html = await run_in_threadpool(_render_report, _sid(sid))
    return HTMLResponse(html)


@app.get("/api/files/{sid}/{name}")
async def files(sid: str, name: str):
    path = get_store().file_path(_sid(sid), name)
    media = "image/png" if name.endswith(".png") else "image/jpeg"
    return FileResponse(path, media_type=media)


def _compare(ref_bgr: np.ndarray, frame_bgr: np.ndarray, methods: tuple) -> dict:
    ref = pipeline.prepare_reference([ref_bgr], PARAMS, methods)
    results = pipeline.inspect_compare(ref, frame_bgr, PARAMS, methods)
    return {
        "results": {m: pipeline.result_to_json(r) for m, r in results.items()},
        "overlays": {m: base64.b64encode(r.images.get("overlay", b"")).decode("ascii")
                     for m, r in results.items()},
    }


@app.post("/api/compare")
async def compare(request: Request, methods: str = "pbp,pixdiff"):
    known = baselines.get_methods()
    names = tuple(dict.fromkeys(m.strip() for m in methods.split(",") if m.strip()))
    if not names or len(names) > len(known) or any(m not in known for m in names):
        raise ApiError(400, "bad_methods", "methods must be a comma list of: " + ", ".join(known))
    files = await _form_files(request, ("reference", "frame"))
    ref_bgr = await run_in_threadpool(decode_image, files["reference"])
    frame_bgr = await run_in_threadpool(decode_image, files["frame"])
    return await _cpu_bound(_compare, ref_bgr, frame_bgr, names)


# Static UI last, so it never shadows an API route.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
