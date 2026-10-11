"""AVA HTTP API (docs/SPEC.md "HTTP API", "HTTP API addition for Mode 1") + the static web UI at "/".

Routes: ``GET /api/health``, ``POST /api/compare`` (Mode 2: reference + frame),
``POST /api/scan`` (Mode 1: one photo of a uniform surface).

Run: ``uvicorn server.app:app --host 0.0.0.0 --port 7860``. Errors are JSON
``{"error": code, "message": text}``. User photos live only in memory for the request;
prepared references are kept in a small in-memory LRU (never on disk).
"""

from __future__ import annotations

import base64
import hashlib
import io
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartParser

from ava import __version__, baselines, pipeline, surface
from ava.align import align
from ava.config import Params

MAX_UPLOAD = 15 * 1024 * 1024          # bytes per uploaded image
MAX_BODY = 2 * MAX_UPLOAD + 64 * 1024  # whole request (compare carries two images)
MAX_PIXELS = 40_000_000                # decoded image size guard (decompression bombs)
DECODE_SIDE = 2048                     # JPEGs are decoded at reduced scale down to about this (long side)
IMAGE_FORMATS = ("JPEG", "PNG", "WEBP")  # decoders reachable from uploads (camera MPO JPEGs open as JPEG)
CPU_SLOTS = 2                          # concurrent CPU-heavy requests (PBP itself is multi-threaded)
REF_CACHE_SIZE = 4                     # prepared references kept in memory
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

PARAMS = Params()
_cpu = threading.BoundedSemaphore(CPU_SLOTS)

# Keep multipart parts in memory: Starlette otherwise spools parts above 1 MB to a temp
# file on disk (the attribute was renamed across Starlette versions).
for _attr in ("max_file_size", "spool_max_size"):
    if hasattr(MultiPartParser, _attr):
        setattr(MultiPartParser, _attr, MAX_BODY)


class ReferenceCache:
    """Thread-safe LRU of prepared references keyed by (sha256 of the upload, methods)."""

    def __init__(self, size: int):
        self.size = size
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            ref = self._items.get(key)
            if ref is not None:
                self._items.move_to_end(key)
            return ref

    def put(self, key, ref) -> None:
        with self._lock:
            self._items[key] = ref
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


ref_cache = ReferenceCache(REF_CACHE_SIZE)


def _warm_up() -> None:
    """Pay OpenCV's one-time ORB/ECC initialisation before the first real request."""
    rng = np.random.default_rng(0)
    a = cv2.GaussianBlur(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), (0, 0), 2)
    align(a, np.roll(a, 3, axis=1), PARAMS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await run_in_threadpool(_warm_up)
    yield


app = FastAPI(title="AVA", version=__version__, lifespan=lifespan,
              docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")


# --------------------------------------------------------------------------- errors

class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class _BodyTooLarge(Exception):
    pass


def _is_body_too_large(exc: BaseException) -> bool:
    """`_BodyTooLarge`, also when a middleware's task group wrapped it in an exception group."""
    if isinstance(exc, _BodyTooLarge):
        return True
    return isinstance(exc, BaseExceptionGroup) and exc.subgroup(_BodyTooLarge) is not None


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": code, "message": message}, status_code=status,
                        headers={"Cache-Control": "no-store"})


@app.exception_handler(ApiError)
async def _api_error(_req: Request, exc: ApiError) -> JSONResponse:
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
        except BaseException as exc:  # noqa: BLE001 - re-raised unless it is the body limit
            if not _is_body_too_large(exc):
                raise
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

def _to_8bit(im: Image.Image) -> Image.Image:
    """16-bit / 32-bit gray modes -> 8-bit "L" (convert("RGB") would clip them to white)."""
    if im.mode.startswith("I;16"):
        arr = np.asarray(im, np.float32) / 257.0
    elif im.mode in ("I", "F"):
        arr = np.asarray(im, np.float32)
        top = float(arr.max()) if arr.size else 0.0
        if top > 65535.0 or float(arr.min()) < 0.0:   # unknown range: stretch min..max
            lo = float(arr.min())
            arr = (arr - lo) * (255.0 / max(top - lo, 1e-6))
        elif top > 255.0:                               # 16-bit data stored as 32-bit
            arr = arr / 257.0
    else:
        return im
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "L")


def decode_image(data: bytes) -> np.ndarray:
    """Uploaded bytes -> BGR uint8, EXIF orientation applied, very large JPEGs decoded downscaled.

    Only JPEG, PNG and WebP are decoded; 16-bit gray images are scaled to 8 bits.
    """
    if not data:
        raise ApiError(400, "empty_file", "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD:
        raise ApiError(413, "too_large", "Upload is larger than 15 MB.")
    try:
        im = Image.open(io.BytesIO(data), formats=IMAGE_FORMATS)
        if im.width * im.height > MAX_PIXELS:
            raise ApiError(413, "too_many_pixels", "Image resolution is too large.")
        side = max(im.size)
        if side > DECODE_SIDE:  # JPEG: decode at 1/2, 1/4, 1/8 scale while the long side stays >= DECODE_SIDE
            im.draft("RGB", (max(1, im.width * DECODE_SIDE // side), max(1, im.height * DECODE_SIDE // side)))
        im = ImageOps.exif_transpose(im)
        rgb = np.asarray(_to_8bit(im).convert("RGB"))
    except ApiError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, KeyError, Image.DecompressionBombError):
        raise ApiError(400, "bad_image", "Could not read the image (JPEG, PNG or WebP expected).") from None
    if min(rgb.shape[:2]) < 64:
        raise ApiError(400, "image_too_small", "Image is too small (min 64 px).")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


async def _form_files(request: Request, names: tuple[str, ...]) -> dict[str, bytes]:
    """Read the named multipart file fields (each at most MAX_UPLOAD bytes)."""
    ctype = request.headers.get("content-type", "")
    if not ctype.startswith("multipart/form-data"):
        raise ApiError(400, "multipart_required", "Send the images as multipart/form-data.")
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


def _parse_methods(methods: str) -> tuple[str, ...]:
    known = baselines.get_methods()
    names = tuple(dict.fromkeys(m.strip() for m in methods.split(",") if m.strip()))
    if not names or any(m not in known for m in names):
        raise ApiError(400, "bad_methods", "methods must be a comma list of: " + ", ".join(known))
    return names


def _cpu_bound(fn, *args):
    """Run CPU-heavy work in the thread pool, at most CPU_SLOTS at a time."""
    def run():
        with _cpu:
            return fn(*args)
    return run_in_threadpool(run)


def _compare(ref_data: bytes, frame_data: bytes, methods: tuple[str, ...]) -> dict:
    ref_sha = hashlib.sha256(ref_data).hexdigest()
    frame_sha = hashlib.sha256(frame_data).hexdigest()
    frame_bgr = decode_image(frame_data)

    key = (ref_sha, tuple(sorted(methods)))
    ref = ref_cache.get(key)
    if ref is None:
        ref = pipeline.prepare_reference([decode_image(ref_data)], PARAMS, key[1])
        ref_cache.put(key, ref)

    results = pipeline.inspect_compare(ref, frame_bgr, PARAMS, methods)
    return {
        "results": {m: pipeline.result_to_json(r) for m, r in results.items()},
        "overlays": {m: base64.b64encode(r.images.get("overlay", b"")).decode("ascii")
                     for m, r in results.items()},
        "reference_sha256": ref_sha,
        "frame_sha256": frame_sha,
    }


def _scan(image_data: bytes) -> dict:
    image_sha = hashlib.sha256(image_data).hexdigest()
    res = surface.scan_surface(decode_image(image_data), PARAMS)
    return {
        "result": pipeline.result_to_json(res),
        "overlay": base64.b64encode(res.images.get("overlay", b"")).decode("ascii"),
        "image_sha256": image_sha,
    }


# --------------------------------------------------------------------------- routes

@app.get("/api/health")
async def health():
    params = PARAMS.to_dict()
    params["surface_threshold"] = surface.method_threshold("pbp", PARAMS)  # Mode 1 (always present)
    return {"ok": True, "version": __version__, "params": params}


@app.post("/api/compare")
async def compare(request: Request, methods: str = "pbp,pixdiff"):
    names = _parse_methods(methods)
    files = await _form_files(request, ("reference", "frame"))
    return await _cpu_bound(_compare, files["reference"], files["frame"], names)


@app.post("/api/scan")
async def scan(request: Request):
    files = await _form_files(request, ("image",))
    return await _cpu_bound(_scan, files["image"])


# Static UI last, so it never shadows an API route.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
