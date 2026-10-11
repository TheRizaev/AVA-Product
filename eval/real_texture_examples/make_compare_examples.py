"""AVA Mode 2 (Compare with reference) demo triples on REAL CC0 texture photos.

    python make_compare_examples.py [--repo "C:/Users/Asus/Desktop/AVA Pr"] [--only disc,flange]

Each candidate is a triple of 1280x720 JPEGs (q92):
    reference.jpg  the good part
    defect.jpg     the same part WITH one synthetic defect + small camera jitter
    light.jpg      the same part, NO defect, strong monotone lighting change + camera jitter

Honest labelling: real texture photos (CC0, Poly Haven, see assets/assets_meta.json) mapped onto
procedurally built part geometry, shaded under one fixed light; the defects and the lighting
change are synthetic.

Rendering (deterministic, no randomness outside seeded numpy Generators):
    * the scene is built at 2x (canvas 2560x1440 + padding) as a height field + albedo + material maps;
    * shading: Lambert + Blinn-Phong under a fixed light (top-left), detail normals from the texture's
      normal map, cast shadows ray-marched through the height field, soft knee (no clipping / glare);
    * camera: affine pose (shift / rotation) on the 2x canvas, lens blur, INTER_AREA down to 1280x720,
      vignette, independent sensor noise, JPEG q92;
    * lighting change (light.jpg only, before sensor noise): gamma, exposure, warm cast and a soft
      large-scale window-light gradient: monotone in every channel.

Every candidate is evaluated with the REAL pipeline (ava / dalil): pbp + pixdiff, margins
(defect >= 1.35 x thr at the defect, all zones on the defect; clean / light <= 0.75 x thr everywhere)
and robustness (JPEG q80 re-encode; resize to 1600 px wide and back): verdicts must not flip.
Outputs: out/<name>/{reference,defect,light}.jpg, candidates.json, contact_sheet.png.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
OUT = HERE / "out"
DEFAULT_REPO = Path(r"C:\Users\Asus\Desktop\AVA Pr")

H, W = 720, 1280
SS = 2                      # supersampling factor of the canvas
PAD = 64                    # canvas padding (2x px) for the camera warp
CH, CW = H * SS + 2 * PAD, W * SS + 2 * PAD
JPEG_Q = 92

MARGIN = 1.35
CLEAN_MARGIN = 0.75
FAR_PX = 40                 # work px: "elsewhere" is farther than this from the defect

# Fixed key light, direction TO the light in image coords (x right, y down, z to the viewer).
LIGHT = np.array([-0.55, -0.48, 0.68], np.float32)
LIGHT /= np.linalg.norm(LIGHT)

# Camera poses (final px, degrees) relative to the reference.
POSE_REF = (0.0, 0.0, 0.0)
POSE_DEFECT = (1.7, -1.2, 0.30)
POSE_LIGHT = (-2.3, 1.5, -0.40)

# Strong, monotone lighting change (light.jpg).
LIGHT_GAMMA = 1.3
LIGHT_EXPOSURE = 0.65
LIGHT_CAST_BGR = (0.84, 0.97, 1.10)        # warm: blue down, red up
LIGHT_GRAD = (1.08, 0.86)                   # window side (upper left) .. far side (lower right)


# =========================================================================== assets

def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "AVA-demo-examples/1.0"})
    return urllib.request.urlopen(req, timeout=60).read()


def ensure_asset(aid: str) -> dict:
    """Download (once) the 2k diffuse + GL normal JPGs of a Poly Haven asset; return its record."""
    ASSETS.mkdir(exist_ok=True)
    meta_path = ASSETS / "assets_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rec = meta.get(aid)
    if rec and all((ASSETS / f["path"]).exists() for f in rec["files"].values()):
        return rec
    files = json.loads(_get(f"https://api.polyhaven.com/files/{aid}"))
    info = json.loads(_get(f"https://api.polyhaven.com/info/{aid}"))
    rec = {"id": aid, "name": info["name"], "authors": list(info["authors"].keys()),
           "url": f"https://polyhaven.com/a/{aid}", "license": "CC0 1.0", "files": {}}
    for key, out in (("Diffuse", "diff"), ("nor_gl", "nor_gl")):
        for res in ("2k", "1k"):
            if key in files and res in files[key] and "jpg" in files[key][res]:
                url = files[key][res]["jpg"]["url"]
                dst = ASSETS / f"{aid}_{out}_{res}.jpg"
                if not dst.exists():
                    dst.write_bytes(_get(url))
                rec["files"][out] = {"path": dst.name, "url": url}
                break
    meta[aid] = rec
    meta_path.write_text(json.dumps(meta, indent=1))
    return rec


_TEX_CACHE: dict = {}


def load_tex(aid: str):
    """(albedo float32 BGR 0..1, normal xy float32 [-1,1] in image coords) of an asset."""
    if aid not in _TEX_CACHE:
        rec = ensure_asset(aid)
        alb = cv2.imread(str(ASSETS / rec["files"]["diff"]["path"]), cv2.IMREAD_COLOR).astype(np.float32) / 255
        nor = cv2.imread(str(ASSETS / rec["files"]["nor_gl"]["path"]), cv2.IMREAD_COLOR).astype(np.float32) / 255
        nxy = np.dstack([nor[..., 2] * 2 - 1, -(nor[..., 1] * 2 - 1)])   # GL: +y up -> image y down
        _TEX_CACHE[aid] = (alb, nxy.astype(np.float32))
    return _TEX_CACHE[aid]


def sample(tex: np.ndarray, scale: float, angle: float = 0.0, offset=(0.0, 0.0)) -> np.ndarray:
    """Tile `tex` over the canvas: canvas px = scale * texture px, rotated by `angle` degrees."""
    th, tw = tex.shape[:2]
    a = math.radians(angle)
    c, s = math.cos(a) / scale, math.sin(a) / scale
    # canvas (x, y) -> texture (u, v)
    M = np.array([[c, -s, offset[0]], [s, c, offset[1]]], np.float32)
    out = cv2.warpAffine(tex, M, (CW, CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_WRAP)
    return out


def sample_normal(nxy: np.ndarray, scale: float, angle: float = 0.0, offset=(0.0, 0.0)) -> np.ndarray:
    """Sample a normal-map xy and rotate the vectors with the mapping."""
    n = sample(nxy, scale, angle, offset)
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    return np.dstack([c * n[..., 0] - s * n[..., 1], s * n[..., 0] + c * n[..., 1]]).astype(np.float32)


# =========================================================================== geometry helpers

YY, XX = np.mgrid[0:CH, 0:CW].astype(np.float32)


def P(x: float, y: float) -> tuple[float, float]:
    """Final-image px -> canvas px."""
    return x * SS + PAD, y * SS + PAD


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def circle_sdf(cx, cy, r):
    """Signed distance (canvas px), positive inside. cx, cy, r in canvas px."""
    return r - np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)


def mask_sdf(mask: np.ndarray) -> np.ndarray:
    """Signed distance of a bool mask, positive inside."""
    m = mask.astype(np.uint8)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    return (din - dout).astype(np.float32)


def capsule_sdf(x0, y0, x1, y1, r):
    """Signed distance to a capsule (segment + radius), positive inside (canvas px)."""
    px, py = XX - x0, YY - y0
    dx, dy = x1 - x0, y1 - y0
    t = np.clip((px * dx + py * dy) / (dx * dx + dy * dy + 1e-9), 0, 1)
    return r - np.sqrt((px - t * dx) ** 2 + (py - t * dy) ** 2)


def aa(sdf: np.ndarray) -> np.ndarray:
    """Anti-aliased coverage of an SDF (canvas px)."""
    return np.clip(sdf + 0.5, 0.0, 1.0).astype(np.float32)


def bevel(sdf: np.ndarray, width: float) -> np.ndarray:
    """0 outside, rises smoothly to 1 over `width` px inside (rounded edge profile)."""
    t = np.clip(sdf / width, 0.0, 1.0)
    return np.sqrt(1 - (1 - t) ** 2).astype(np.float32)   # quarter-circle edge


def fbm(rng: np.random.Generator, shape, scales=(64, 16, 4), weights=(1.0, 0.5, 0.25)) -> np.ndarray:
    """Smooth zero-mean noise (sum of blurred white noise), roughly unit std."""
    out = np.zeros(shape, np.float32)
    for s, w in zip(scales, weights):
        n = rng.standard_normal((shape[0] // s + 4, shape[1] // s + 4)).astype(np.float32)
        n = cv2.resize(n, (shape[1] + 4 * s, shape[0] + 4 * s), interpolation=cv2.INTER_CUBIC)
        out += w * n[2 * s:2 * s + shape[0], 2 * s:2 * s + shape[1]]
    return out / (out.std() + 1e-6)


def tint(alb: np.ndarray, sat: float = 1.0, gain: float = 1.0, bgr=(1.0, 1.0, 1.0)) -> np.ndarray:
    g = alb.mean(axis=2, keepdims=True)
    out = g + sat * (alb - g)
    return np.clip(out * gain * np.asarray(bgr, np.float32), 0, 1).astype(np.float32)


# =========================================================================== scene + renderer

class Scene:
    """Height field + per-pixel albedo / material, composited layer by layer (painter's order)."""

    def __init__(self, bg_alb: np.ndarray, bg_nxy: np.ndarray, bg_nstr: float = 0.6,
                 kd=0.85, ks=0.04, shin=8.0, amb=0.32):
        self.height = np.zeros((CH, CW), np.float32)
        self.alb = bg_alb.copy()
        self.nxy = bg_nxy * bg_nstr
        self.kd = np.full((CH, CW), kd, np.float32)
        self.ks = np.full((CH, CW), ks, np.float32)
        self.shin = np.full((CH, CW), shin, np.float32)
        self.amb = np.full((CH, CW), amb, np.float32)
        self.spec_mod = np.ones((CH, CW), np.float32)

    def layer(self, alpha, height, alb, nxy=None, nstr=0.5, kd=0.85, ks=0.2, shin=20.0, amb=0.3,
              spec_mod=None):
        """Put a layer on top: `height` is absolute (canvas px) where alpha > 0 (bbox-local blend)."""
        rows = np.flatnonzero(alpha.any(axis=1))
        cols = np.flatnonzero(alpha.any(axis=0))
        if rows.size == 0:
            return
        sl = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))

        def cut(v):
            return v[sl] if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[:2] == (CH, CW) else v

        a = alpha[sl].astype(np.float32)
        a3 = a[..., None]
        self.height[sl] = self.height[sl] * (1 - a) + cut(height) * a
        self.alb[sl] = self.alb[sl] * (1 - a3) + cut(alb) * a3
        n = 0.0 if nxy is None else cut(nxy) * nstr
        self.nxy[sl] = self.nxy[sl] * (1 - a3) + n * a3
        for name, v in (("kd", kd), ("ks", ks), ("shin", shin), ("amb", amb)):
            cur = getattr(self, name)
            cur[sl] = cur[sl] * (1 - a) + cut(v) * a
        sm = 1.0 if spec_mod is None else cut(spec_mod)
        self.spec_mod[sl] = self.spec_mod[sl] * (1 - a) + sm * a

    def copy(self) -> "Scene":
        s = Scene.__new__(Scene)
        for k, v in self.__dict__.items():
            s.__dict__[k] = v.copy()
        return s


def cast_shadow(height: np.ndarray, max_h: float, soft: float = 2.0) -> np.ndarray:
    """0..1 shadow from the height field along the key light (ray march toward the light)."""
    l2 = LIGHT[:2] / np.linalg.norm(LIGHT[:2])
    tan_e = LIGHT[2] / np.linalg.norm(LIGHT[:2])
    sh = np.zeros_like(height)
    kmax = int(max_h / tan_e) + 4
    for k in range(2, kmax, 2):
        M = np.float32([[1, 0, k * l2[0]], [0, 1, k * l2[1]]])   # shifted[p] = height[p + k*l]
        occ = cv2.warpAffine(height, M, (CW, CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
        np.maximum(sh, np.clip((occ - height - k * tan_e) / soft, 0, 1), out=sh)
    return cv2.GaussianBlur(sh, (0, 0), 2.5)


def shade(sc: Scene, gain: float, rng_seed: int = 0) -> np.ndarray:
    """Linear-ish radiance (BGR float, ~0..1) of the scene under the key light."""
    hgt = cv2.GaussianBlur(sc.height, (0, 0), 1.2)
    gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    nx = -gx + sc.nxy[..., 0]
    ny = -gy + sc.nxy[..., 1]
    nz = np.ones_like(nx)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx * inv, ny * inv, nz * inv
    ndl = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    hv = LIGHT + np.array([0, 0, 1], np.float32)
    hv /= np.linalg.norm(hv)
    ndh = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0, 1)
    spec = ndh ** sc.shin * sc.ks * sc.spec_mod
    sh = cast_shadow(sc.height, float(sc.height.max()) + 2)
    lit = 1 - 0.85 * sh
    diff = sc.amb + sc.kd * ndl * lit
    out = sc.alb * diff[..., None] * gain + (spec * lit * gain)[..., None] * np.float32([0.97, 0.99, 1.0])
    return out.astype(np.float32)


def soft_knee(v: np.ndarray, knee: float = 0.78, top: float = 0.93) -> np.ndarray:
    span = top - knee
    return np.where(v < knee, v, knee + span * np.tanh((v - knee) / span)).astype(np.float32)


def camera(radiance: np.ndarray, pose, noise_seed: int, light_change: bool = False) -> np.ndarray:
    """Canvas radiance -> 1280x720 BGR uint8 photo (pose = (dx, dy, deg) in final px)."""
    dx, dy, deg = pose
    cx, cy = CW / 2, CH / 2
    M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    M[0, 2] += dx * SS - PAD
    M[1, 2] += dy * SS - PAD
    img = cv2.warpAffine(radiance, M, (W * SS, H * SS), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    img = cv2.GaussianBlur(img, (0, 0), 0.75)                       # lens
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    img = soft_knee(img)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    img = img * (1 - 0.07 * r2)[..., None]                           # vignette (same lens every shot)
    if light_change:
        img = lighting_change(img)
    rng = np.random.default_rng(noise_seed)
    sigma = 0.004 + 0.006 * np.sqrt(np.clip(img, 0, 1))              # shot + read noise
    img = img + rng.standard_normal(img.shape).astype(np.float32) * sigma
    return np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)


def lighting_change(img: np.ndarray) -> np.ndarray:
    """Gamma, exposure, warm cast and a soft window-light gradient (monotone per channel)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = ((xx / W) * 0.65 + (yy / H) * 0.35)                          # 0 at the window, ~1 far side
    t = smoothstep(0.0, 1.0, t)
    hi, lo = LIGHT_GRAD
    grad = hi + (lo - hi) * t
    x = np.clip(img, 0, 1) ** np.float32(LIGHT_GAMMA) * np.float32(LIGHT_EXPOSURE)
    return (x * grad[..., None] * np.asarray(LIGHT_CAST_BGR, np.float32)).astype(np.float32)


def jpeg_bytes(img: np.ndarray, q: int = JPEG_Q) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    return buf.tobytes()


def canvas_mask_to_final(m: np.ndarray) -> np.ndarray:
    """Defect mask on the canvas -> bool mask in REFERENCE image coordinates (1280x720)."""
    crop = m[PAD:PAD + H * SS, PAD:PAD + W * SS].astype(np.float32)
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA) > 0.05


# =========================================================================== defects

def crack_path(rng, x0, y0, ang, length, step=3.0, wander=0.22):
    pts = [(x0, y0)]
    a = ang
    for _ in range(int(length / step)):
        a += rng.normal(0, wander) * 0.5 + (ang - a) * 0.15
        x, y = pts[-1]
        pts.append((x + step * math.cos(a), y + step * math.sin(a)))
    return np.array(pts, np.float32)


def draw_tapered(pts: np.ndarray, w0: float, w1: float) -> np.ndarray:
    """Anti-aliased coverage of a polyline whose width tapers from w0 to w1 (canvas px)."""
    m = np.zeros((CH, CW), np.float32)
    n = len(pts) - 1
    for i in range(n):
        w = w0 + (w1 - w0) * i / max(1, n - 1)
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        lo_x, hi_x = int(min(x0, x1) - w - 3), int(max(x0, x1) + w + 4)
        lo_y, hi_y = int(min(y0, y1) - w - 3), int(max(y0, y1) + w + 4)
        sl = (slice(max(lo_y, 0), hi_y), slice(max(lo_x, 0), hi_x))
        px, py = XX[sl] - x0, YY[sl] - y0
        dx, dy = x1 - x0, y1 - y0
        t = np.clip((px * dx + py * dy) / (dx * dx + dy * dy + 1e-9), 0, 1)
        d = np.sqrt((px - t * dx) ** 2 + (py - t * dy) ** 2)
        np.maximum(m[sl], np.clip(w / 2 - d + 0.5, 0, 1), out=m[sl])
    return m


def add_crack(sc: Scene, rng, x0, y0, ang, length, w0=4.0, n_branches=5):
    """Fatigue crack from a drilled hole: a dark main crack with short side cracks (heat-check
    network around it) and a thin rust bleed; returns the canvas mask."""
    main = crack_path(rng, x0, y0, ang, length, wander=0.30)
    m = draw_tapered(main, w0, 1.4)
    for i in range(n_branches):
        k = int(len(main) * (0.12 + 0.75 * i / max(1, n_branches - 1)))
        side = (1 if i % 2 == 0 else -1) * rng.uniform(0.35, 0.8)
        br = crack_path(rng, *main[k], ang + side, length * rng.uniform(0.12, 0.24), wander=0.35)
        m = np.maximum(m, draw_tapered(br, w0 * 0.5, 1.0))
    halo = cv2.GaussianBlur(m, (0, 0), 5.0)
    halo = np.clip(halo * 2.0, 0, 1)
    rust = np.float32([0.10, 0.22, 0.42])
    sc.alb = sc.alb * (1 - 0.45 * halo[..., None]) + rust * (0.45 * halo[..., None])
    sc.alb = sc.alb * (1 - 0.88 * m[..., None])
    sc.height = sc.height - 3.0 * m
    sc.ks = sc.ks * (1 - m)
    return np.maximum(m, (halo > 0.15).astype(np.float32))


def add_scratch(sc: Scene, rng, pts: np.ndarray, w=5.0):
    """Paint scratch down to bare steel: bright core, dark raised paint lips, micro-scratches."""
    core = draw_tapered(pts, w, w * 0.45)
    lips = np.clip(draw_tapered(pts, w + 5, w * 0.45 + 4) - core, 0, 1)
    steel = np.float32([0.70, 0.71, 0.72]) * (1 + 0.08 * fbm(rng, (CH, CW), (8, 2), (1, .5)))[..., None]
    sc.alb = sc.alb * (1 - core[..., None]) + steel * core[..., None]
    sc.alb = sc.alb * (1 - 0.35 * lips[..., None])
    sc.height = sc.height - 1.5 * core + 0.6 * lips
    sc.ks = sc.ks * (1 - core) + 0.45 * core
    sc.shin = sc.shin * (1 - core) + 30 * core
    m = np.maximum(core, lips)
    for j in range(3):   # a few faint parallel micro-scratches (same impact)
        off = rng.normal(0, 1, 2) * 6 + 10 * (j + 1) * np.float32([0.3, 1.0])
        sub = pts[int(len(pts) * 0.2):int(len(pts) * (0.55 + 0.1 * j))] + off.astype(np.float32)
        mm = draw_tapered(sub, 2.0, 0.8)
        sc.alb = sc.alb * (1 - 0.6 * mm[..., None]) + steel * (0.6 * mm[..., None])
        m = np.maximum(m, mm)
    return m


def add_tape(sc: Scene, rng, cx, cy, length, width, ang_deg):
    """Strip of crepe masking tape with torn ends, draped over the surface below."""
    a = math.radians(ang_deg)
    ux, uy = math.cos(a), math.sin(a)
    u = (XX - cx) * ux + (YY - cy) * uy           # along the strip
    v = -(XX - cx) * uy + (YY - cy) * ux          # across
    # torn ends: jagged profile across the width
    vv = np.linspace(-width, width, 64)
    tear0 = np.cumsum(rng.normal(0, 1.6, 64)); tear0 -= tear0.mean()
    tear1 = np.cumsum(rng.normal(0, 1.6, 64)); tear1 -= tear1.mean()
    e0 = np.interp(v, vv, -length / 2 + tear0).astype(np.float32)
    e1 = np.interp(v, vv, length / 2 + tear1).astype(np.float32)
    sdf = np.minimum.reduce([width / 2 - np.abs(v), u - e0, e1 - u])
    alpha = aa(sdf)
    crepe = np.sin(u * 2 * math.pi / 5.5 + 0.8 * fbm(rng, (CH, CW), (16,), (1,))) * 0.5
    fib = fbm(rng, (CH, CW), (6, 2), (1, 0.6))
    base = np.float32([0.64, 0.79, 0.86])
    alb = base * (1 + 0.035 * crepe + 0.03 * fib)[..., None]
    under = cv2.GaussianBlur(sc.alb.mean(axis=2), (0, 0), 3)
    alb = alb * (0.88 + 0.12 * under / (under.mean() + 1e-6))[..., None]    # slight translucency
    hgt = cv2.GaussianBlur(sc.height, (0, 0), 7) + 2.5
    nxy = np.dstack([crepe * 0.05 * uy * 0 + 0.06 * np.cos(u * 2 * math.pi / 5.5) * ux,
                     0.06 * np.cos(u * 2 * math.pi / 5.5) * uy]).astype(np.float32)
    sc.layer(alpha, hgt, alb, nxy, nstr=1.0, kd=0.9, ks=0.06, shin=6, amb=0.3)
    return alpha


# =========================================================================== parts

# Texture scales: any texture that covers more than 2048 canvas px is sampled at >= TEX_NOREP canvas px
# per texture px, so its tiling never repeats inside the frame (a repeat would be a second, false
# "motion" for the ORB registration).
TEX_NOREP = 1.40


def scene_disc(rng, defect: bool):
    """Drilled + slotted brake disc (cast iron, flash rust) on a garage floor; defect: crack."""
    alb, nxy = load_tex("garage_floor")
    bg = tint(sample(alb, TEX_NOREP, 12), sat=0.85, gain=1.05)
    sc = Scene(bg, sample_normal(nxy, TEX_NOREP, 12), bg_nstr=0.7, kd=0.85, ks=0.03, shin=6, amb=0.32)
    cx, cy = P(640, 360)
    R = 330 * SS
    r = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    talb, tn = load_tex("rusty_metal_sheet")
    iron = tint(sample(talb, 1.1, 37, (300, 500)), sat=0.35, gain=1.10, bgr=(1.0, 1.0, 1.02))
    low = cv2.GaussianBlur(iron, (0, 0), 25)
    iron = low + 0.12 * (iron - low)      # friction face: pad wear has polished most rust away
    tnx = sample_normal(tn, 1.1, 37, (300, 500))
    # lathe marks on the friction ring (concentric), modulating albedo and specular sheen
    rings = np.interp(r, np.arange(0, 2000, 1.0),
                      np.convolve(np.random.default_rng(5).standard_normal(2000), np.ones(3) / 3, "same")
                      ).astype(np.float32)
    hr = 0.028 * R
    holes_sdf = np.full((CH, CW), 1e6, np.float32)
    hole_centres = {}
    for k in range(18):
        if k % 3 == 1:          # these sectors carry a slot instead of drillings
            continue
        for j, rr in enumerate((0.64, 0.77, 0.90)):
            a = (k / 18) * 2 * math.pi + j * 0.065 + 0.12
            hx, hy = cx + rr * R * math.cos(a), cy + rr * R * math.sin(a)
            hole_centres[(k, j)] = (hx, hy, a)
            holes_sdf = np.minimum(holes_sdf, np.sqrt((XX - hx) ** 2 + (YY - hy) ** 2) - hr)
    hub_holes = np.full((CH, CW), 1e6, np.float32)
    for k in range(5):
        a = k / 5 * 2 * math.pi - math.pi / 2
        hx, hy = cx + 0.31 * R * math.cos(a), cy + 0.31 * R * math.sin(a)
        hub_holes = np.minimum(hub_holes, np.sqrt((XX - hx) ** 2 + (YY - hy) ** 2) - 0.052 * R)
    bore = r - 0.19 * R
    body_sdf = np.minimum.reduce([R - r, holes_sdf, hub_holes, bore])
    ring_alb = iron * (1 + 0.010 * rings)[..., None]
    ring_alb = ring_alb * (1 + 0.10 * smoothstep(0.55 * R, 0.60 * R, r) * smoothstep(R, 0.96 * R, r))[..., None]
    hgt = 28 + 6 * bevel(body_sdf, 7)
    # curved slots (milled grooves, not through), one per empty sector
    slot_sdf = np.full((CH, CW), 1e6, np.float32)
    for k in range(1, 18, 3):
        a0 = (k / 18) * 2 * math.pi + 0.05
        pts = []
        for t in np.linspace(0, 1, 14):
            rr = (0.61 + 0.30 * t) * R
            a = a0 + 0.22 * t
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            slot_sdf = np.minimum(slot_sdf, -capsule_sdf(x0, y0, x1, y1, 0.013 * R))
    groove = bevel(-slot_sdf, 3)
    hgt = hgt - 4 * groove
    ring_alb = ring_alb * (1 - 0.35 * groove)[..., None]
    spec_mod = (1 + 0.15 * rings).clip(0.3, 2)
    sc.layer(aa(body_sdf), hgt, ring_alb, tnx, nstr=0.12, kd=0.80, ks=0.30, shin=18, amb=0.30,
             spec_mod=spec_mod)
    # hat (hub): raised, darker, more rust
    hat_body = np.minimum.reduce([0.50 * R - r, hub_holes, bore])
    hat_alb = tint(sample(talb, 0.9, 101, (900, 100)), sat=0.75, gain=0.78)
    hat_n = sample_normal(tn, 0.9, 101, (900, 100))
    sc.layer(aa(hat_body), 52 + 8 * bevel(hat_body, 10), hat_alb, hat_n, nstr=0.5,
             kd=0.82, ks=0.15, shin=12, amb=0.30)
    mask = None
    if defect:   # fatigue crack from a middle-row drilled hole toward the outer edge
        hx, hy, a = hole_centres[(15, 1)]
        x0, y0 = hx + (hr + 1) * math.cos(a), hy + (hr + 1) * math.sin(a)
        mask = add_crack(sc, rng, x0, y0, a + 0.10, 0.215 * R, w0=7.0, n_branches=8)
    return sc, mask, 1.55


def hex_sdf(cx, cy, s, rot):
    """Hexagon (across flats = 2s) SDF, positive inside."""
    d = np.full((CH, CW), -1e6, np.float32)
    for k in range(6):
        a = rot + k * math.pi / 3
        d = np.maximum(d, (XX - cx) * math.cos(a) + (YY - cy) * math.sin(a) - s)
    return -d


def scene_flange(rng, defect: bool):
    """Green painted steel flange with a pipe collar and 8 zinc hex bolts on an OSB bench;
    defect: one bolt missing."""
    alb, nxy = load_tex("oriented_strand_board")
    bg = tint(sample(alb, TEX_NOREP, 0), sat=0.8, gain=0.92)
    sc = Scene(bg, sample_normal(nxy, TEX_NOREP, 0), bg_nstr=0.8, kd=0.85, ks=0.04, shin=6, amb=0.32)
    cx, cy = P(640, 360)
    half = 300 * SS
    rot = math.radians(4)
    u = (XX - cx) * math.cos(rot) + (YY - cy) * math.sin(rot)
    v = -(XX - cx) * math.sin(rot) + (YY - cy) * math.cos(rot)
    rc = 70 * SS
    qx, qy = np.abs(u) - (half - rc), np.abs(v) - (half - rc)
    plate_sdf = rc - (np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0))
    r = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    talb, tn = load_tex("green_metal_rust")
    paint = tint(sample(talb, 1.0, 90, (400, 200)), sat=0.95, gain=2.05)
    pn = sample_normal(tn, 1.0, 90, (400, 200))
    plate = np.minimum(plate_sdf, r - 140 * SS)
    sc.layer(aa(plate), 20 + 5 * bevel(plate, 8), paint, pn, nstr=0.6, kd=0.82, ks=0.22, shin=24, amb=0.30)
    coll = np.minimum(r - 112 * SS, 158 * SS - r)
    coll_alb = tint(sample(talb, 0.8, 10, (100, 900)), sat=0.95, gain=2.15)
    sc.layer(aa(coll), 44 + 10 * bevel(coll, 12), coll_alb, sample_normal(tn, 0.8, 10, (100, 900)),
             nstr=0.5, kd=0.8, ks=0.3, shin=26, amb=0.3)
    bore = 112 * SS - r
    sc.layer(aa(bore), np.zeros_like(r) - 10, np.float32([0.07, 0.07, 0.075]) * np.ones((CH, CW, 3), np.float32),
             None, kd=0.5, ks=0.0, amb=0.25)
    zalb, zn = load_tex("concrete_floor_worn_001")
    zinc = tint(sample(zalb, 0.6, 33), sat=0.15, gain=1.85, bgr=(1.03, 1.0, 0.97))
    zinc_n = sample_normal(zn, 0.6, 33)
    missing = 2
    mask = None
    for k in range(8):
        a = k / 8 * 2 * math.pi + rot + math.pi / 8
        bx, by = cx + 222 * SS * math.cos(a), cy + 222 * SS * math.sin(a)
        rr = np.sqrt((XX - bx) ** 2 + (YY - by) ** 2)
        if defect and k == missing:
            # empty threaded hole + cleaner washer imprint on the paint
            imprint = aa(37 * SS - rr) * aa(rr - 20 * SS)
            ring = np.clip(1 - np.abs(rr - 37 * SS) / (2.0 * SS), 0, 1)
            low = cv2.GaussianBlur(sc.alb, (0, 0), 12)     # paint protected under the washer: clean
            sc.alb = sc.alb * (1 - 0.85 * imprint[..., None]) + tint(low, 0.9, 1.10) * (0.85 * imprint[..., None])
            sc.alb = sc.alb * (1 - 0.45 * ring[..., None])
            thread = 0.5 + 0.5 * np.cos(rr / (1.6 * SS) * 2 * math.pi)
            # thread crests catch the light, roots stay dark
            hole_alb = (np.float32([0.17, 0.17, 0.18]) * (0.35 + 1.3 * thread ** 2)[..., None])
            hole = 20 * SS - rr
            sc.layer(aa(hole), 20 - 26 * bevel(hole, 6), hole_alb, None, kd=0.6, ks=0.1, shin=10, amb=0.25)
            mask = aa(40 * SS - rr)
            continue
        washer = 37 * SS - rr
        sc.layer(aa(washer), 24 + 3 * bevel(washer, 4), zinc, zinc_n, nstr=0.3, kd=0.8, ks=0.35, shin=26, amb=0.3)
        head = hex_sdf(bx, by, 26 * SS, a * 0.37 + k * 0.4)
        top = np.minimum(head, 33 * SS - rr)          # chamfered hex: corners turned off
        sc.layer(aa(top), 24 + 3 + 22 * bevel(top, 7), zinc * 0.95, zinc_n, nstr=0.3,
                 kd=0.8, ks=0.4, shin=30, amb=0.3)
    return sc, mask, 1.20


def scene_panel(rng, defect: bool):
    """Riveted lap joint of two painted steel panels, mat at the left edge; defect: paint scratch."""
    alb, nxy = load_tex("rubber_tiles")
    bg = tint(sample(alb, TEX_NOREP, 0), gain=2.4)
    sc = Scene(bg, sample_normal(nxy, TEX_NOREP, 0), bg_nstr=0.5, kd=0.85, ks=0.08, shin=10, amb=0.32)
    talb, tn = load_tex("blue_metal_plate")
    paint = tint(sample(talb, TEX_NOREP, 90, (200, 300)), sat=0.9, gain=2.6)
    pn = sample_normal(tn, TEX_NOREP, 90, (200, 300))
    x_edge = P(150, 0)[0] + (YY - PAD) * 0.06
    lower = XX - x_edge
    sc.layer(aa(lower), 10 + 4 * bevel(lower, 5), paint, pn, nstr=0.55, kd=0.82, ks=0.28, shin=30, amb=0.3)
    y_edge = P(0, 300)[1] + (XX - PAD) * 0.025
    paint2 = tint(sample(talb, TEX_NOREP, 180, (1200, 700)), sat=0.9, gain=2.65)
    upper = np.minimum(y_edge - YY, lower)
    sc.layer(aa(upper), 20 + 4 * bevel(upper, 5), paint2, sample_normal(tn, TEX_NOREP, 180, (1200, 700)),
             nstr=0.55, kd=0.82, ks=0.28, shin=30, amb=0.3)
    xs = P(1010, 0)[0]
    strip = np.minimum.reduce([XX - xs, xs + 120 * SS - XX, lower])
    sc.layer(aa(strip), sc.height + 8 * bevel(strip, 10), tint(paint, 1, 1.05), pn, nstr=0.5,
             kd=0.82, ks=0.3, shin=30, amb=0.3)
    rivets = []
    for i in range(12):
        x = 230 + i * 66
        if x > 990:
            break
        y = 300 - 26 + x * 0.025
        rivets.append((x, y - 6))
        rivets.append((x + 33, y + 34))
    for j in range(6):
        rivets.append((1070, 80 + j * 110))
    for (x, y) in rivets:
        bx, by = P(x, y)
        rr = np.sqrt((XX - bx) ** 2 + (YY - by) ** 2)
        rad = 13 * SS
        dome = np.sqrt(np.clip(1 - (rr / rad) ** 2, 0, 1)) * 9
        base = float(sc.height[int(by), int(bx)])
        sc.layer(aa(rad - rr), base + dome, tint(paint, 0.9, 1.08), None, kd=0.82, ks=0.4, shin=40, amb=0.3)
    mask = None
    if defect:
        t = np.linspace(0, 1, 140)
        xs_ = 330 + 430 * t
        ys_ = 470 + 120 * t - 40 * np.sin(t * math.pi) + 1.5 * np.sin(t * 23)
        pts = np.array([P(x, y) for x, y in zip(xs_, ys_)], np.float32)
        mask = add_scratch(sc, rng, pts, w=6.0)
    return sc, mask, 1.20


def gear_radius(th, n, r_root, r_tip, broken=None, rng=None):
    """Tooth profile r(theta) of a spur gear; `broken` = index of a fractured tooth."""
    ph = (th / (2 * math.pi) * n) % 1.0
    d = np.abs(ph - 0.5)
    prof = smoothstep(0.30, 0.18, d)
    r = r_root + (r_tip - r_root) * prof
    if broken is not None:
        idx = np.floor(th / (2 * math.pi) * n) % n
        sel = idx == broken
        jag = np.interp(ph, np.linspace(0, 1, 40), rng.normal(0, 1, 40).cumsum() * 0.6).astype(np.float32)
        r_b = r_root + (r_tip - r_root) * np.clip(0.05 + 0.05 * jag, 0, 0.2) * prof
        r = np.where(sel, np.minimum(r, r_b), r)
    return r


def scene_gear(rng, defect: bool):
    """Spur gear (steel, light surface rust) with lightening holes and keyed hub on a concrete floor;
    defect: one tooth broken off."""
    alb, nxy = load_tex("rubber_tiles")
    bg = tint(sample(alb, TEX_NOREP, 0, (700, 300)), sat=0.15, gain=2.8)
    low = cv2.GaussianBlur(bg, (0, 0), 6)
    bg = low + 0.5 * (bg - low)
    sc = Scene(bg, sample_normal(nxy, TEX_NOREP, 0, (700, 300)), bg_nstr=0.5, kd=0.85, ks=0.08, shin=10, amb=0.32)
    cx, cy = P(660, 360)
    r = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    th = np.arctan2(YY - cy, XX - cx) + math.pi
    n = 28
    r_tip, r_root = 318 * SS, 288 * SS
    prof = gear_radius(th, n, r_root, r_tip, broken=10 if defect else None, rng=rng)
    gear = prof - r
    talb, tn = load_tex("rusty_metal_02")
    steel = tint(sample(talb, 1.2, 15, (300, 300)), sat=0.32, gain=0.95)
    gn = sample_normal(tn, 1.2, 15, (300, 300))
    holes = np.full((CH, CW), 1e6, np.float32)
    for k in range(6):
        a = k / 6 * 2 * math.pi + 0.2
        hx, hy = cx + 185 * SS * math.cos(a), cy + 185 * SS * math.sin(a)
        holes = np.minimum(holes, np.sqrt((XX - hx) ** 2 + (YY - hy) ** 2) - 52 * SS)
    bore = r - 62 * SS
    key = np.maximum(np.abs(XX - cx) - 14 * SS, np.abs(YY - (cy - 64 * SS)) - 14 * SS)
    bore = np.minimum(bore, key)
    body = np.minimum.reduce([gear, holes, bore])
    sc.layer(aa(body), 26 + 6 * bevel(body, 7), steel, gn, nstr=0.4, kd=0.8, ks=0.30, shin=20, amb=0.3)
    hub = np.minimum(118 * SS - r, bore)
    sc.layer(aa(hub), 40 + 8 * bevel(hub, 9), tint(steel, 0.8, 1.05), gn, nstr=0.4,
             kd=0.8, ks=0.35, shin=24, amb=0.3)
    mask = None
    if defect:
        prof0 = gear_radius(th, n, r_root, r_tip)
        lost = np.clip(np.clip((prof0 - r) + 0.5, 0, 1) - np.clip((prof - r) + 0.5, 0, 1), 0, 1)
        rim = aa(prof - r) * aa(6 * SS - (prof - r)) * (np.abs(prof - prof0) > 1).astype(np.float32)
        rim = cv2.GaussianBlur(rim, (0, 0), 1.0)
        fresh = np.float32([0.70, 0.71, 0.72]) * (1 + 0.15 * fbm(rng, (CH, CW), (4, 2), (1, .5)))[..., None]
        sc.alb = sc.alb * (1 - rim[..., None]) + fresh * rim[..., None]
        mask = np.maximum(lost, rim)
    return sc, mask, 1.30


def scene_bracket(rng, defect: bool):
    """Stamped zinc-plated steel bracket with beads, holes and a flanged cut-out on a worn wooden
    bench; defect: strip of masking tape."""
    alb, nxy = load_tex("wood_table_worn")
    bg = tint(sample(alb, TEX_NOREP, 90), sat=0.85, gain=2.1)
    sc = Scene(bg, sample_normal(nxy, TEX_NOREP, 90), bg_nstr=0.6, kd=0.85, ks=0.06, shin=10, amb=0.32)
    pts = np.array([(250, 120), (900, 85), (1060, 200), (1080, 470), (930, 640), (560, 610),
                    (390, 655), (205, 560), (180, 300)], np.float32)
    poly = np.array([P(x, y) for x, y in pts], np.float32)
    m = np.zeros((CH, CW), np.uint8)
    cv2.fillPoly(m, [np.round(poly * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    sdf = mask_sdf(m > 0)
    sdf = cv2.GaussianBlur(sdf, (0, 0), 22)
    zalb, zn = load_tex("concrete_floor_worn_001")
    zinc = tint(sample(zalb, 1.0, 64, (500, 100)), sat=0.25, gain=1.85, bgr=(1.03, 1.0, 0.96))
    zn_ = sample_normal(zn, 1.0, 64, (500, 100))
    holes = np.full((CH, CW), 1e6, np.float32)
    for x, y in ((290, 190), (860, 165), (980, 450), (300, 520)):
        hx, hy = P(x, y)
        holes = np.minimum(holes, np.sqrt((XX - hx) ** 2 + (YY - hy) ** 2) - 22 * SS)
    slot = -capsule_sdf(*P(560, 520), *P(700, 515), 16 * SS)
    cut = -capsule_sdf(*P(560, 330), *P(760, 360), 62 * SS)
    body = np.minimum.reduce([sdf, holes, slot, cut])
    hgt = 18 + 4 * bevel(body, 6)
    lip = np.clip(1 - np.abs(-cut - 12 * SS) / (12 * SS), 0, 1)
    hgt = hgt + 9 * np.sin(lip * math.pi / 2) ** 2
    for (x0, y0, x1, y1) in ((330, 260, 470, 450), (820, 250, 930, 420)):
        cap = capsule_sdf(*P(x0, y0), *P(x1, y1), 18 * SS)
        hgt = hgt + 8 * np.sqrt(np.clip(cap / (18 * SS), 0, 1))
    sc.layer(aa(body), hgt, zinc, zn_, nstr=0.45, kd=0.80, ks=0.30, shin=22, amb=0.3)
    mask = None
    if defect:
        cx, cy = P(470, 220)
        mask = add_tape(sc, rng, cx, cy, 230 * SS, 46 * SS, -24)
    return sc, mask, 1.30


CANDIDATES = {
    "disc_crack": dict(fn=scene_disc, seed=11, assets=["rusty_metal_sheet", "garage_floor"],
                       desc="Drilled and slotted cast-iron brake disc with flash rust on a garage floor; "
                            "defect: hairline fatigue crack running from a drilled hole toward the rim."),
    "flange_missing_bolt": dict(fn=scene_flange, seed=12,
                                assets=["green_metal_rust", "concrete_floor_worn_001", "oriented_strand_board"],
                                desc="Green painted steel pipe flange with 8 zinc hex bolts on an OSB workbench; "
                                     "defect: one bolt missing (empty threaded hole, washer imprint)."),
    "panel_scratch": dict(fn=scene_panel, seed=13, assets=["blue_metal_plate", "rubber_tiles"],
                          desc="Blue painted steel lap joint with two rows of domed rivets and a stiffener; "
                               "defect: scratch through the paint down to bare steel."),
    "gear_broken_tooth": dict(fn=scene_gear, seed=14, assets=["rusty_metal_02", "rubber_tiles"],
                              desc="28-tooth spur gear (steel with light surface rust, lightening holes, keyed "
                                   "hub) on a rubber mat; defect: one tooth broken off (fresh bright fracture)."),
    "bracket_tape": dict(fn=scene_bracket, seed=15, assets=["concrete_floor_worn_001", "wood_table_worn"],
                         desc="Stamped zinc-plated bracket with beads, holes, slot and flanged cut-out on a "
                              "worn wooden bench; defect: torn strip of masking tape on the bracket."),
}


def render_triple(name: str):
    spec = CANDIDATES[name]
    seed = spec["seed"]
    sc_ref, _, gain = spec["fn"](np.random.default_rng([seed, 0]), False)
    sc_def, mask, _ = spec["fn"](np.random.default_rng([seed, 0]), True)
    rad_ref = shade(sc_ref, gain)
    rad_def = shade(sc_def, gain)
    ref = camera(rad_ref, POSE_REF, noise_seed=seed * 10 + 1)
    bad = camera(rad_def, POSE_DEFECT, noise_seed=seed * 10 + 2)
    light = camera(rad_ref, POSE_LIGHT, noise_seed=seed * 10 + 3, light_change=True)
    return ref, bad, light, canvas_mask_to_final(mask)


# =========================================================================== evaluation

def import_pkg(repo: Path):
    sys.path.insert(0, str(repo))
    try:
        import ava as pkg  # noqa
        from ava import pipeline, config
        from ava.pbp import cell_mask_from_pixel_mask
        name = "ava"
    except ImportError:
        import dalil as pkg  # noqa
        from dalil import pipeline, config
        from dalil.pbp import cell_mask_from_pixel_mask
        name = "dalil"
    return name, pipeline, config, cell_mask_from_pixel_mask


def decode(b: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


def variant(img: np.ndarray, kind: str) -> np.ndarray:
    if kind == "jpeg80":
        return decode(jpeg_bytes(img, 80))
    if kind == "resize1600":
        h, w = img.shape[:2]
        big = cv2.resize(img, (1600, round(h * 1600 / w)), interpolation=cv2.INTER_CUBIC)
        return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    return img


class Evaluator:
    def __init__(self, repo: Path):
        self.pkg, self.pipeline, config, self.cells = import_pkg(repo)
        self.P = config.Params()
        self.methods = ("pbp", "pixdiff")

    def _mask_at(self, mask, work_size, dilate_px):
        m = cv2.resize(mask.astype(np.uint8), tuple(work_size), interpolation=cv2.INTER_NEAREST)
        if dilate_px:
            m = cv2.dilate(m, np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8))
        return m > 0

    def _max(self, res, region=None):
        if res.score is None:
            return float("nan")
        sel = res.checked.copy()
        if region is not None:
            sel &= self.cells(region, self.P.win, self.P.step, frac=1.0 / (self.P.win ** 2))
        return float(res.score[sel].max()) if sel.any() else 0.0

    def run(self, ref, bad, light, mask) -> dict:
        P, pl = self.P, self.pipeline
        reference = pl.prepare_reference([ref], P, self.methods)
        t0 = time.perf_counter()
        rb = pl.inspect_compare(reference, bad, P, self.methods)
        rl = pl.inspect_compare(reference, light, P, self.methods)
        ms = (time.perf_counter() - t0) * 500
        thr = rb["pbp"].threshold
        ws = rb["pbp"].work_size
        mw = self._mask_at(mask, ws, 8)
        far = ~self._mask_at(mask, ws, FAR_PX)
        zones_on = [bool(mw[y:y + h, x:x + w].any()) for x, y, w, h in (z.bbox for z in rb["pbp"].zones)]
        s_at = self._max(rb["pbp"], mw)
        s_off = self._max(rb["pbp"], far)
        s_light = self._max(rl["pbp"])
        px_b, px_l = rb["pixdiff"], rl["pixdiff"]
        row = {
            "threshold": round(thr, 4), "threshold_pixdiff": round(px_b.threshold, 4),
            "defect": {"pbp_verdict": rb["pbp"].verdict, "pbp_reason": rb["pbp"].reason,
                       "zones": len(rb["pbp"].zones), "zones_on_defect": sum(zones_on),
                       "max_at_defect": round(s_at, 4), "max_elsewhere": round(s_off, 4),
                       "ratio_at_defect": round(s_at / thr, 3), "ratio_elsewhere": round(s_off / thr, 3),
                       "pixdiff_verdict": px_b.verdict, "pixdiff_reason": px_b.reason,
                       "align": rb["pbp"].align.get("method")},
            "light": {"pbp_verdict": rl["pbp"].verdict, "pbp_reason": rl["pbp"].reason,
                      "max": round(s_light, 4), "ratio": round(s_light / thr, 3),
                      "pixdiff_verdict": px_l.verdict, "pixdiff_reason": px_l.reason,
                      "pixdiff_changed_frac": round(px_l.changed_frac, 3),
                      "mean_gray": round(float(cv2.cvtColor(light, cv2.COLOR_BGR2GRAY).mean()), 1),
                      "align": rl["pbp"].align.get("method")},
            "ref_mean_gray": round(float(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).mean()), 1),
            "ms_per_check": round(ms, 0),
        }
        row["ok"] = bool(
            rb["pbp"].verdict == "fail" and zones_on and all(zones_on)
            and s_at >= MARGIN * thr and s_off <= CLEAN_MARGIN * thr
            and rl["pbp"].verdict == "pass" and s_light <= CLEAN_MARGIN * thr
            and px_l.verdict != "pass")
        row["_res"] = (rb, rl)
        return row

    def flips(self, ref, bad, light, mask) -> dict:
        """Verdicts under JPEG q80 and 1600-px resize round trips (applied to all three images)."""
        out = {}
        for kind in ("jpeg80", "resize1600"):
            r = self.run(variant(ref, kind), variant(bad, kind), variant(light, kind), mask)
            d, l_ = r["defect"], r["light"]
            stable = (d["pbp_verdict"] == "fail" and d["zones"] > 0 and d["zones_on_defect"] == d["zones"]
                      and l_["pbp_verdict"] == "pass")
            out[kind] = {"defect": d["pbp_verdict"], "defect_ratio": d["ratio_at_defect"],
                         "defect_zones_on_defect": f'{d["zones_on_defect"]}/{d["zones"]}',
                         "defect_elsewhere_ratio": d["ratio_elsewhere"],
                         "light": l_["pbp_verdict"], "light_ratio": l_["ratio"],
                         "light_pixdiff": f'{l_["pixdiff_verdict"]}/{l_["pixdiff_reason"]}',
                         "stable": bool(stable)}
        return out


# =========================================================================== contact sheet

TILE_W, TILE_H, CAP = 400, 225, 44


def _tile(img: np.ndarray, title: str, verdict: str | None = None, color=(40, 40, 40)) -> np.ndarray:
    t = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    cap = np.full((CAP, TILE_W, 3), 250, np.uint8)
    cv2.putText(cap, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (30, 30, 30), 1, cv2.LINE_AA)
    if verdict:
        cv2.putText(cap, verdict, (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return np.vstack([t, cap])


def _vcolor(v: str):
    return {"pass": (40, 140, 40), "fail": (30, 30, 200)}.get(v, (90, 90, 90))


def contact_sheet(rows: list, path: Path):
    blocks = []
    header = np.full((60, TILE_W * 4, 3), 255, np.uint8)
    cv2.putText(header, "AVA Mode 2 demo candidates - real texture photos (CC0, Poly Haven) + synthetic "
                "defect / synthetic lighting change", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(header, "row 1: reference | defect | AVA (pbp) | plain subtraction (pixdiff)     "
                "row 2: name | lighting change, no defect | AVA | plain subtraction",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    blocks.append(header)
    for name, imgs, row in rows:
        ref, bad, light = imgs
        rb, rl = row["_res"]
        d, l_ = row["defect"], row["light"]

        def vtxt(res, ratio=None):
            v = res.verdict.upper()
            if res.verdict == "fail":
                v += f" - {len(res.zones)} zone(s)"
            elif res.verdict == "retake":
                v += f" - {res.reason}"
            if ratio is not None:
                v += f"  max {ratio:.2f}x thr"
            return v

        ov = lambda res: decode(res.images["overlay"])  # noqa: E731
        r1 = np.hstack([
            _tile(ref, "Reference"),
            _tile(bad, "Check: defect"),
            _tile(ov(rb["pbp"]), "AVA (pbp)", vtxt(rb["pbp"], d["ratio_at_defect"]), _vcolor(rb["pbp"].verdict)),
            _tile(ov(rb["pixdiff"]), "Plain subtraction", vtxt(rb["pixdiff"]), _vcolor(rb["pixdiff"].verdict)),
        ])
        label = np.full((TILE_H + CAP, TILE_W, 3), 245, np.uint8)
        cv2.putText(label, name, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        ok = "ACCEPTED (margins + robust)" if row["ok"] and row["robust"] else "REJECTED"
        cv2.putText(label, ok, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (40, 140, 40) if "ACC" in ok else (30, 30, 200), 2, cv2.LINE_AA)
        lines = [f"defect: {d['ratio_at_defect']:.2f}x thr at defect,",
                 f"  {d['ratio_elsewhere']:.2f}x elsewhere, zones {d['zones_on_defect']}/{d['zones']} on it",
                 f"light: AVA max {l_['ratio']:.2f}x thr",
                 f"  pixdiff {l_['pixdiff_verdict']}/{l_['pixdiff_reason']}, "
                 f"{100 * l_['pixdiff_changed_frac']:.0f}% changed",
                 "q80: " + ("ok" if row["flips"]["jpeg80"]["stable"] else "FLIP")
                 + "   resize1600: " + ("ok" if row["flips"]["resize1600"]["stable"] else "FLIP")]
        for i, s in enumerate(lines):
            cv2.putText(label, s, (10, 104 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (40, 40, 40), 1, cv2.LINE_AA)
        r2 = np.hstack([
            label,
            _tile(light, "Check: lighting change, no defect"),
            _tile(ov(rl["pbp"]), "AVA (pbp)", vtxt(rl["pbp"], l_["ratio"]), _vcolor(rl["pbp"].verdict)),
            _tile(ov(rl["pixdiff"]), "Plain subtraction", vtxt(rl["pixdiff"]),
                  (30, 30, 200) if rl["pixdiff"].verdict != "pass" else (40, 140, 40)),
        ])
        blocks += [r1, r2, np.full((14, TILE_W * 4, 3), 200, np.uint8)]
    cv2.imwrite(str(path), np.vstack(blocks))


# =========================================================================== main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    ap.add_argument("--only", default="")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    names = [n for n in CANDIDATES if not args.only or n in args.only.split(",")]
    meta = {}
    for n in names:
        for aid in CANDIDATES[n]["assets"]:
            meta[aid] = ensure_asset(aid)
    ev = None if args.no_eval else Evaluator(Path(args.repo))
    rows, out_json = [], []
    for n in names:
        t0 = time.time()
        ref, bad, light, mask = render_triple(n)
        d = OUT / n
        d.mkdir(parents=True, exist_ok=True)
        files = {}
        imgs = []
        for key, img in (("reference", ref), ("defect", bad), ("light", light)):
            b = jpeg_bytes(img)
            (d / f"{key}.jpg").write_bytes(b)
            files[key] = str(d / f"{key}.jpg")
            imgs.append(decode(b))                      # evaluate exactly what is on disk
        cv2.imwrite(str(d / "defect_mask.png"), mask.astype(np.uint8) * 255)
        print(f"[{n}] rendered in {time.time() - t0:.1f}s", flush=True)
        if ev is None:
            continue
        row = ev.run(*imgs, mask)
        row["flips"] = ev.flips(*imgs, mask)
        row["robust"] = all(v["stable"] for v in row["flips"].values())
        printable = {k: v for k, v in row.items() if k != "_res"}
        print(json.dumps(printable, indent=1), flush=True)
        rows.append((n, imgs, row))
        out_json.append({
            "name": n, "files": files, "defect_mask": str(d / "defect_mask.png"),
            "description": CANDIDATES[n]["desc"],
            "label": "real texture photo (CC0, Poly Haven) + synthetic defect / synthetic lighting change",
            "accepted": bool(row["ok"] and row["robust"]),
            **printable,
            "assets": [{k: meta[a][k] for k in ("id", "name", "authors", "url", "license")} for a in
                       CANDIDATES[n]["assets"]],
            "package": ev.pkg,
        })
    if ev is not None:
        stem = "" if not args.only else "_partial"     # a partial run never overwrites the full results
        (HERE / f"candidates{stem}.json").write_text(json.dumps({
            "generator": str(Path(__file__).resolve()),
            "criteria": {"defect_min_ratio": MARGIN, "clean_max_ratio": CLEAN_MARGIN, "far_px": FAR_PX,
                         "robustness": ["jpeg q80 re-encode", "resize to 1600 px wide and back"]},
            "lighting_change": {"gamma": LIGHT_GAMMA, "exposure": LIGHT_EXPOSURE, "cast_bgr": LIGHT_CAST_BGR,
                                "window_gradient": LIGHT_GRAD},
            "camera_poses": {"reference": POSE_REF, "defect": POSE_DEFECT, "light": POSE_LIGHT},
            "candidates": out_json}, indent=1))
        contact_sheet(rows, HERE / f"contact_sheet{stem}.png")
        print("wrote", HERE / f"candidates{stem}.json", HERE / f"contact_sheet{stem}.png")


if __name__ == "__main__":
    main()
