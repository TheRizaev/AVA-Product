/*
 * AVA — one-screen client with two modes:
 *   Surface check (uniform texture, 1 photo)        → POST /api/scan     (field "image")
 *   Compare with reference (complex parts, 2 photos) → POST /api/compare (fields "reference", "frame")
 * Vanilla JS, no build step. Photos are downscaled on the device (≤ 1600 px, EXIF-free JPEG) and live only
 * in this page; the result history is kept for this session only (nothing is stored on the server).
 * Example buttons only load the example images into the slots; the verdict always comes from the real API.
 */
(function () {
  'use strict';

  const t = window.TEXT.t;

  // ---------------------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------------------

  const SCAN_URL = '/api/scan';
  const COMPARE_URL = '/api/compare?methods=pbp,pixdiff';
  const EXAMPLES_DIR = '/examples/';
  const MAX_SIDE = 1600;               // longest side of an uploaded photo, px
  const JPEG_QUALITY = 0.9;
  const REQUEST_TIMEOUT_MS = 120000;
  const EXAMPLE_TIMEOUT_MS = 20000;  // a stalled example download must not leave the page busy
  const HISTORY_MAX = 10;
  const GHOST_DEFAULT = 40;            // % opacity of the reference over the live camera
  const LS_GHOST = 'ava.ghostOpacity';
  const LS_MODE = 'ava.mode';
  const CAMERA_CONSTRAINTS = {
    audio: false,
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
  };

  /** Per mode: photo slots, endpoint and the built-in examples (slot → file in web/examples/). */
  const MODES = {
    surface: {
      slots: ['image'],
      examples: [
        { id: 'surface-defect', label: 'ex_defect', files: { image: 'surface_defect.jpg' } },
        { id: 'surface-ok', label: 'ex_ok', files: { image: 'surface_ok.jpg' } },
      ],
    },
    compare: {
      slots: ['reference', 'frame'],
      examples: [
        { id: 'compare-defect', label: 'ex_defect', files: { reference: 'compare_reference.jpg', frame: 'compare_defect.jpg' } },
        { id: 'compare-light', label: 'ex_light', files: { reference: 'compare_reference.jpg', frame: 'compare_light.jpg' } },
      ],
    },
  };
  const MODE_OF_SLOT = { image: 'surface', reference: 'compare', frame: 'compare' };

  // ---------------------------------------------------------------------------
  // Small utilities
  // ---------------------------------------------------------------------------

  /** localStorage wrapper that never throws (private mode, blocked storage). */
  const store = {
    get(key) { try { return window.localStorage.getItem(key); } catch (e) { return null; } },
    set(key, value) { try { window.localStorage.setItem(key, value); } catch (e) { /* ignore */ } },
  };

  /**
   * Create a DOM element. `props`: class, on<event> (function), anything else becomes an attribute
   * (true -> boolean attribute, null/false -> skipped). Children: nodes, strings, arrays, null/false.
   */
  function el(tag, props, ...children) {
    const node = document.createElement(tag);
    for (const key of Object.keys(props || {})) {
      const value = props[key];
      if (value == null || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? '' : String(value));
    }
    appendChildren(node, children);
    return node;
  }

  function appendChildren(node, children) {
    for (const child of children) {
      if (child == null || child === false) continue;
      if (Array.isArray(child)) appendChildren(node, child);
      else if (child instanceof Node) node.appendChild(child);
      else node.appendChild(document.createTextNode(String(child)));
    }
  }

  function clearNode(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function replaceContent(node, ...children) {
    clearNode(node);
    appendChildren(node, children);
  }

  const SVG_NS = 'http://www.w3.org/2000/svg';

  function svg(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const key of Object.keys(attrs || {})) node.setAttribute(key, String(attrs[key]));
    return node;
  }

  /** Small stroke icons (24×24, currentColor). */
  const ICON_PATHS = {
    check: 'M5 12.5l4.5 4.5L19 7.5',
    cross: 'M6 6l12 12M18 6L6 18',
    retake: 'M20 11a8 8 0 10-2.3 5.7M20 4v7h-7',
    camera: 'M4 8h3l2-3h6l2 3h3a1 1 0 011 1v10a1 1 0 01-1 1H4a1 1 0 01-1-1V9a1 1 0 011-1zM12 17a4 4 0 100-8 4 4 0 000 8z',
    image: 'M4 5h16v14H4zM4 16l5-5 4 4 3-3 4 4M15.5 9.5h.01',
    compare: 'M4 6h7v12H4zM13 6h7v12h-7M7.5 3v18',
    scan: 'M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4M4 12h16',
    plus: 'M12 5v14M5 12h14',
    play: 'M8 5v14l11-7z',
  };

  function icon(name, cls) {
    const node = svg('svg', {
      viewBox: '0 0 24 24', width: 24, height: 24, fill: 'none', stroke: 'currentColor',
      'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round',
      'aria-hidden': 'true', focusable: 'false', class: 'icon' + (cls ? ' ' + cls : ''),
    });
    node.appendChild(svg('path', { d: ICON_PATHS[name] || '' }));
    return node;
  }

  const $ = (id) => document.getElementById(id);
  const reducedMotion = () => !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const scrollBehavior = () => (reducedMotion() ? 'auto' : 'smooth');

  function formatTime(date) {
    try {
      return date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch (e) {
      return date.toTimeString().slice(0, 8);
    }
  }

  function formatNumber(x, digits) {
    try {
      return x.toLocaleString('en-GB', { maximumFractionDigits: digits });
    } catch (e) {
      return x.toFixed(digits);
    }
  }

  function vibrate(pattern) {
    try { if (navigator.vibrate) navigator.vibrate(pattern); } catch (e) { /* unsupported */ }
  }

  // ---------------------------------------------------------------------------
  // DOM + state
  // ---------------------------------------------------------------------------

  const dom = {
    slots: $('slots'), runBtn: $('runBtn'), runHint: $('runHint'), result: $('result'), history: $('history'),
    modeHint: $('modeHint'), examples: $('examples'), modeBtns: Array.from(document.querySelectorAll('[data-mode]')),
    toasts: $('toasts'), insecureNote: $('insecureNote'), qrBtn: $('qrBtn'),
    camera: $('camera'), camTitle: $('camTitle'), camClose: $('camClose'), camVideo: $('camVideo'),
    camGhost: $('camGhost'), camMsg: $('camMsg'), camTap: $('camTap'), camFlash: $('camFlash'),
    camShutter: $('camShutter'), camGallery: $('camGallery'),
    ghostRow: $('ghostRow'), ghostRange: $('ghostRange'), ghostValue: $('ghostValue'),
    qrModal: $('qrModal'), qrBox: $('qrBox'), qrUrl: $('qrUrl'), qrLocal: $('qrLocal'), qrClose: $('qrClose'),
  };

  const savedMode = store.get(LS_MODE);
  const state = {
    mode: MODES[savedMode] ? savedMode : 'surface',
    photos: { image: null, reference: null, frame: null }, // { blob, url, w, h, example } per slot
    shown: { surface: null, compare: null },              // history entry on screen, per mode
    history: [],                                          // newest first, at most HISTORY_MAX
    busy: false,
  };
  let historySeq = 0;

  const slotsOf = (mode) => MODES[mode].slots;
  const ready = (mode) => slotsOf(mode).every((s) => !!state.photos[s]);

  // ---------------------------------------------------------------------------
  // Toasts
  // ---------------------------------------------------------------------------

  /** Show a toast; an identical message already on screen is not repeated. */
  function toast(message, kind) {
    const k = kind || 'error';
    for (const node of Array.from(dom.toasts.children)) {
      if (node.dataset.msg === message) return;
    }
    const node = el('div', { class: `toast toast-${k}`, role: k === 'error' ? 'alert' : 'status' }, message);
    node.dataset.msg = message;
    const remove = () => { if (node.parentNode) node.parentNode.removeChild(node); };
    node.addEventListener('click', remove);
    dom.toasts.appendChild(node);
    while (dom.toasts.children.length > 3) dom.toasts.removeChild(dom.toasts.firstChild);
    setTimeout(remove, k === 'error' ? 6000 : 3000);
  }

  function showError(err) {
    console.warn(err);
    toast((err && err.message) || t('err_generic'), 'error');
  }

  // ---------------------------------------------------------------------------
  // Photos: everything goes through a canvas → ≤ MAX_SIDE, EXIF-free JPEG
  // ---------------------------------------------------------------------------

  function canvasToBlob(canvas) {
    return new Promise((resolve, reject) => {
      if (canvas.toBlob) {
        canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error(t('err_decode')))), 'image/jpeg', JPEG_QUALITY);
        return;
      }
      try {
        const [head, body] = canvas.toDataURL('image/jpeg', JPEG_QUALITY).split(',');
        const bin = atob(body);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        resolve(new Blob([bytes], { type: (head.match(/data:([^;]+)/) || [])[1] || 'image/jpeg' }));
      } catch (e) {
        reject(e);
      }
    });
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(t('err_decode')));
      img.src = src;
    });
  }

  /** Draw an <img>/<video> scaled to ≤ MAX_SIDE and encode it as a JPEG photo record. */
  async function encodePhoto(source, w, h) {
    if (!w || !h) throw new Error(t('err_no_frame'));
    const s = Math.min(1, MAX_SIDE / Math.max(w, h));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(w * s));
    canvas.height = Math.max(1, Math.round(h * s));
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    const blob = await canvasToBlob(canvas);
    return { blob, url: URL.createObjectURL(blob), w: canvas.width, h: canvas.height, example: false };
  }

  /** Gallery file (or any image Blob) → photo record (browsers apply the EXIF orientation when drawing an <img>). */
  async function photoFromFile(file) {
    const url = URL.createObjectURL(file);
    try {
      const img = await loadImage(url);
      return await encodePhoto(img, img.naturalWidth, img.naturalHeight);
    } catch (e) {
      throw new Error(t('err_decode'));
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  function setPhoto(slot, photo) {
    const old = state.photos[slot];
    if (old) URL.revokeObjectURL(old.url);
    state.photos[slot] = photo;
    // The result on screen no longer matches the photos (it stays in the history).
    state.shown[MODE_OF_SLOT[slot]] = null;
    renderAll();
  }

  async function handleFile(slot, file) {
    if (!slot || !file || state.busy) return;
    try {
      setPhoto(slot, await photoFromFile(file));
      afterPhoto();
    } catch (err) {
      showError(err);
    }
  }

  /** Once all photos of the mode are there, bring the run button into view. */
  function afterPhoto() {
    if (!ready(state.mode)) return;
    dom.runBtn.scrollIntoView({ block: 'center', behavior: scrollBehavior() });
    try { dom.runBtn.focus({ preventScroll: true }); } catch (e) { dom.runBtn.focus(); }
  }

  /** Briefly highlight a slot's gallery button (fallback when the camera is not available). */
  function pulseGallery(slot) {
    const btn = dom.slots.querySelector(`[data-gallery="${slot}"]`);
    if (!btn) return;
    btn.scrollIntoView({ block: 'center', behavior: scrollBehavior() });
    btn.classList.remove('pulse');
    void btn.offsetWidth; // restart the CSS animation
    btn.classList.add('pulse');
  }

  // ---------------------------------------------------------------------------
  // Built-in examples: load the files into the slots exactly like a gallery pick, then run the real request
  // ---------------------------------------------------------------------------

  async function fetchExample(file) {
    const ctrl = typeof AbortController === 'function' ? new AbortController() : null;
    const timer = ctrl ? setTimeout(() => ctrl.abort(), EXAMPLE_TIMEOUT_MS) : null;
    let blob;
    try {
      const res = await fetch(EXAMPLES_DIR + file, {
        cache: 'no-cache', credentials: 'same-origin', signal: ctrl ? ctrl.signal : undefined,
      });
      if (!res.ok) throw new ApiError(t('ex_failed'));
      blob = await res.blob(); // the body can stall too: still under the timer
    } catch (e) {
      if (e instanceof ApiError) throw new Error(e.message);
      throw new Error(t(e && e.name === 'AbortError' ? 'err_timeout' : 'err_network'));
    } finally {
      if (timer) clearTimeout(timer);
    }
    const photo = await photoFromFile(blob);
    photo.example = true;
    return photo;
  }

  async function runExample(mode, example) {
    if (state.busy) return;
    if (state.mode !== mode) setMode(mode);
    setBusy(true);
    let loaded = null;
    try {
      const slots = Object.keys(example.files);
      const photos = await Promise.all(slots.map((s) => fetchExample(example.files[s])));
      loaded = slots.map((s, i) => [s, photos[i]]);
    } catch (err) {
      showError(err);
    }
    setBusy(false);
    if (!loaded) return;
    loaded.forEach(([slot, photo]) => setPhoto(slot, photo));
    await run();
  }

  // ---------------------------------------------------------------------------
  // Mode switch, examples row, slots, run button
  // ---------------------------------------------------------------------------

  function setMode(mode) {
    if (!MODES[mode] || state.busy || mode === state.mode) return;
    state.mode = mode;
    store.set(LS_MODE, mode);
    renderAll();
  }

  function renderModes() {
    dom.modeBtns.forEach((b) => {
      b.setAttribute('aria-pressed', String(b.getAttribute('data-mode') === state.mode));
      b.disabled = state.busy;
    });
    dom.modeHint.textContent = t(`mode_hint_${state.mode}`);
    document.body.setAttribute('data-mode', state.mode);
  }

  function renderExamples() {
    const mode = state.mode;
    replaceContent(dom.examples,
      el('span', { class: 'examples-label' }, t('examples_label')),
      MODES[mode].examples.map((ex) => el('button', {
        class: 'btn btn-example', type: 'button', disabled: state.busy, 'data-example': ex.id,
        onclick: () => runExample(mode, ex),
      }, icon('play', 'icon-sm'), el('span', null, t(ex.label)))));
  }

  /** A <label> styled as a button around a file input (more reliable on iOS than input.click()). */
  function galleryButton(slot, textKey) {
    const input = el('input', { type: 'file', accept: 'image/*', class: 'visually-hidden', disabled: state.busy });
    input.addEventListener('change', () => {
      const file = input.files && input.files[0];
      input.value = '';
      if (file) handleFile(slot, file);
    });
    return el('label', { class: `btn btn-secondary${state.busy ? ' is-disabled' : ''}`, 'data-gallery': slot },
      input, icon('image'), el('span', null, t(textKey)));
  }

  function slotCard(slot, index, single) {
    const photo = state.photos[slot];
    const preview = photo
      ? el('div', { class: 'slot-preview' },
        el('img', { src: photo.url, alt: t(`slot_${slot}`) }),
        photo.example ? el('span', { class: 'slot-tag' }, t('ex_tag')) : null,
        el('span', { class: 'slot-size' }, t('photo_size', { w: photo.w, h: photo.h })))
      : el('div', { class: 'slot-preview slot-empty' }, icon('image', 'icon-xl'), el('span', null, t('slot_empty')));
    return el('article', { class: `slot${photo ? ' slot-filled' : ''}${single ? ' slot-single' : ''}`, 'aria-label': t(`slot_${slot}`), 'data-slot': slot },
      el('header', { class: 'slot-head' },
        single ? null : el('span', { class: 'slot-num', 'aria-hidden': 'true' }, String(index + 1)),
        el('div', null,
          el('h2', { class: 'slot-title' }, t(`slot_${slot}`)),
          el('p', { class: 'slot-sub' }, t(`slot_${slot}_sub`)))),
      preview,
      el('div', { class: 'slot-actions' },
        el('button', { class: 'btn btn-primary', type: 'button', disabled: state.busy, onclick: () => openCamera(slot) },
          icon('camera'), el('span', null, t(photo ? 'btn_retake' : 'btn_camera'))),
        galleryButton(slot, photo ? 'btn_replace' : 'btn_gallery')));
  }

  function renderSlots() {
    const slots = slotsOf(state.mode);
    dom.slots.classList.toggle('slots-single', slots.length === 1);
    replaceContent(dom.slots, slots.map((slot, i) => slotCard(slot, i, slots.length === 1)));
  }

  function renderRun() {
    const mode = state.mode;
    const ok = ready(mode);
    dom.runBtn.disabled = !ok || state.busy;
    dom.runBtn.setAttribute('aria-busy', String(state.busy));
    replaceContent(dom.runBtn,
      state.busy ? el('span', { class: 'spinner spinner-btn', 'aria-hidden': 'true' }) : icon(mode === 'surface' ? 'scan' : 'compare'),
      el('span', null, t(state.busy ? `busy_${mode}` : `run_${mode}`)));
    dom.runHint.textContent = t(`need_${mode}`);
    dom.runHint.hidden = ok;
  }

  function renderAll() {
    renderModes();
    renderExamples();
    renderSlots();
    renderRun();
    renderResult();
    renderHistory();
  }

  function setBusy(on) {
    state.busy = on;
    renderModes();
    renderExamples();
    renderSlots();
    renderRun();
    dom.result.querySelectorAll('button').forEach((b) => { b.disabled = on; });
    dom.history.querySelectorAll('button').forEach((b) => { b.disabled = on; });
  }

  // ---------------------------------------------------------------------------
  // API
  // ---------------------------------------------------------------------------

  class ApiError extends Error {}

  /** Friendly text for a known server error code, else the server's own message. */
  function errorMessage(data, status) {
    if (data && typeof data.error === 'string') {
      const key = `api_${data.error}`;
      const known = t(key);
      if (known !== key) return known;
    }
    if (data && typeof data.message === 'string' && data.message) return data.message;
    if (status === 413) return t('api_too_large');
    return t('err_http', { status });
  }

  /** POST a multipart form, return the parsed JSON; `valid(data)` checks the response shape. */
  async function postForm(url, form, valid) {
    const ctrl = typeof AbortController === 'function' ? new AbortController() : null;
    const timer = ctrl ? setTimeout(() => ctrl.abort(), REQUEST_TIMEOUT_MS) : null;
    try {
      const res = await fetch(url, {
        method: 'POST', body: form, cache: 'no-store', credentials: 'same-origin',
        headers: { Accept: 'application/json' }, signal: ctrl ? ctrl.signal : undefined,
      });
      let data = null;
      if ((res.headers.get('content-type') || '').indexOf('json') >= 0) {
        try { data = await res.json(); } catch (e) { data = null; }
      }
      if (!res.ok) throw new ApiError(errorMessage(data, res.status));
      if (!data || !valid(data)) throw new ApiError(t('err_bad_response'));
      return data;
    } catch (e) {
      if (e instanceof ApiError) throw e;
      throw new ApiError(t(e && e.name === 'AbortError' ? 'err_timeout' : 'err_network'));
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  /** Surface check: one photo → /api/scan → { result, overlay, image_sha256 }. */
  async function requestScan() {
    const form = new FormData();
    form.append('image', state.photos.image.blob, 'image.jpg');
    const data = await postForm(SCAN_URL, form, (d) => !!d.result);
    return { main: data.result, mainOverlay: data.overlay, base: null, baseOverlay: null };
  }

  /** Compare with reference: two photos → /api/compare → AVA (pbp) + plain subtraction (pixdiff). */
  async function requestCompare() {
    const form = new FormData();
    form.append('reference', state.photos.reference.blob, 'reference.jpg');
    form.append('frame', state.photos.frame.blob, 'frame.jpg');
    const data = await postForm(COMPARE_URL, form, (d) => !!(d.results && d.results.pbp));
    const overlays = data.overlays || {};
    return {
      main: data.results.pbp, mainOverlay: overlays.pbp,
      base: data.results.pixdiff || null, baseOverlay: overlays.pixdiff,
    };
  }

  async function run() {
    const mode = state.mode;
    if (!ready(mode) || state.busy) return;
    const examples = slotsOf(mode).every((s) => state.photos[s].example);
    setBusy(true);
    let entry = null;
    try {
      const out = await (mode === 'surface' ? requestScan() : requestCompare());
      entry = Object.assign({ id: ++historySeq, at: new Date(), mode, example: examples }, out);
      state.history.unshift(entry);
      if (state.history.length > HISTORY_MAX) state.history.length = HISTORY_MAX;
      state.shown[mode] = entry;
      vibrate(verdictOf(entry.main) === 'fail' ? [80, 60, 80] : 40);
    } catch (err) {
      showError(err);
    } finally {
      setBusy(false);
      renderResult();
      renderHistory();
    }
    if (entry && state.mode === mode) dom.result.scrollIntoView({ block: 'start', behavior: scrollBehavior() });
  }

  // ---------------------------------------------------------------------------
  // Result rendering
  // ---------------------------------------------------------------------------

  function verdictOf(rec) {
    const v = rec && rec.verdict;
    return v === 'pass' || v === 'fail' ? v : 'retake';
  }

  function zonesOf(rec) {
    return rec && Array.isArray(rec.zones) ? rec.zones : [];
  }

  function reasonText(reason) {
    const key = `rr_${reason}`;
    const s = t(key);
    return s === key ? t('rr_unknown') : s;
  }

  /** Advice under a RETAKE verdict (glare → change the angle, etc.), or null. */
  function retakeHint(rec, mode) {
    const hints = Array.isArray(rec.hints) ? rec.hints : [];
    let key = hints.indexOf('change_angle') >= 0 ? 'rh_glare' : `rh_${rec.reason}`;
    if (hints.indexOf('bright_spot') >= 0) key = 'rh_bright_spot';
    if (rec.reason === 'low_coverage') key = `rh_low_coverage_${mode}`;
    const s = t(key);
    return s === key ? null : s;
  }

  function hintText(hint) {
    const key = `zh_${hint}`;
    const s = t(key);
    return s === key ? null : s;
  }

  function failTitle(n) {
    return n === 1 ? t('v_fail_one') : t('v_fail_many', { n });
  }

  /**
   * Baseline display rule: plain subtraction's "mismatch" (the API answers RETAKE when more than
   * Params.mismatch_frac of the checked cells changed) is shown as FAIL — a large part of the frame
   * changed, with the measured fraction.
   */
  const baselineMismatch = (rec) => !!rec && rec.reason === 'mismatch';

  const ICON_FOR = { pass: 'check', fail: 'cross', retake: 'retake' };

  function banner(v, title, sub, small) {
    return el('div', { class: `verdict verdict-${v}${small ? ' verdict-small' : ''}`, role: 'status', 'data-verdict': v },
      icon(ICON_FOR[v], 'verdict-icon'),
      el('div', { class: 'verdict-text' },
        el('strong', { class: 'verdict-title' }, title),
        sub ? el('span', { class: 'verdict-sub' }, sub) : null));
  }

  /** PASS — no changes / defects found / FAIL — N defect zones / RETAKE — reason. */
  function verdictBanner(rec, mode, small) {
    const v = verdictOf(rec);
    if (v === 'pass') return banner(v, t(mode === 'surface' ? 'v_pass_surface' : 'v_pass'), null, small);
    if (v === 'fail') return banner(v, failTitle(zonesOf(rec).length), null, small);
    return banner(v, t('v_retake', { reason: reasonText(rec.reason) }), retakeHint(rec, mode), small);
  }

  /** Short verdict for chips; `baseline` applies the baseline display rule. */
  function shortVerdict(rec, baseline) {
    if (baseline && baselineMismatch(rec)) return { v: 'fail', text: t('v_fail_short') };
    const v = verdictOf(rec);
    if (v === 'pass') return { v, text: t('v_pass_short') };
    if (v === 'fail') return { v, text: `${t('v_fail_short')} · ${zonesOf(rec).length}` };
    return { v, text: t('v_retake_short') };
  }

  function chip(rec, baseline) {
    const s = shortVerdict(rec, baseline);
    return el('span', { class: `chip chip-${s.v}` }, s.text);
  }

  const imgSrc = (b64) => (b64 ? `data:image/jpeg;base64,${b64}` : null);

  /**
   * Overlay image with an SVG layer of zone boxes (bbox_norm), so a zone can be highlighted.
   * Returns { node, rects, highlight(i) } or null without an image.
   */
  function zoneFigure(src, zones, alt) {
    if (!src) return null;
    const layer = svg('svg', { class: 'fig-zones', viewBox: '0 0 100 100', preserveAspectRatio: 'none', 'aria-hidden': 'true' });
    const dim = svg('path', { class: 'fig-dim', 'fill-rule': 'evenodd', d: '' });
    layer.appendChild(dim);
    const boxes = zones.map((z) => {
      const b = Array.isArray(z.bbox_norm) ? z.bbox_norm.map((v) => Number(v) * 100) : [0, 0, 0, 0];
      return { x: b[0], y: b[1], w: b[2], h: b[3] };
    });
    const rects = boxes.map((b) => {
      const r = svg('rect', { class: 'fig-zone', x: b.x, y: b.y, width: b.w, height: b.h });
      layer.appendChild(r);
      return r;
    });
    const node = el('figure', { class: 'fig' },
      el('div', { class: 'fig-box' }, el('img', { class: 'fig-img', src, alt }), layer));
    return {
      node,
      rects,
      highlight(i) {
        rects.forEach((r, k) => r.classList.toggle('is-selected', k === i));
        const b = boxes[i];
        // Dim everything except the selected box (a frame-sized path with a hole).
        dim.setAttribute('d', b ? `M0 0H100V100H0Z M${b.x} ${b.y}h${b.w}v${b.h}h${-b.w}Z` : '');
      },
    };
  }

  /** Tappable zone list; selecting a zone highlights it on the figure. */
  function zoneList(rec, fig) {
    const zones = zonesOf(rec);
    if (!zones.length) return null;
    const ws = Array.isArray(rec.work_size) ? rec.work_size : null;
    const total = ws ? ws[0] * ws[1] : 0;
    const buttons = [];
    let selected = -1;

    function select(i) {
      selected = selected === i ? -1 : i;
      buttons.forEach((b, k) => b.setAttribute('aria-pressed', String(k === selected)));
      if (!fig) return;
      fig.highlight(selected);
      if (selected >= 0) fig.node.scrollIntoView({ block: 'nearest', behavior: scrollBehavior() });
    }

    zones.forEach((z, i) => {
      const n = typeof z.id === 'number' ? z.id : i + 1; // same number as on the overlay image
      const pct = total && z.area_px ? (100 * z.area_px) / total : null;
      const details = [
        pct == null ? null : t('zone_area', { p: pct < 0.1 ? `< ${formatNumber(0.1, 1)}` : formatNumber(pct, 1) }),
        z.hint ? hintText(z.hint) : null,
      ].filter(Boolean).join(' · ');
      buttons.push(el('button', { class: 'zone', type: 'button', 'aria-pressed': 'false', onclick: () => select(i) },
        el('span', { class: 'zone-num' }, String(n)),
        el('span', { class: 'zone-body' },
          el('strong', null, t('zone_name', { n })),
          details ? el('span', { class: 'zone-meta' }, details) : null)));
      if (fig && fig.rects[i]) fig.rects[i].addEventListener('click', () => select(i));
    });

    return el('div', { class: 'zones' },
      el('h3', { class: 'zones-title' }, t('zones_title')),
      el('p', { class: 'help small' }, t('zones_help')),
      el('div', { class: 'zone-list' }, buttons));
  }

  /** Short SHA-256 fingerprint; tap shows the full value. */
  function shaButton(sha) {
    const short = `${t('meta_sha')} ${sha.slice(0, 12)}…`;
    const btn = el('button', { class: 'sha', type: 'button', 'aria-expanded': 'false', title: t('sha_tap') }, short);
    btn.addEventListener('click', () => {
      const open = btn.getAttribute('aria-expanded') !== 'true';
      btn.setAttribute('aria-expanded', String(open));
      btn.textContent = open ? `${t('meta_sha')} ${sha}` : short;
    });
    return btn;
  }

  /** Processing ms · coverage % · fingerprint. */
  function metaLine(rec) {
    const ms = rec.timing_ms && typeof rec.timing_ms.total === 'number' ? rec.timing_ms.total : null;
    const sha = typeof rec.signature_sha256 === 'string' && rec.signature_sha256 ? rec.signature_sha256 : null;
    return el('div', { class: 'meta' },
      ms == null ? null : el('span', null, t('meta_ms', { n: Math.round(ms) })),
      typeof rec.coverage === 'number' ? el('span', null, t('meta_coverage', { n: Math.round(rec.coverage * 100) })) : null,
      sha ? shaButton(sha) : null);
  }

  function mainPanel(entry) {
    const rec = entry.main;
    const fig = zoneFigure(imgSrc(entry.mainOverlay), zonesOf(rec), t('overlay_alt'));
    return el('section', { class: 'panel panel-main', 'data-panel': 'main' },
      el('h2', { class: 'panel-title' }, t(`result_title_${entry.mode}`)),
      verdictBanner(rec, entry.mode, false),
      fig ? fig.node : null,
      zoneList(rec, fig),
      metaLine(rec));
  }

  /** The baseline's "mismatch" banner: the measured changed fraction; the AVA note only if AVA passed. */
  function mismatchBanner(rec, main) {
    const p = Math.round(100 * (Number(rec.changed_frac) || 0));
    const why = main && verdictOf(main) === 'pass' ? ` ${t('base_mismatch_why')}` : '';
    return banner('fail', t('base_mismatch'), t('base_mismatch_frac', { p }) + why, true);
  }

  /** Plain reference subtraction: pixdiff on the same aligned frame (the key demo moment). */
  function baselinePanel(rec, overlay, main) {
    let body;
    if (!rec) {
      body = el('p', { class: 'muted' }, t('base_none'));
    } else {
      const src = imgSrc(overlay);
      body = [
        baselineMismatch(rec) ? mismatchBanner(rec, main) : verdictBanner(rec, 'compare', true),
        src ? el('figure', { class: 'fig' }, el('div', { class: 'fig-box' }, el('img', { class: 'fig-img', src, alt: t('base_alt') }))) : null,
      ];
    }
    return el('details', { class: 'panel panel-base', open: true, 'data-panel': 'baseline' },
      el('summary', { class: 'base-summary' }, el('span', { class: 'base-title' }, t('base_title')), rec ? chip(rec, true) : null),
      el('p', { class: 'help small' }, t('base_sub')),
      body);
  }

  function renderResult() {
    const entry = state.shown[state.mode];
    dom.result.hidden = !entry;
    if (!entry) {
      clearNode(dom.result);
      return;
    }
    const surface = entry.mode === 'surface';
    const actions = surface
      ? [el('button', { class: 'btn btn-accent btn-lg', type: 'button', disabled: state.busy, onclick: newPhoto },
        icon('camera'), el('span', null, t('new_photo')))]
      : [
        el('button', { class: 'btn btn-accent btn-lg', type: 'button', disabled: state.busy, onclick: newCheck },
          icon('camera'), el('span', null, t('new_check'))),
        el('button', { class: 'btn btn-secondary btn-lg', type: 'button', disabled: state.busy, onclick: newReference },
          icon('plus'), el('span', null, t('new_reference'))),
      ];
    replaceContent(dom.result,
      el('div', { class: `result-grid${surface ? ' result-grid-single' : ''}` },
        mainPanel(entry),
        surface ? null : baselinePanel(entry.base, entry.baseOverlay, entry.main)),
      el('div', { class: `result-actions${surface ? ' result-actions-single' : ''}` }, actions));
  }

  function renderHistory() {
    dom.history.hidden = !state.history.length;
    if (!state.history.length) {
      clearNode(dom.history);
      return;
    }
    replaceContent(dom.history,
      el('h2', { class: 'section-title' }, t('history_title')),
      el('div', { class: 'hchips' }, state.history.map((entry) => {
        const s = shortVerdict(entry.main, false);
        return el('button', {
          class: `hchip hchip-${s.v}`, type: 'button', disabled: state.busy,
          'aria-pressed': String(entry === state.shown[state.mode]),
          onclick: () => {
            if (state.busy) return;
            state.shown[entry.mode] = entry;
            if (entry.mode !== state.mode) setMode(entry.mode);
            else { renderResult(); renderHistory(); }
            dom.result.scrollIntoView({ block: 'start', behavior: scrollBehavior() });
          },
        },
        el('span', { class: 'hchip-m' }, t(`mode_${entry.mode}_short`)),
        el('span', { class: 'hchip-v' }, s.text),
        el('span', { class: 'hchip-t' }, formatTime(entry.at)));
      })));
  }

  /** Compare: keep the reference, clear the Check photo and reopen its camera (inside the click: a user gesture). */
  function newCheck() {
    setPhoto('frame', null);
    openCamera('frame');
  }

  function newReference() {
    setPhoto('frame', null);
    setPhoto('reference', null);
    window.scrollTo({ top: 0, behavior: scrollBehavior() });
  }

  /** Surface: clear the photo and reopen the camera. */
  function newPhoto() {
    setPhoto('image', null);
    openCamera('image');
  }

  // ---------------------------------------------------------------------------
  // Camera (full-screen, rear). iOS Safari: playsinline + muted before srcObject, play() may need a tap.
  // ---------------------------------------------------------------------------

  const cam = { slot: null, stream: null, token: 0, opener: null };

  const cameraSupported = () => !!(window.isSecureContext && navigator.mediaDevices
    && typeof navigator.mediaDevices.getUserMedia === 'function');

  // Non-secure context (plain HTTP on a LAN): no getUserMedia, so "Camera" opens the system camera app.
  const captureInput = el('input', {
    type: 'file', accept: 'image/*', capture: 'environment', class: 'visually-hidden', tabindex: '-1', 'aria-hidden': 'true',
  });
  let captureSlot = null;
  captureInput.addEventListener('change', () => {
    const file = captureInput.files && captureInput.files[0];
    captureInput.value = '';
    if (file) handleFile(captureSlot, file);
  });
  document.body.appendChild(captureInput);

  function stopTracks(stream) {
    if (!stream) return;
    stream.getTracks().forEach((track) => { try { track.stop(); } catch (e) { /* ignore */ } });
  }

  function stopStream() {
    stopTracks(cam.stream);
    cam.stream = null;
    try { dom.camVideo.pause(); } catch (e) { /* ignore */ }
    dom.camVideo.srcObject = null;
    dom.camShutter.disabled = true;
  }

  async function getStream() {
    try {
      return await navigator.mediaDevices.getUserMedia(CAMERA_CONSTRAINTS);
    } catch (e) {
      // Some devices reject resolution hints; retry with the bare minimum.
      if (e && (e.name === 'OverconstrainedError' || e.name === 'ConstraintNotSatisfiedError')) {
        return navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: 'environment' } });
      }
      throw e;
    }
  }

  function setCamMessage(on) {
    dom.camMsg.hidden = !on;
  }

  /**
   * Fit the reference ghost to the live video's content box (the video is letterboxed with
   * object-fit: contain) and crop it to that box (object-fit: cover), so it lines up with the stream
   * even when the reference photo has another aspect ratio (e.g. a 4:3 gallery photo over 16:9 video).
   */
  function fitGhost() {
    const v = dom.camVideo;
    const g = dom.camGhost;
    const stage = v.parentElement;
    if (g.hidden || !v.videoWidth || !v.videoHeight || !stage) {
      g.style.left = g.style.top = g.style.width = g.style.height = '';
      return;
    }
    const sw = stage.clientWidth;
    const sh = stage.clientHeight;
    const s = Math.min(sw / v.videoWidth, sh / v.videoHeight);
    const w = v.videoWidth * s;
    const h = v.videoHeight * s;
    g.style.left = `${(sw - w) / 2}px`;
    g.style.top = `${(sh - h) / 2}px`;
    g.style.width = `${w}px`;
    g.style.height = `${h}px`;
  }

  function onVideoReady() {
    const v = dom.camVideo;
    fitGhost();
    if (!cam.stream || !v.videoWidth || v.paused) return;
    setCamMessage(false);
    dom.camTap.hidden = true;
    dom.camShutter.disabled = false;
  }

  async function startStream() {
    const token = ++cam.token;
    stopStream();
    dom.camTap.hidden = true;
    setCamMessage(true);
    let stream;
    try {
      stream = await getStream();
    } catch (e) {
      if (token === cam.token) cameraFailed(e && (e.name === 'NotAllowedError' || e.name === 'SecurityError') ? 'denied' : 'unavailable');
      return;
    }
    if (token !== cam.token) { stopTracks(stream); return; } // closed while the permission prompt was open
    cam.stream = stream;
    const v = dom.camVideo;
    v.muted = true;
    v.playsInline = true;
    v.setAttribute('muted', '');
    v.setAttribute('playsinline', '');
    v.setAttribute('webkit-playsinline', '');
    v.srcObject = stream;
    try {
      await v.play();
    } catch (e) {
      if (token === cam.token) { setCamMessage(false); dom.camTap.hidden = false; } // autoplay blocked: needs a tap
      return;
    }
    onVideoReady();
  }

  function applyGhost(value) {
    dom.ghostRange.value = String(value);
    dom.ghostValue.textContent = `${value}%`;
    dom.camGhost.style.opacity = String(value / 100);
  }

  function savedGhost() {
    const v = parseInt(store.get(LS_GHOST), 10);
    return isFinite(v) ? Math.min(100, Math.max(0, v)) : GHOST_DEFAULT;
  }

  function openCamera(slot) {
    if (state.busy) return;
    if (!cameraSupported()) {
      captureSlot = slot;
      captureInput.click();
      return;
    }
    cam.slot = slot;
    cam.opener = document.activeElement;
    dom.camTitle.textContent = t(`cam_title_${slot}`);
    const ref = slot === 'frame' ? state.photos.reference : null;
    dom.ghostRow.hidden = !ref;
    dom.camGhost.hidden = !ref;
    if (ref) {
      dom.camGhost.src = ref.url;
      applyGhost(savedGhost());
    } else {
      dom.camGhost.removeAttribute('src');
    }
    fitGhost();
    dom.camera.hidden = false;
    document.body.classList.add('modal-open');
    dom.camClose.focus();
    startStream(); // getUserMedia is requested synchronously inside the user's click
  }

  function closeCamera() {
    if (dom.camera.hidden) return;
    cam.token++;
    stopStream();
    dom.camera.hidden = true;
    document.body.classList.remove('modal-open');
    const opener = cam.opener;
    cam.slot = null;
    cam.opener = null;
    if (opener && document.body.contains(opener)) opener.focus();
  }

  function cameraFailed(reason) {
    const slot = cam.slot;
    closeCamera();
    toast(`${t(`cam_fail_${reason}`)} ${t('cam_fallback')}`, 'error');
    pulseGallery(slot);
  }

  async function shoot() {
    const slot = cam.slot;
    const v = dom.camVideo;
    if (!slot || !v.videoWidth) {
      toast(t('err_no_frame'), 'error');
      return;
    }
    dom.camShutter.disabled = true;
    dom.camFlash.classList.remove('on');
    void dom.camFlash.offsetWidth; // restart the CSS animation
    dom.camFlash.classList.add('on');
    vibrate(30);
    try {
      const photo = await encodePhoto(v, v.videoWidth, v.videoHeight);
      closeCamera();
      setPhoto(slot, photo);
      afterPhoto();
    } catch (err) {
      dom.camShutter.disabled = false;
      showError(err);
    }
  }

  dom.camVideo.addEventListener('loadedmetadata', onVideoReady);
  dom.camVideo.addEventListener('playing', onVideoReady);
  dom.camVideo.addEventListener('resize', fitGhost); // stream resolution changed (rotation)
  window.addEventListener('resize', () => { if (!dom.camera.hidden) fitGhost(); });
  dom.camShutter.addEventListener('click', shoot);
  dom.camClose.addEventListener('click', closeCamera);
  dom.camTap.addEventListener('click', () => {
    dom.camVideo.play().then(onVideoReady).catch(() => { /* still blocked: keep the button */ });
  });
  dom.camGallery.addEventListener('change', () => {
    const file = dom.camGallery.files && dom.camGallery.files[0];
    const slot = cam.slot;
    dom.camGallery.value = '';
    if (!file) return;
    closeCamera();
    handleFile(slot, file);
  });
  dom.ghostRange.addEventListener('input', () => {
    const v = Number(dom.ghostRange.value);
    applyGhost(v);
    store.set(LS_GHOST, String(v));
  });
  // Release the camera while the tab is hidden; resume when it is back.
  document.addEventListener('visibilitychange', () => {
    if (dom.camera.hidden) return;
    if (document.hidden) {
      cam.token++;
      stopStream();
    } else {
      startStream();
    }
  });
  window.addEventListener('pagehide', closeCamera);

  // ---------------------------------------------------------------------------
  // QR modal: link to this site for phones (a laptop shows it at the demo)
  // ---------------------------------------------------------------------------

  let qrOpener = null;

  function isLocalHost(host) {
    return /^(localhost|127\.|0\.0\.0\.0$|\[?::1\]?$)/.test(host);
  }

  function openQr() {
    const url = location.origin;
    dom.qrUrl.textContent = url;
    const ok = window.AvaQR && window.AvaQR.render(dom.qrBox, url, { ecc: 'M', label: url });
    if (!ok) replaceContent(dom.qrBox, el('p', { class: 'note' }, t('qr_failed')));
    dom.qrLocal.hidden = !isLocalHost(location.hostname);
    qrOpener = document.activeElement;
    dom.qrModal.hidden = false;
    document.body.classList.add('modal-open');
    dom.qrClose.focus();
  }

  function closeQr() {
    if (dom.qrModal.hidden) return;
    dom.qrModal.hidden = true;
    if (dom.camera.hidden) document.body.classList.remove('modal-open');
    if (qrOpener && document.body.contains(qrOpener)) qrOpener.focus();
    qrOpener = null;
  }

  dom.qrBtn.addEventListener('click', openQr);
  dom.qrClose.addEventListener('click', closeQr);
  dom.qrModal.querySelector('[data-close]').addEventListener('click', closeQr);

  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (!dom.qrModal.hidden) closeQr();
    else closeCamera();
  });

  // ---------------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------------

  dom.modeBtns.forEach((b) => b.addEventListener('click', () => setMode(b.getAttribute('data-mode'))));
  dom.runBtn.addEventListener('click', run);
  dom.insecureNote.hidden = cameraSupported();
  renderAll();
})();
