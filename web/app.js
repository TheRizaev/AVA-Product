/*
 * AVA — one-screen client: a reference photo (good part) + a photo to check → POST /api/compare → verdict.
 * Vanilla JS, no build step. Photos are downscaled on the device (≤ 1600 px, EXIF-free JPEG) and live only
 * in this page; the result history is kept for this session only (nothing is stored on the server).
 */
(function () {
  'use strict';

  const I18N = window.I18N;
  const t = I18N.t;

  // ---------------------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------------------

  const COMPARE_URL = '/api/compare?methods=pbp,pixdiff';
  const SLOTS = ['reference', 'frame'];
  const MAX_SIDE = 1600;               // longest side of an uploaded photo, px
  const JPEG_QUALITY = 0.9;
  const REQUEST_TIMEOUT_MS = 120000;
  const HISTORY_MAX = 10;
  const GHOST_DEFAULT = 40;            // % opacity of the reference over the live camera
  const LS_GHOST = 'ava.ghostOpacity';
  const CAMERA_CONSTRAINTS = {
    audio: false,
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
  };

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
    plus: 'M12 5v14M5 12h14',
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

  function locale() {
    return I18N.getLang() === 'ru' ? 'ru-RU' : 'en-GB';
  }

  function formatTime(date) {
    try {
      return date.toLocaleTimeString(locale(), { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch (e) {
      return date.toTimeString().slice(0, 8);
    }
  }

  function formatNumber(x, digits) {
    try {
      return x.toLocaleString(locale(), { maximumFractionDigits: digits });
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
    toasts: $('toasts'), insecureNote: $('insecureNote'), qrBtn: $('qrBtn'),
    camera: $('camera'), camTitle: $('camTitle'), camClose: $('camClose'), camVideo: $('camVideo'),
    camGhost: $('camGhost'), camMsg: $('camMsg'), camTap: $('camTap'), camFlash: $('camFlash'),
    camShutter: $('camShutter'), camGallery: $('camGallery'),
    ghostRow: $('ghostRow'), ghostRange: $('ghostRange'), ghostValue: $('ghostValue'),
    qrModal: $('qrModal'), qrBox: $('qrBox'), qrUrl: $('qrUrl'), qrLocal: $('qrLocal'), qrClose: $('qrClose'),
  };

  const state = {
    photos: { reference: null, frame: null }, // { blob, url, w, h } per slot
    shown: null,                              // history entry on screen
    history: [],                              // newest first, at most HISTORY_MAX
    busy: false,
  };
  let historySeq = 0;

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
    console.error(err);
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
    return { blob, url: URL.createObjectURL(blob), w: canvas.width, h: canvas.height };
  }

  /** Gallery file → photo record (browsers apply the EXIF orientation when drawing an <img>). */
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
    state.shown = null; // the result on screen no longer matches the photos (it stays in the history)
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

  /** Once both photos are there, bring «Сравнить» into view. */
  function afterPhoto() {
    if (!(state.photos.reference && state.photos.frame)) return;
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
  // Slots + run button
  // ---------------------------------------------------------------------------

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

  function slotCard(slot, index) {
    const photo = state.photos[slot];
    const preview = photo
      ? el('div', { class: 'slot-preview' },
        el('img', { src: photo.url, alt: t(`slot_${slot}`) }),
        el('span', { class: 'slot-size' }, t('photo_size', { w: photo.w, h: photo.h })))
      : el('div', { class: 'slot-preview slot-empty' }, icon('image', 'icon-xl'), el('span', null, t('slot_empty')));
    return el('article', { class: `slot${photo ? ' slot-filled' : ''}`, 'aria-label': t(`slot_${slot}`) },
      el('header', { class: 'slot-head' },
        el('span', { class: 'slot-num', 'aria-hidden': 'true' }, String(index + 1)),
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
    replaceContent(dom.slots, SLOTS.map((slot, i) => slotCard(slot, i)));
  }

  function renderRun() {
    const ready = !!(state.photos.reference && state.photos.frame);
    dom.runBtn.disabled = !ready || state.busy;
    dom.runBtn.setAttribute('aria-busy', String(state.busy));
    replaceContent(dom.runBtn,
      state.busy ? el('span', { class: 'spinner spinner-btn', 'aria-hidden': 'true' }) : icon('compare'),
      el('span', null, t(state.busy ? 'compare_busy' : 'compare_run')));
    dom.runHint.hidden = ready;
  }

  function renderAll() {
    renderSlots();
    renderRun();
    renderResult();
    renderHistory();
  }

  function setBusy(on) {
    state.busy = on;
    renderSlots();
    renderRun();
    dom.result.querySelectorAll('button').forEach((b) => { b.disabled = on; });
    dom.history.querySelectorAll('button').forEach((b) => { b.disabled = on; });
  }

  // ---------------------------------------------------------------------------
  // API
  // ---------------------------------------------------------------------------

  class ApiError extends Error {}

  /** Localised text for a known server error code, else the server's own message. */
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

  async function postCompare(form) {
    const ctrl = typeof AbortController === 'function' ? new AbortController() : null;
    const timer = ctrl ? setTimeout(() => ctrl.abort(), REQUEST_TIMEOUT_MS) : null;
    try {
      const res = await fetch(COMPARE_URL, {
        method: 'POST', body: form, cache: 'no-store', credentials: 'same-origin',
        headers: { Accept: 'application/json' }, signal: ctrl ? ctrl.signal : undefined,
      });
      let data = null;
      if ((res.headers.get('content-type') || '').indexOf('json') >= 0) {
        try { data = await res.json(); } catch (e) { data = null; }
      }
      if (!res.ok) throw new ApiError(errorMessage(data, res.status));
      if (!data || !data.results || !data.results.pbp) throw new ApiError(t('err_bad_response'));
      return data;
    } catch (e) {
      if (e instanceof ApiError) throw e;
      throw new ApiError(t(e && e.name === 'AbortError' ? 'err_timeout' : 'err_network'));
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  async function runCompare() {
    const { reference, frame } = state.photos;
    if (!reference || !frame || state.busy) return;
    setBusy(true);
    try {
      const form = new FormData();
      form.append('reference', reference.blob, 'reference.jpg');
      form.append('frame', frame.blob, 'frame.jpg');
      const data = await postCompare(form);
      const entry = { id: ++historySeq, at: new Date(), data };
      state.history.unshift(entry);
      if (state.history.length > HISTORY_MAX) state.history.length = HISTORY_MAX;
      state.shown = entry;
      const v = verdictOf(data.results.pbp);
      vibrate(v === 'fail' ? [80, 60, 80] : 40);
    } catch (err) {
      showError(err);
    } finally {
      setBusy(false);
      renderResult();
      renderHistory();
    }
    if (state.shown) dom.result.scrollIntoView({ block: 'start', behavior: scrollBehavior() });
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
  function retakeHint(rec) {
    const hints = Array.isArray(rec.hints) ? rec.hints : [];
    const key = hints.indexOf('change_angle') >= 0 ? 'rh_glare' : `rh_${rec.reason}`;
    const s = t(key);
    return s === key ? null : s;
  }

  function hintText(hint) {
    const key = `zh_${hint}`;
    const s = t(key);
    return s === key ? null : s;
  }

  /** Baseline display rule: plain subtraction's "mismatch" means it saw almost the whole frame as changed. */
  const baselineMismatch = (rec) => !!rec && rec.reason === 'mismatch';

  const ICON_FOR = { pass: 'check', fail: 'cross', retake: 'retake' };

  function banner(v, title, sub, small) {
    return el('div', { class: `verdict verdict-${v}${small ? ' verdict-small' : ''}`, role: 'status' },
      icon(ICON_FOR[v], 'verdict-icon'),
      el('div', { class: 'verdict-text' },
        el('strong', { class: 'verdict-title' }, title),
        sub ? el('span', { class: 'verdict-sub' }, sub) : null));
  }

  /** ГОДНА — изменений нет / БРАК — найдено зон: N / ПЕРЕСНЯТЬ: reason. */
  function verdictBanner(rec, small) {
    const v = verdictOf(rec);
    if (v === 'pass') return banner(v, t('v_pass'), null, small);
    if (v === 'fail') return banner(v, t('v_fail', { n: zonesOf(rec).length }), null, small);
    return banner(v, t('v_retake', { reason: reasonText(rec.reason) }), retakeHint(rec), small);
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
      if (fig) fig.rects[i].addEventListener('click', () => select(i));
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

  function mainPanel(rec, overlay) {
    const fig = zoneFigure(imgSrc(overlay), zonesOf(rec), t('overlay_alt'));
    return el('section', { class: 'panel panel-main' },
      el('h2', { class: 'panel-title' }, t('result_title')),
      verdictBanner(rec, false),
      fig ? fig.node : null,
      zoneList(rec, fig),
      metaLine(rec));
  }

  /** «Простое вычитание из эталона»: pixdiff on the same aligned frame (the key demo moment). */
  function baselinePanel(rec, overlay) {
    let body;
    if (!rec) {
      body = el('p', { class: 'muted' }, t('base_none'));
    } else {
      const src = imgSrc(overlay);
      body = [
        baselineMismatch(rec) ? banner('fail', t('base_mismatch'), t('base_mismatch_why'), true) : verdictBanner(rec, true),
        src ? el('figure', { class: 'fig' }, el('div', { class: 'fig-box' }, el('img', { class: 'fig-img', src, alt: t('base_alt') }))) : null,
      ];
    }
    return el('details', { class: 'panel panel-base', open: true },
      el('summary', { class: 'base-summary' }, el('span', { class: 'base-title' }, t('base_title')), rec ? chip(rec, true) : null),
      el('p', { class: 'help small' }, t('base_sub')),
      body);
  }

  function renderResult() {
    const entry = state.shown;
    dom.result.hidden = !entry;
    if (!entry) {
      clearNode(dom.result);
      return;
    }
    const results = entry.data.results;
    const overlays = entry.data.overlays || {};
    replaceContent(dom.result,
      el('div', { class: 'result-grid' },
        mainPanel(results.pbp, overlays.pbp),
        baselinePanel(results.pixdiff, overlays.pixdiff)),
      el('div', { class: 'result-actions' },
        el('button', { class: 'btn btn-accent btn-lg', type: 'button', disabled: state.busy, onclick: newCheck },
          icon('camera'), el('span', null, t('new_check'))),
        el('button', { class: 'btn btn-secondary btn-lg', type: 'button', disabled: state.busy, onclick: newReference },
          icon('plus'), el('span', null, t('new_reference')))));
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
        const s = shortVerdict(entry.data.results.pbp, false);
        return el('button', {
          class: `hchip hchip-${s.v}`, type: 'button', disabled: state.busy,
          'aria-pressed': String(entry === state.shown),
          onclick: () => {
            state.shown = entry;
            renderResult();
            renderHistory();
            dom.result.scrollIntoView({ block: 'start', behavior: scrollBehavior() });
          },
        }, el('span', { class: 'hchip-v' }, s.text), el('span', { class: 'hchip-t' }, formatTime(entry.at)));
      })));
  }

  /** Keep the reference, clear «Проверка» and reopen its camera (runs inside the click: a user gesture). */
  function newCheck() {
    setPhoto('frame', null);
    openCamera('frame');
  }

  function newReference() {
    setPhoto('frame', null);
    setPhoto('reference', null);
    window.scrollTo({ top: 0, behavior: scrollBehavior() });
  }

  // ---------------------------------------------------------------------------
  // Camera (full-screen, rear). iOS Safari: playsinline + muted before srcObject, play() may need a tap.
  // ---------------------------------------------------------------------------

  const cam = { slot: null, stream: null, token: 0, opener: null };

  const cameraSupported = () => !!(window.isSecureContext && navigator.mediaDevices
    && typeof navigator.mediaDevices.getUserMedia === 'function');

  // Non-secure context (plain HTTP on a LAN): no getUserMedia, so «Камера» opens the system camera app.
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

  function onVideoReady() {
    const v = dom.camVideo;
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
  // Language toggle + boot
  // ---------------------------------------------------------------------------

  function syncLang() {
    const lang = I18N.getLang();
    document.querySelectorAll('[data-lang]').forEach((b) => {
      b.setAttribute('aria-pressed', String(b.getAttribute('data-lang') === lang));
    });
    I18N.applyStatic(document);
    if (cam.slot) dom.camTitle.textContent = t(`cam_title_${cam.slot}`);
    renderAll();
  }

  document.querySelectorAll('[data-lang]').forEach((b) => {
    b.addEventListener('click', () => {
      const lang = b.getAttribute('data-lang');
      if (lang === I18N.getLang()) return;
      I18N.setLang(lang);
      syncLang();
    });
  });

  dom.runBtn.addEventListener('click', runCompare);
  dom.insecureNote.hidden = cameraSupported();
  syncLang();
})();
