/*
 * Dalil QA — station web client (vanilla JS, no build step, no frameworks, no CDNs).
 * Talks to the HTTP API in docs/SPEC.md using relative "/api/..." URLs.
 *
 * Screens (hash routing):
 *   #/                    home
 *   #/new                 create a station (name, SKU)
 *   #/s/<sid>/setup       golden reference shots (1–5)
 *   #/s/<sid>/run         operator: live camera, "Check" / auto mode, verdict, QR for viewers
 *   #/s/<sid>/view        viewer (judge's phone): polls inspections, latest verdict, review buttons
 *   #/s/<sid>/journal     shift journal: stats, list, CSV, shift report
 *   #/compare             stateless quick compare of two photos (PBP vs plain subtraction)
 */
(function () {
  'use strict';

  const I18N = window.I18N;
  const t = I18N.t;

  // ---------------------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------------------

  const SID_RE = /^[A-Z2-9]{6}$/;
  const LS_SID = 'dalil.qa.sid';
  const LS_GHOST = 'dalil.qa.ghostOpacity';
  const LS_COMPARE = 'dalil.qa.compare';
  const LS_AUTO_EVERY = 'dalil.qa.autoEvery';

  const CAMERA_CONSTRAINTS = {
    audio: false,
    video: {
      facingMode: { ideal: 'environment' },
      width: { ideal: 1920 },
      height: { ideal: 1080 },
    },
  };
  const JPEG_QUALITY = 0.9;
  const UPLOAD_MAX_SIDE = 2560;            // gallery photos are downscaled to this before upload
  const UPLOAD_PASS_BYTES = 3 * 1024 * 1024; // smaller files are sent as-is
  const MAX_REFERENCES = 5;
  const AUTO_CHOICES = [2, 3, 5];          // seconds between automatic checks
  const POLL_MS = 1500;                    // viewer polling period
  const PAGE_LIMIT = 50;                   // inspections per request

  // ---------------------------------------------------------------------------
  // Small utilities
  // ---------------------------------------------------------------------------

  /** localStorage wrapper that never throws (private mode, blocked storage). */
  const store = {
    get(key) { try { return window.localStorage.getItem(key); } catch (e) { return null; } },
    set(key, value) { try { window.localStorage.setItem(key, value); } catch (e) { /* ignore */ } },
    remove(key) { try { window.localStorage.removeItem(key); } catch (e) { /* ignore */ } },
  };

  const enc = encodeURIComponent;

  function safeDecode(s) {
    try { return decodeURIComponent(s); } catch (e) { return s; }
  }

  /** Normalise a typed station code: upper case, drop spaces/dashes. */
  function normalizeSid(raw) {
    return String(raw || '').toUpperCase().replace(/[^A-Z0-9]/g, '');
  }

  /**
   * Create a DOM element. `props` keys: class, text, style (object), on<event> (function),
   * anything else becomes an attribute (true -> boolean attribute, null/false -> skipped).
   * Children may be nodes, strings, numbers, arrays, or null/false (skipped).
   */
  function el(tag, props, ...children) {
    const node = document.createElement(tag);
    if (props) {
      for (const key of Object.keys(props)) {
        const value = props[key];
        if (value == null || value === false) continue;
        if (key === 'class') node.className = value;
        else if (key === 'text') node.textContent = value;
        else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value);
        else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
        else node.setAttribute(key, value === true ? '' : String(value));
      }
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

  function svg(tag, attrs, ...children) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const key of Object.keys(attrs || {})) node.setAttribute(key, String(attrs[key]));
    for (const child of children) if (child) node.appendChild(child);
    return node;
  }

  /** Small stroke icons (24×24, currentColor). */
  const ICON_PATHS = {
    check: 'M5 12.5l4.5 4.5L19 7.5',
    cross: 'M6 6l12 12M18 6L6 18',
    alert: 'M12 8v5M12 16.5v.5M10.3 3.9L2.6 17.5A2 2 0 004.3 20.5h15.4a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z',
    retake: 'M20 11a8 8 0 10-2.3 5.7M20 4v7h-7',
    camera: 'M4 8h3l2-3h6l2 3h3a1 1 0 011 1v10a1 1 0 01-1 1H4a1 1 0 01-1-1V9a1 1 0 011-1zM12 17a4 4 0 100-8 4 4 0 000 8z',
    image: 'M4 5h16v14H4zM4 16l5-5 4 4 3-3 4 4M15.5 9.5h.01',
    doc: 'M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6',
    plus: 'M12 5v14M5 12h14',
    compare: 'M4 6h7v12H4zM13 6h7v12h-7M7.5 3v18',
    link: 'M10 14a4 4 0 005.7 0l3-3a4 4 0 00-5.7-5.7l-1 1M14 10a4 4 0 00-5.7 0l-3 3a4 4 0 005.7 5.7l1-1',
    eye: 'M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12zM12 15a3 3 0 100-6 3 3 0 000 6z',
    list: 'M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01',
    download: 'M12 4v11M7 10l5 5 5-5M5 20h14',
    target: 'M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5M9 9h6v6H9z',
    play: 'M8 5v14l11-7z',
    trash: 'M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13',
    copy: 'M9 9h10v10H9zM5 15V5h10',
    open: 'M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 01-1 1H5a1 1 0 01-1-1V7a1 1 0 011-1h5',
    refresh: 'M20 11a8 8 0 10-2.3 5.7M20 4v7h-7',
  };

  function icon(name, cls) {
    return svg('svg', {
      viewBox: '0 0 24 24', width: 24, height: 24, fill: 'none', stroke: 'currentColor',
      'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round',
      'aria-hidden': 'true', focusable: 'false', class: 'icon' + (cls ? ' ' + cls : ''),
    }, svg('path', { d: ICON_PATHS[name] || '' }));
  }

  function locale() {
    return I18N.getLang() === 'ru' ? 'ru-RU' : 'en-GB';
  }

  function parseDate(iso) {
    if (iso == null || iso === '') return null;
    const d = typeof iso === 'number' ? new Date(iso < 1e12 ? iso * 1000 : iso) : new Date(iso);
    return isNaN(d.getTime()) ? null : d;
  }

  function formatTime(iso) {
    const d = parseDate(iso);
    if (!d) return iso ? String(iso) : '';
    try {
      return d.toLocaleTimeString(locale(), { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch (e) {
      return d.toISOString().slice(11, 19);
    }
  }

  function formatDateTime(iso) {
    const d = parseDate(iso);
    if (!d) return iso ? String(iso) : '';
    try {
      return d.toLocaleString(locale(), { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      return d.toISOString().slice(0, 16).replace('T', ' ');
    }
  }

  /** Total processing time from a timing_ms dict (prefers an explicit "total"). */
  function totalMs(timing) {
    if (typeof timing === 'number') return timing;
    if (!timing || typeof timing !== 'object') return null;
    for (const key of ['total', 'total_ms', 'all']) {
      if (typeof timing[key] === 'number') return timing[key];
    }
    let sum = 0;
    let any = false;
    for (const key of Object.keys(timing)) {
      if (typeof timing[key] === 'number') { sum += timing[key]; any = true; }
    }
    return any ? sum : null;
  }

  function percent(frac) {
    if (typeof frac !== 'number' || !isFinite(frac)) return null;
    return Math.round(frac * 100);
  }

  /** Append a cache-busting query (outline/reference URLs are reused when the golden set changes). */
  function bust(url, stamp) {
    if (!url || stamp == null || stamp === '') return url;
    return url + (url.indexOf('?') >= 0 ? '&' : '?') + 'v=' + enc(String(stamp));
  }

  function shortHash(h) {
    return typeof h === 'string' && h.length > 16 ? `${h.slice(0, 8)}…${h.slice(-8)}` : (h || '');
  }

  function vibrate(pattern) {
    try { if (navigator.vibrate) navigator.vibrate(pattern); } catch (e) { /* unsupported */ }
  }

  function canvasToBlob(canvas, type, quality) {
    return new Promise((resolve, reject) => {
      if (canvas.toBlob) {
        canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('toBlob failed'))), type, quality);
      } else {
        try { resolve(dataUrlToBlob(canvas.toDataURL(type, quality))); } catch (e) { reject(e); }
      }
    });
  }

  function dataUrlToBlob(dataUrl) {
    const [head, body] = dataUrl.split(',');
    const mime = (head.match(/data:([^;]+)/) || [])[1] || 'application/octet-stream';
    const bin = atob(body);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Blob([bytes], { type: mime });
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('image decode failed'));
      img.src = src;
    });
  }

  /**
   * Shrink large gallery photos before upload (faster on mobile data, below the 15 MB limit).
   * Small files pass through untouched, so the server still sees their EXIF orientation.
   */
  async function prepareFile(file) {
    if (!file || file.size <= UPLOAD_PASS_BYTES) return file;
    const url = URL.createObjectURL(file);
    try {
      const img = await loadImage(url); // modern browsers apply EXIF orientation when drawing
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      const scale = Math.min(1, UPLOAD_MAX_SIDE / Math.max(w, h));
      const canvas = document.createElement('canvas');
      canvas.width = Math.round(w * scale);
      canvas.height = Math.round(h * scale);
      canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
      return await canvasToBlob(canvas, 'image/jpeg', JPEG_QUALITY);
    } catch (e) {
      return file; // let the server report the problem
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  // ---------------------------------------------------------------------------
  // HTTP API
  // ---------------------------------------------------------------------------

  class ApiError extends Error {
    constructor(message, status, code) {
      super(message);
      this.status = status;
      this.code = code || null;
    }
  }

  function errorMessage(data, res) {
    if (data && typeof data.error === 'string') {
      const key = `api_${data.error}`;
      const known = t(key);
      if (known !== key) return known; // localised text for a known server error code
    }
    if (data && typeof data.message === 'string' && data.message) return data.message;
    if (data && typeof data.detail === 'string') return data.detail;
    if (data && Array.isArray(data.detail)) {
      const msg = data.detail.map((d) => (d && d.msg) || '').filter(Boolean).join('; ');
      if (msg) return msg;
    }
    if (res.status === 413) return t('err_too_large');
    if (res.status === 404) return t('err_not_found');
    return t('err_http', { status: res.status });
  }

  /**
   * Fetch JSON from the API. Options: method, json (object body), form (FormData body).
   * Throws ApiError carrying the server's {"error","message"} on non-2xx responses.
   */
  async function api(path, options) {
    const opts = options || {};
    const init = { method: opts.method || 'GET', headers: { Accept: 'application/json' }, credentials: 'same-origin', cache: 'no-store' };
    if (opts.json !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(opts.json);
    } else if (opts.form) {
      init.body = opts.form;
    }
    let res;
    try {
      res = await fetch(path, init);
    } catch (e) {
      throw new ApiError(t('err_network'), 0, 'network');
    }
    let data = null;
    const type = res.headers.get('content-type') || '';
    if (type.indexOf('json') >= 0) {
      try { data = await res.json(); } catch (e) { data = null; }
    }
    if (!res.ok) throw new ApiError(errorMessage(data, res), res.status, data && data.error);
    return data;
  }

  /** Every endpoint the client uses (see docs/SPEC.md "HTTP API"). */
  const paths = {
    health: () => '/api/health',
    stations: () => '/api/stations',
    station: (sid) => `/api/stations/${enc(sid)}`,
    reference: (sid) => `/api/stations/${enc(sid)}/reference`,
    inspect: (sid, compare) => `/api/stations/${enc(sid)}/inspect` + (compare ? '?compare=1' : ''),
    inspections: (sid, after, limit) => `/api/stations/${enc(sid)}/inspections?after=${enc(after)}&limit=${enc(limit)}`,
    review: (sid, seq) => `/api/stations/${enc(sid)}/inspections/${enc(seq)}/review`,
    journalCsv: (sid) => `/api/stations/${enc(sid)}/journal.csv`,
    report: (sid) => `/api/stations/${enc(sid)}/report`,
    compare: () => '/api/compare?methods=pbp,pixdiff',
  };

  const getStation = (sid) => api(paths.station(sid));

  /** Load every inspection of a station (pages of PAGE_LIMIT, oldest first). */
  async function loadAllInspections(sid, ctx) {
    const all = [];
    let after = 0;
    for (let page = 0; page < 40; page++) {
      const data = await api(paths.inspections(sid, after, PAGE_LIMIT));
      if (ctx && !ctx.alive()) break;
      const items = (data && Array.isArray(data.items)) ? data.items : [];
      all.push(...items);
      if (items.length < PAGE_LIMIT) break;
      after = items[items.length - 1].seq;
    }
    return all;
  }

  // ---------------------------------------------------------------------------
  // Chrome: header, toasts, busy overlay
  // ---------------------------------------------------------------------------

  const $app = document.getElementById('app');
  const $back = document.getElementById('backBtn');
  const $brand = document.getElementById('brand');
  const $title = document.getElementById('screenTitle');
  const $toasts = document.getElementById('toasts');
  const $busy = document.getElementById('busy');
  const $busyText = document.getElementById('busyText');

  let backTarget = null;

  function setChrome(opts) {
    const o = opts || {};
    backTarget = o.back == null ? null : o.back;
    $back.hidden = backTarget === null;
    $brand.hidden = backTarget !== null;
    $title.textContent = o.title || '';
    document.title = o.title ? `${o.title} · Dalil QA` : 'Dalil QA';
  }

  $back.addEventListener('click', () => { if (backTarget !== null) go(backTarget); });

  function mount(...nodes) {
    clearNode($app);
    appendChildren($app, nodes);
    window.scrollTo(0, 0);
  }

  /** Show a toast; an identical message already on screen is not repeated (auto mode errors). */
  function toast(message, kind) {
    const k = kind || 'error';
    for (const node of Array.from($toasts.children)) {
      if (node.dataset.msg === message) return;
    }
    const node = el('div', { class: `toast toast-${k}`, role: k === 'error' ? 'alert' : 'status' }, message);
    node.dataset.msg = message;
    const remove = () => { if (node.parentNode) node.parentNode.removeChild(node); };
    node.addEventListener('click', remove);
    $toasts.appendChild(node);
    while ($toasts.children.length > 3) $toasts.removeChild($toasts.firstChild);
    setTimeout(remove, k === 'error' ? 6000 : 3000);
  }

  function showError(err) {
    console.error(err);
    toast((err && err.message) || t('err_generic'), 'error');
  }

  let busyCount = 0;

  function showBusy(message) {
    busyCount += 1;
    $busyText.textContent = message || t('busy_default');
    $busy.hidden = false;
    document.body.setAttribute('aria-busy', 'true');
  }

  function hideBusy(reset) {
    busyCount = reset ? 0 : Math.max(0, busyCount - 1);
    if (busyCount === 0) {
      $busy.hidden = true;
      document.body.removeAttribute('aria-busy');
    }
  }

  /** Run `fn` with the spinner overlay and the given buttons disabled. */
  async function withBusy(message, buttons, fn) {
    const btns = (buttons || []).filter(Boolean);
    btns.forEach((b) => { b.disabled = true; });
    showBusy(message);
    try {
      return await fn();
    } finally {
      hideBusy();
      btns.forEach((b) => { b.disabled = false; });
    }
  }

  // ---------------------------------------------------------------------------
  // Router
  // ---------------------------------------------------------------------------

  let routeSeq = 0;
  let leaveFns = [];

  /** Register cleanup to run when the current screen is left (camera, timers, listeners). */
  function onLeave(fn) {
    leaveFns.push(fn);
  }

  function leaveScreen() {
    const fns = leaveFns;
    leaveFns = [];
    for (const fn of fns) {
      try { fn(); } catch (e) { console.error(e); }
    }
  }

  function go(path) {
    const hash = '#/' + path;
    if (location.hash === hash) router();
    else location.hash = hash;
  }

  function currentParts() {
    return (location.hash || '').replace(/^#\/?/, '').split('/').filter(Boolean).map(safeDecode);
  }

  const STATION_SCREENS = {
    setup: renderSetup,
    run: renderRun,
    view: renderViewer,
    journal: renderJournal,
  };

  async function router() {
    leaveScreen();
    hideBusy(true);
    const seq = ++routeSeq;
    const ctx = { alive: () => seq === routeSeq };
    const parts = currentParts();
    try {
      if (parts.length === 0) return await renderHome(ctx);
      if (parts[0] === 'new' && parts.length === 1) return renderNew(ctx);
      if (parts[0] === 'compare' && parts.length === 1) return renderCompare(ctx);
      if (parts[0] === 's' && parts[1]) {
        const sid = normalizeSid(parts[1]);
        if (!SID_RE.test(sid)) {
          toast(t('err_bad_code'), 'error');
          return go('');
        }
        const screen = STATION_SCREENS[parts[2] || 'run'];
        if (screen && parts.length <= 3) return await screen(ctx, sid);
      }
      go('');
    } catch (err) {
      if (!ctx.alive()) return;
      hideBusy(true);
      if (err instanceof ApiError && err.status === 404 && parts[0] === 's' && store.get(LS_SID) === normalizeSid(parts[1])) {
        store.remove(LS_SID);
      }
      showError(err);
      renderErrorScreen(err);
    }
  }

  function renderErrorScreen(err) {
    setChrome({ title: t('err_title'), back: '' });
    mount(el('section', { class: 'card stack center narrow' },
      icon('alert', 'icon-xl muted'),
      el('p', { class: 'lead' }, (err && err.message) || t('err_generic')),
      el('button', { class: 'btn btn-primary btn-lg', type: 'button', onclick: () => router() }, icon('refresh'), t('retry')),
      el('a', { class: 'btn btn-ghost', href: '#/' }, t('to_home'))));
  }

  // ---------------------------------------------------------------------------
  // Shared components
  // ---------------------------------------------------------------------------

  /** Station code split in two groups, monospace, large. */
  function stationCode(sid, cls) {
    return el('div', { class: 'code' + (cls ? ' ' + cls : ''), 'aria-label': `${t('station_code')}: ${sid.split('').join(' ')}` },
      el('span', null, sid.slice(0, 3)), el('span', null, sid.slice(3)));
  }

  /** Tabs between the station screens. */
  function stationNav(sid, active) {
    const items = [
      ['setup', 'target', t('nav_setup')],
      ['run', 'camera', t('nav_run')],
      ['view', 'eye', t('nav_view')],
      ['journal', 'list', t('nav_journal')],
    ];
    return el('nav', { class: 'station-nav', 'aria-label': t('station') },
      items.map(([key, ic, label]) => el('a', {
        class: 'nav-item', href: `#/s/${sid}/${key}`, 'aria-current': key === active ? 'page' : null,
      }, icon(ic, 'icon-sm'), el('span', null, label))));
  }

  function stationHeader(station) {
    return el('div', { class: 'station-head' },
      el('div', { class: 'station-id' },
        el('div', { class: 'station-name' }, station.name || t('station')),
        station.sku ? el('div', { class: 'station-sku' }, `${t('sku')}: ${station.sku}`) : null),
      stationCode(station.id, 'code-sm'));
  }

  function viewerUrl(sid) {
    return `${location.origin}${location.pathname}#/s/${sid}/view`;
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(() => toast(t('copied'), 'ok'), () => toast(t('copy_failed'), 'error'));
      return;
    }
    try {
      const ta = el('textarea', { readonly: true, class: 'visually-hidden' });
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      toast(ok ? t('copied') : t('copy_failed'), ok ? 'ok' : 'error');
    } catch (e) {
      toast(t('copy_failed'), 'error');
    }
  }

  /** QR code (viewer link) + the big station code. */
  function qrCard(sid) {
    const url = viewerUrl(sid);
    const box = el('div', { class: 'qr-box' });
    const svgNode = window.DalilQR ? window.DalilQR.render(box, url, { ecc: 'M', label: t('qr_label') }) : null;
    if (!svgNode) box.appendChild(el('p', { class: 'muted small' }, t('qr_unavailable')));
    return el('section', { class: 'card qr-card' },
      box,
      el('div', { class: 'qr-text' },
        el('div', { class: 'label-caps' }, t('station_code')),
        stationCode(sid),
        el('p', { class: 'help' }, t('qr_help')),
        el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: () => copyText(url) }, icon('copy', 'icon-sm'), t('copy_link'))));
  }

  /** Russian/English plural of "zone". */
  function zonesWord(n) {
    return I18N.plural(n, 'zone_forms');
  }

  function verdictClass(verdict) {
    return verdict === 'pass' || verdict === 'fail' || verdict === 'retake' ? verdict : 'retake';
  }

  function zoneCount(rec) {
    if (Array.isArray(rec.zones)) return rec.zones.length;
    if (typeof rec.n_zones === 'number') return rec.n_zones;
    return 0;
  }

  function reasonText(reason) {
    const key = `rr_${reason}`;
    const s = t(key);
    return s === key ? t('rr_unknown') : s;
  }

  /** Extra advice under a RETAKE verdict (glare → change the angle, etc.). */
  function retakeHint(rec) {
    const hints = Array.isArray(rec.hints) ? rec.hints : [];
    if (rec.reason === 'glare' || hints.indexOf('change_angle') >= 0) return t('hint_change_angle');
    const key = `rh_${rec.reason}`;
    const s = t(key);
    return s === key ? null : s;
  }

  /** Short verdict text for chips: "ГОДНА" / "БРАК · 2" / "ПЕРЕСНЯТЬ". */
  function verdictShort(rec) {
    const v = verdictClass(rec.verdict);
    if (v === 'pass') return t('v_pass');
    if (v === 'fail') return `${t('v_fail_short')} · ${zoneCount(rec)}`;
    return t('v_retake_short');
  }

  function verdictChip(rec, prefix) {
    const v = verdictClass(rec.verdict);
    return el('span', { class: `chip chip-${v}` }, prefix ? `${prefix}: ` : null, verdictShort(rec));
  }

  /** The large verdict banner: ГОДНА / БРАК — N зон / ПЕРЕСНЯТЬ: reason. */
  function verdictBanner(rec, big) {
    const v = verdictClass(rec.verdict);
    let title;
    let sub = null;
    let ic;
    if (v === 'pass') {
      title = t('v_pass');
      ic = 'check';
    } else if (v === 'fail') {
      const n = zoneCount(rec);
      title = `${t('v_fail')} — ${n} ${zonesWord(n)}`;
      if (rec.reason === 'mismatch') sub = reasonText('mismatch');
      ic = 'cross';
    } else {
      title = `${t('v_retake')}: ${reasonText(rec.reason)}`;
      sub = rec.reason === 'mismatch' ? null : retakeHint(rec);
      ic = 'retake';
    }
    return el('div', { class: `verdict verdict-${v}${big ? ' verdict-big' : ''}`, role: v === 'pass' ? 'status' : 'alert' },
      icon(ic, 'verdict-icon'),
      el('div', { class: 'verdict-text' },
        el('strong', { class: 'verdict-title' }, title),
        sub ? el('span', { class: 'verdict-sub' }, sub) : null));
  }

  /** Small line under the verdict: #seq · time · coverage · ms. */
  function metaLine(rec, extra) {
    const parts = [];
    if (rec.seq != null) parts.push(`#${rec.seq}`);
    if (rec.at) parts.push(formatTime(rec.at));
    const cov = percent(rec.coverage);
    if (cov != null) parts.push(t('coverage_pct', { n: cov }));
    const ms = totalMs(rec.timing_ms);
    if (ms != null) parts.push(t('ms', { n: Math.round(ms) }));
    if (rec.align && rec.align.method) parts.push(t('align_method', { m: rec.align.method }));
    if (extra) parts.push(extra);
    return el('p', { class: 'meta' }, parts.join(' · '));
  }

  /** Content hints of the zones (water/dirt, removed, added) as chips. */
  function zoneHints(rec) {
    const zones = Array.isArray(rec.zones) ? rec.zones : [];
    const seen = [];
    zones.forEach((z) => { if (z && z.hint && seen.indexOf(z.hint) < 0) seen.push(z.hint); });
    if (!seen.length) return null;
    return el('div', { class: 'hint-chips' }, seen.map((h) => {
      const key = `zh_${h}`;
      const s = t(key);
      return el('span', { class: 'chip chip-hint' }, icon('alert', 'icon-sm'), s === key ? h : s);
    }));
  }

  /** Image with a placeholder when it fails to load. */
  function safeImg(src, alt, cls) {
    const img = el('img', { class: cls || 'figure-img', alt: alt || '', decoding: 'async', loading: 'lazy' });
    const missing = el('div', { class: 'img-missing', hidden: true }, icon('image', 'icon-lg'), el('span', null, t('image_missing')));
    img.addEventListener('error', () => { img.hidden = true; missing.hidden = false; });
    if (src) img.src = src; else { img.hidden = true; missing.hidden = false; }
    return [img, missing];
  }

  /** Overlay (zones) / aligned frame switcher for one inspection. */
  function resultFigure(rec) {
    const images = [];
    if (rec.overlay_url) images.push({ key: 'overlay', label: t('tab_zones'), src: rec.overlay_url });
    if (rec.frame_url) images.push({ key: 'frame', label: t('tab_frame'), src: rec.frame_url });
    if (!images.length) return null;
    const wrap = el('div', { class: 'figure-wrap' });
    const tabs = {};
    function show(key) {
      const im = images.find((x) => x.key === key) || images[0];
      Object.keys(tabs).forEach((k) => tabs[k].setAttribute('aria-selected', String(k === im.key)));
      replaceContent(wrap, el('a', { href: im.src, target: '_blank', rel: 'noopener', class: 'figure-link' }, safeImg(im.src, im.label)));
    }
    const tabBar = images.length > 1
      ? el('div', { class: 'seg', role: 'tablist' }, images.map((im) => {
        tabs[im.key] = el('button', { class: 'seg-btn', type: 'button', role: 'tab', onclick: () => show(im.key) }, im.label);
        return tabs[im.key];
      }))
      : null;
    show(images[0].key);
    return el('figure', { class: 'figure' }, tabBar, wrap);
  }

  /**
   * Side-by-side: Dalil (PBP) vs plain reference subtraction on the same aligned frame.
   * `left`/`right`: {rec (verdict + zones/n_zones), src}.
   */
  function compareGrid(left, right) {
    const col = (title, sub, side) => el('div', { class: 'cmp-col' },
      el('div', { class: 'cmp-head' },
        el('div', null, el('div', { class: 'cmp-title' }, title), sub ? el('div', { class: 'cmp-sub' }, sub) : null),
        verdictChip(side.rec)),
      el('a', { href: side.src || '#', target: '_blank', rel: 'noopener', class: 'figure-link' },
        safeImg(side.src, title, 'cmp-img')));
    const differ = verdictClass(left.rec.verdict) !== verdictClass(right.rec.verdict);
    return el('div', { class: 'cmp' },
      el('div', { class: 'cmp-grid' },
        col(t('cmp_dalil'), t('cmp_dalil_sub'), left),
        col(t('cmp_baseline'), t('cmp_baseline_sub'), right)),
      differ ? el('p', { class: 'cmp-note' }, icon('alert', 'icon-sm'), t('cmp_differ')) : null);
  }

  /** Baseline comparison for a stored inspection record (needs record.baseline). */
  function recordCompare(rec) {
    const b = rec.baseline;
    if (!b) return el('p', { class: 'muted small' }, t('cmp_none'));
    return compareGrid(
      { rec, src: rec.overlay_url },
      { rec: { verdict: b.verdict, n_zones: typeof b.n_zones === 'number' ? b.n_zones : 0 }, src: b.overlay_url });
  }

  /** Review label (operator confirmation) as a small chip. */
  function reviewChip(label) {
    if (!label) return null;
    const key = `rv_${label}`;
    const s = t(key);
    return el('span', { class: `chip chip-review chip-review-${label}` }, icon('check', 'icon-sm'), s === key ? label : s);
  }

  /** Review buttons: fail → defect / false alarm; pass → ok / missed defect; retake → none. */
  function reviewBar(sid, rec, onUpdate, compact) {
    const v = verdictClass(rec.verdict);
    let options;
    if (v === 'fail') options = [['defect', 'review_defect'], ['false_alarm', 'review_false_alarm']];
    else if (v === 'pass') options = [['ok', 'review_ok'], ['defect', 'review_missed']];
    else return null;
    const buttons = options.map(([label, key]) => el('button', {
      class: `btn review-btn review-${label}${compact ? ' btn-sm' : ''}`, type: 'button',
      'aria-pressed': String(rec.review === label),
      onclick: async () => {
        if (rec.review === label) return;
        buttons.forEach((b) => { b.disabled = true; });
        try {
          const updated = await api(paths.review(sid, rec.seq), { method: 'POST', json: { label } });
          toast(t('review_saved'), 'ok');
          onUpdate(Object.assign({}, rec, updated || {}, { review: (updated && updated.review) || label }));
        } catch (err) {
          showError(err);
          buttons.forEach((b) => { b.disabled = false; });
        }
      },
    }, t(key)));
    return el('div', { class: 'review-bar' + (compact ? ' review-bar-compact' : '') }, buttons);
  }

  /** Stats tiles from station.stats. */
  function statsTiles(stats) {
    const s = stats || {};
    const tile = (key, cls, value) => el('div', { class: `tile${cls ? ' tile-' + cls : ''}` },
      el('div', { class: 'tile-num' }, String(value == null ? 0 : value)),
      el('div', { class: 'tile-label' }, t(key)));
    return el('div', { class: 'tiles' },
      tile('st_total', null, s.total),
      tile('st_pass', 'pass', s.pass),
      tile('st_fail', 'fail', s.fail),
      tile('st_retake', 'retake', s.retake),
      tile('st_confirmed', null, s.confirmed_defects),
      tile('st_false', null, s.false_alarms));
  }

  /** One row of the journal list. */
  function recordRow(sid, rec, onUpdate) {
    const v = verdictClass(rec.verdict);
    const thumbSrc = v === 'fail' ? (rec.overlay_url || rec.frame_url) : (rec.frame_url || rec.overlay_url);
    let detail;
    if (v === 'retake') detail = reasonText(rec.reason);
    else if (v === 'fail') detail = `${zoneCount(rec)} ${zonesWord(zoneCount(rec))}`;
    else detail = t('v_pass_detail');
    const cov = percent(rec.coverage);
    const ms = totalMs(rec.timing_ms);
    return el('li', { class: `rec rec-${v}` },
      el('a', { class: 'rec-thumb', href: thumbSrc || '#', target: '_blank', rel: 'noopener', 'aria-label': t('open_image') },
        safeImg(thumbSrc, '', 'rec-img')),
      el('div', { class: 'rec-body' },
        el('div', { class: 'rec-top' },
          el('span', { class: 'rec-seq' }, `#${rec.seq}`),
          verdictChip(rec),
          el('span', { class: 'rec-time' }, formatTime(rec.at))),
        el('div', { class: 'rec-sub' },
          detail,
          cov != null ? ` · ${t('coverage_pct', { n: cov })}` : '',
          ms != null ? ` · ${t('ms', { n: Math.round(ms) })}` : ''),
        el('div', { class: 'rec-chips' },
          rec.baseline ? verdictChip({ verdict: rec.baseline.verdict, n_zones: rec.baseline.n_zones }, t('baseline_short')) : null,
          reviewChip(rec.review)),
        reviewBar(sid, rec, onUpdate, true)));
  }

  // ---------------------------------------------------------------------------
  // Camera
  // ---------------------------------------------------------------------------

  function stopTracks(stream) {
    if (!stream) return;
    stream.getTracks().forEach((track) => { try { track.stop(); } catch (e) { /* ignore */ } });
  }

  /**
   * Live rear camera bound to a <video>.
   * handlers: onReady(), onFail(reason: insecure|denied|unavailable), onNeedTap() when iOS blocks play().
   */
  function createCamera(video, handlers) {
    let stream = null;
    let active = false;

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

    async function play() {
      try {
        await video.play();
        return true;
      } catch (e) {
        return false;
      }
    }

    async function start() {
      if (active) return;
      active = true;
      if (!window.isSecureContext || !navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
        active = false;
        handlers.onFail('insecure');
        return;
      }
      try {
        const s = await getStream();
        if (!active) { stopTracks(s); return; } // screen left while the permission prompt was open
        stream = s;
        // iOS Safari: these must be set before attaching the stream, or it goes fullscreen / stays black.
        video.setAttribute('playsinline', '');
        video.setAttribute('webkit-playsinline', '');
        video.setAttribute('muted', '');
        video.muted = true;
        video.playsInline = true;
        video.srcObject = s;
        const ready = () => { if (active && video.videoWidth) handlers.onReady(); };
        video.addEventListener('loadedmetadata', ready, { once: true });
        video.addEventListener('playing', ready, { once: true });
        if (!(await play()) && active) handlers.onNeedTap(); // autoplay blocked → needs a user gesture
        ready();
      } catch (e) {
        active = false;
        handlers.onFail(e && (e.name === 'NotAllowedError' || e.name === 'SecurityError') ? 'denied' : 'unavailable');
      }
    }

    function stop() {
      active = false;
      stopTracks(stream);
      stream = null;
      try { video.pause(); } catch (e) { /* ignore */ }
      video.srcObject = null;
    }

    return {
      start,
      stop,
      play, // call from a click handler (user gesture) when onNeedTap fired
      isActive: () => active,
      isReady: () => active && !!video.videoWidth && video.readyState >= 2,
    };
  }

  /** Grab the full-resolution current video frame as a JPEG blob. */
  function grabFrame(video) {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return Promise.reject(new Error(t('err_no_frame')));
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    canvas.getContext('2d').drawImage(video, 0, 0, w, h);
    return canvasToBlob(canvas, 'image/jpeg', JPEG_QUALITY);
  }

  /** A <label> styled as a button that opens a file picker (more reliable on iOS than input.click()). */
  function fileButton(text, iconName, cls, inputAttrs, onFile) {
    const input = el('input', Object.assign({ type: 'file', accept: 'image/*', class: 'visually-hidden' }, inputAttrs || {}));
    input.addEventListener('change', () => {
      const file = input.files && input.files[0];
      input.value = '';
      if (file) onFile(file);
    });
    return el('label', { class: `btn ${cls}` }, input, icon(iconName), el('span', null, text));
  }

  /**
   * Camera stage: <video> + optional reference outline + status message, wired to the camera
   * lifecycle (stops on leave and while the tab is hidden). Returns { node, video, camera, flash, setOutline, failed }.
   */
  function cameraStage(ctx, opts) {
    const o = opts || {};
    const video = el('video', { class: 'stage-media', playsinline: true, 'webkit-playsinline': true, muted: true, autoplay: true });
    video.muted = true;
    const msg = el('div', { class: 'stage-msg' }, el('div', { class: 'spinner spinner-light' }), el('span', null, t('camera_starting')));
    const flash = el('div', { class: 'stage-flash' });
    const tapBtn = el('button', { class: 'btn btn-accent stage-tap', type: 'button', hidden: true }, icon('play'), t('camera_tap'));
    const ghost = el('img', { class: 'stage-media ghost', alt: '', draggable: 'false', hidden: true });
    const node = el('div', { class: 'stage' }, video, ghost, msg, tapBtn, flash);
    const state = { failed: false };

    ghost.addEventListener('error', () => { ghost.hidden = true; });
    ghost.addEventListener('load', () => { ghost.hidden = state.failed; });

    const camera = createCamera(video, {
      onReady() {
        msg.hidden = true;
        tapBtn.hidden = true;
        if (o.onReady) o.onReady();
      },
      onNeedTap() {
        msg.hidden = true;
        tapBtn.hidden = false;
      },
      onFail(reason) {
        state.failed = true;
        node.classList.add('no-camera');
        video.hidden = true;
        ghost.hidden = true;
        tapBtn.hidden = true;
        replaceContent(msg, icon('camera', 'icon-xl'), el('span', null, t(`cam_fail_${reason}`)), el('span', { class: 'small' }, t('cam_fallback')));
        msg.hidden = false;
        if (o.onFail) o.onFail(reason);
      },
    });
    tapBtn.addEventListener('click', async () => {
      if (await camera.play()) { tapBtn.hidden = true; if (video.videoWidth && o.onReady) o.onReady(); }
    });

    camera.start();
    onLeave(camera.stop);
    const onVisibility = () => {
      if (document.hidden) camera.stop();
      else if (ctx.alive() && !state.failed) camera.start();
    };
    document.addEventListener('visibilitychange', onVisibility);
    onLeave(() => document.removeEventListener('visibilitychange', onVisibility));

    return {
      node,
      video,
      camera,
      flash() {
        flash.classList.remove('on');
        void flash.offsetWidth; // restart the CSS animation
        flash.classList.add('on');
      },
      setOutline(url) {
        if (!url || state.failed) { ghost.hidden = true; ghost.removeAttribute('src'); return; }
        if (ghost.getAttribute('src') !== url) ghost.src = url;
      },
      ghost,
      failed: () => state.failed,
    };
  }

  /** Outline opacity slider bound to the stage ghost (persisted per device). */
  function opacitySlider(stage) {
    const saved = parseFloat(store.get(LS_GHOST));
    const opacity = isFinite(saved) ? Math.min(1, Math.max(0, saved)) : 0.5;
    stage.ghost.style.opacity = String(opacity);
    const id = `ghost${Date.now()}`;
    const slider = el('input', { type: 'range', min: 0, max: 100, step: 5, value: Math.round(opacity * 100), id });
    const value = el('span', { class: 'range-value' }, `${Math.round(opacity * 100)}%`);
    slider.addEventListener('input', () => {
      const v = Number(slider.value) / 100;
      stage.ghost.style.opacity = String(v);
      value.textContent = `${slider.value}%`;
      store.set(LS_GHOST, String(v));
    });
    return el('div', { class: 'range-row' }, el('label', { for: id }, t('outline_opacity')), slider, value);
  }

  function outlineUrl(station) {
    const refs = station.references || [];
    const last = refs.length ? refs[refs.length - 1].at : '';
    return station.outline_url ? bust(station.outline_url, `${refs.length}-${last}`) : null;
  }

  // ---------------------------------------------------------------------------
  // Screen: Home
  // ---------------------------------------------------------------------------

  async function renderHome(ctx) {
    setChrome({ title: '' });
    const savedSid = store.get(LS_SID);

    const codeInput = el('input', {
      id: 'joinCode', type: 'text', inputmode: 'text', autocomplete: 'off', autocapitalize: 'characters',
      spellcheck: 'false', maxlength: 8, placeholder: 'ABC234', class: 'input-code', enterkeyhint: 'go',
      'aria-label': t('station_code'),
    });
    codeInput.addEventListener('input', () => {
      const v = normalizeSid(codeInput.value).slice(0, 6);
      if (codeInput.value !== v) codeInput.value = v;
    });
    const joinBtn = el('button', { class: 'btn btn-primary', type: 'submit' }, t('join'));
    const joinForm = el('form', { class: 'join', novalidate: true },
      el('label', { for: 'joinCode', class: 'label-caps' }, t('join_title')),
      el('div', { class: 'join-row' }, codeInput, joinBtn));
    joinForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const sid = normalizeSid(codeInput.value);
      if (!SID_RE.test(sid)) {
        toast(t('err_bad_code'), 'error');
        codeInput.focus();
        return;
      }
      try {
        await withBusy(t('busy_connecting'), [joinBtn], () => getStation(sid));
        if (ctx.alive()) go(`s/${sid}/view`);
      } catch (err) {
        showError(err);
      }
    });

    const health = el('p', { class: 'footer-note' });

    mount(
      el('section', { class: 'hero' },
        el('h2', { class: 'wordmark' }, 'Dalil', el('span', { class: 'wordmark-tag' }, 'QA')),
        el('p', { class: 'tagline' }, t('tagline'))),
      el('div', { class: 'stack home-actions' },
        el('a', { class: 'btn btn-accent btn-xl', href: '#/new' }, icon('plus'), t('new_station')),
        savedSid && SID_RE.test(savedSid)
          ? el('a', { class: 'btn btn-secondary btn-lg', href: `#/s/${savedSid}/run` }, icon('camera'), t('continue_station', { code: savedSid }))
          : null,
        el('section', { class: 'card' }, joinForm),
        el('a', { class: 'btn btn-secondary btn-lg', href: '#/compare' }, icon('compare'), t('quick_compare'))),
      el('ol', { class: 'steps' },
        el('li', null, el('strong', null, t('step1_title')), el('span', null, t('step1_text'))),
        el('li', null, el('strong', null, t('step2_title')), el('span', null, t('step2_text'))),
        el('li', null, el('strong', null, t('step3_title')), el('span', null, t('step3_text')))),
      el('p', { class: 'privacy' }, t('privacy_note')),
      health);

    try {
      const h = await api(paths.health());
      if (ctx.alive() && h) health.textContent = t('server_ok', { v: h.version || '?' });
    } catch (e) {
      if (ctx.alive()) health.textContent = t('server_down');
    }
  }

  // ---------------------------------------------------------------------------
  // Screen: New station
  // ---------------------------------------------------------------------------

  function renderNew(ctx) {
    setChrome({ title: t('new_station'), back: '' });
    const nameInput = el('input', {
      id: 'stName', type: 'text', autocomplete: 'off', maxlength: 80, required: true,
      placeholder: t('name_placeholder'), enterkeyhint: 'next',
    });
    const skuInput = el('input', {
      id: 'stSku', type: 'text', autocomplete: 'off', maxlength: 64,
      placeholder: t('sku_placeholder'), enterkeyhint: 'go',
    });
    const submit = el('button', { class: 'btn btn-accent btn-lg', type: 'submit' }, t('create_station'));
    const form = el('form', { class: 'card stack narrow', novalidate: true },
      el('div', { class: 'field' }, el('label', { for: 'stName' }, t('station_name')), nameInput),
      el('div', { class: 'field' },
        el('label', { for: 'stSku' }, t('sku'), el('span', { class: 'optional' }, t('optional'))),
        skuInput,
        el('p', { class: 'help' }, t('sku_help'))),
      submit);

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const name = nameInput.value.trim();
      if (!name) {
        toast(t('err_name_required'), 'error');
        nameInput.focus();
        return;
      }
      const body = { name };
      const sku = skuInput.value.trim();
      if (sku) body.sku = sku;
      try {
        const station = await withBusy(t('busy_creating'), [submit], () => api(paths.stations(), { method: 'POST', json: body }));
        if (!ctx.alive()) return;
        store.set(LS_SID, station.id);
        go(`s/${station.id}/setup`);
      } catch (err) {
        showError(err);
      }
    });

    mount(form);
    nameInput.focus();
  }

  // ---------------------------------------------------------------------------
  // Screen: Station setup (golden reference)
  // ---------------------------------------------------------------------------

  async function renderSetup(ctx, sid) {
    let station = await getStation(sid);
    if (!ctx.alive()) return;
    store.set(LS_SID, sid);
    setChrome({ title: t('setup_title'), back: '' });

    const shootBtn = el('button', { class: 'btn btn-accent btn-xl', type: 'button', disabled: true }, icon('camera'), t('shoot_reference'));
    const sysCamera = fileButton(t('use_system_camera'), 'camera', 'btn-secondary', { capture: 'environment' }, onFile);
    const gallery = fileButton(t('from_gallery'), 'image', 'btn-ghost', null, onFile);
    const countLine = el('div', { class: 'ref-count' });
    const thumbs = el('div', { class: 'ref-thumbs' });
    const warnBox = el('div');
    const clearBtn = el('button', { class: 'btn btn-ghost btn-danger', type: 'button' }, icon('trash'), t('clear_reference'));
    const startBtn = el('a', { class: 'btn btn-primary btn-lg', href: `#/s/${sid}/run` }, icon('play'), t('start_control'));
    const stage = cameraStage(ctx, { onReady: () => syncButtons() });
    const slider = opacitySlider(stage);

    function syncButtons() {
      const n = (station.references || []).length;
      const full = n >= MAX_REFERENCES;
      shootBtn.disabled = full || !stage.camera.isReady();
      [sysCamera, gallery].forEach((b) => b.classList.toggle('is-disabled', full));
      [sysCamera, gallery].forEach((b) => { b.querySelector('input').disabled = full; });
      clearBtn.hidden = n === 0;
      startBtn.classList.toggle('is-disabled', n === 0);
      startBtn.setAttribute('aria-disabled', String(n === 0));
    }

    function renderRefs() {
      const refs = station.references || [];
      replaceContent(countLine,
        el('strong', null, t('ref_count', { n: refs.length, max: MAX_REFERENCES })),
        refs.length ? null : el('span', { class: 'muted' }, ` — ${t('ref_none')}`));
      replaceContent(thumbs, refs.map((r, i) => el('a', { class: 'ref-thumb', href: r.url, target: '_blank', rel: 'noopener' },
        safeImg(bust(r.url, r.at), t('ref_alt', { n: i + 1 }), 'ref-img'),
        el('span', { class: 'ref-idx' }, String(i + 1)))));
      stage.setOutline(outlineUrl(station));
      syncButtons();
    }

    function renderCaptureReport(report) {
      clearNode(warnBox);
      const warnings = (report && Array.isArray(report.warnings)) ? report.warnings : [];
      if (!report) return;
      if (!warnings.length) {
        warnBox.appendChild(el('div', { class: 'notice notice-ok', role: 'status' }, icon('check'), t('ref_ok')));
        return;
      }
      warnBox.appendChild(el('div', { class: 'notice notice-warn', role: 'alert' },
        icon('alert'),
        el('div', null,
          el('strong', null, t('ref_warn_title')),
          el('ul', { class: 'warn-list' }, warnings.map((w) => el('li', null, t(`warn_${w}`)))),
          el('span', { class: 'small' }, t('ref_warn_hint')))));
    }

    async function upload(blob, filename) {
      const form = new FormData();
      form.append('file', blob, filename);
      try {
        const data = await withBusy(t('busy_reference'), [shootBtn], () => api(paths.reference(sid), { method: 'POST', form }));
        if (!ctx.alive()) return;
        station = data;
        renderRefs();
        renderCaptureReport(data.last_capture || null);
      } catch (err) {
        showError(err);
      }
      syncButtons();
    }

    shootBtn.addEventListener('click', async () => {
      try {
        const blob = await grabFrame(stage.video);
        stage.flash();
        upload(blob, `reference_${Date.now()}.jpg`);
      } catch (err) {
        showError(err);
      }
    });

    async function onFile(file) {
      upload(await prepareFile(file), file.name || 'reference.jpg');
    }

    clearBtn.addEventListener('click', async () => {
      if (!window.confirm(t('clear_confirm'))) return;
      try {
        station = await withBusy(t('busy_default'), [clearBtn], () => api(paths.reference(sid), { method: 'DELETE' }));
        if (!ctx.alive()) return;
        clearNode(warnBox);
        renderRefs();
      } catch (err) {
        showError(err);
      }
    });

    startBtn.addEventListener('click', (e) => {
      if ((station.references || []).length === 0) {
        e.preventDefault();
        toast(t('need_reference'), 'info');
      }
    });

    mount(
      stationNav(sid, 'setup'),
      el('div', { class: 'two-col' },
        el('div', { class: 'col-main' },
          stage.node,
          slider,
          el('div', { class: 'stack' }, shootBtn, el('div', { class: 'stack-row' }, sysCamera, gallery))),
        el('div', { class: 'col-side stack' },
          el('section', { class: 'card stack' },
            stationHeader(station),
            el('p', { class: 'help strong' }, t('setup_hint')),
            countLine,
            thumbs,
            warnBox,
            clearBtn),
          startBtn)));
    renderRefs();
  }

  // ---------------------------------------------------------------------------
  // Screen: Operator run
  // ---------------------------------------------------------------------------

  async function renderRun(ctx, sid) {
    const station = await getStation(sid);
    if (!ctx.alive()) return;
    store.set(LS_SID, sid);
    if (!(station.references || []).length) {
      toast(t('need_reference'), 'info');
      go(`s/${sid}/setup`);
      return;
    }
    setChrome({ title: t('run_title'), back: '' });

    let inflight = false;
    let autoOn = false;
    let autoTimer = null;
    let compareOn = store.get(LS_COMPARE) !== '0'; // on by default: it is the key demo moment
    let autoEvery = Number(store.get(LS_AUTO_EVERY));
    if (AUTO_CHOICES.indexOf(autoEvery) < 0) autoEvery = 3;
    let lastRec = null;

    // --- controls ---
    const checkLabel = el('span', null, t('check'));
    const checkSpin = el('span', { class: 'spinner spinner-btn', hidden: true, 'aria-hidden': 'true' });
    const checkBtn = el('button', { class: 'btn btn-accent btn-xl btn-check', type: 'button', disabled: true }, icon('target'), checkLabel, checkSpin);
    const stage = cameraStage(ctx, { onReady: () => { checkBtn.disabled = inflight; } });
    const slider = opacitySlider(stage);
    stage.setOutline(outlineUrl(station));
    const autoSwitch = el('button', { class: 'switch', type: 'button', role: 'switch', 'aria-checked': 'false' },
      el('span', { class: 'switch-track' }, el('span', { class: 'switch-thumb' })), el('span', null, t('auto')));
    const everyBtns = AUTO_CHOICES.map((n) => el('button', {
      class: 'seg-btn', type: 'button', 'aria-pressed': String(n === autoEvery),
      onclick: () => {
        autoEvery = n;
        store.set(LS_AUTO_EVERY, String(n));
        everyBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(AUTO_CHOICES[i] === n)));
        if (autoOn) scheduleAuto();
      },
    }, t('every_s', { n })));
    const sysCamera = fileButton(t('use_system_camera'), 'camera', 'btn-ghost btn-sm', { capture: 'environment' }, onFile);
    const autoStatus = el('span', { class: 'auto-status muted small' });

    // --- result area ---
    const resultBox = el('div', { class: 'result-box' },
      el('div', { class: 'verdict verdict-empty verdict-big' }, icon('target', 'verdict-icon'),
        el('div', { class: 'verdict-text' }, el('strong', { class: 'verdict-title' }, t('waiting_check')),
          el('span', { class: 'verdict-sub' }, t('waiting_check_sub')))));
    const compareSwitch = el('button', { class: 'switch', type: 'button', role: 'switch', 'aria-checked': String(compareOn) },
      el('span', { class: 'switch-track' }, el('span', { class: 'switch-thumb' })), el('span', null, t('cmp_toggle')));
    const compareBox = el('div', { class: 'compare-box', hidden: !compareOn });

    function setInflight(on) {
      inflight = on;
      checkBtn.disabled = on || !stage.camera.isReady();
      checkSpin.hidden = !on;
      checkBtn.classList.toggle('is-busy', on);
      checkLabel.textContent = on ? t('checking') : t('check');
    }

    function showRecord(rec) {
      lastRec = rec;
      replaceContent(resultBox,
        verdictBanner(rec, true),
        metaLine(rec),
        zoneHints(rec),
        reviewBar(sid, rec, (r) => showRecord(r)),
        resultFigure(rec));
      renderCompare();
    }

    function renderCompare() {
      compareBox.hidden = !compareOn;
      if (!compareOn) return;
      if (!lastRec) replaceContent(compareBox, el('p', { class: 'muted small' }, t('cmp_after_check')));
      else replaceContent(compareBox, recordCompare(lastRec));
    }

    async function inspect(blob, filename) {
      if (inflight) return;
      setInflight(true);
      const form = new FormData();
      form.append('file', blob, filename);
      try {
        const rec = await api(paths.inspect(sid, compareOn), { method: 'POST', form });
        if (!ctx.alive()) return;
        showRecord(rec);
        if (rec.verdict === 'fail') vibrate([120, 60, 120]);
      } catch (err) {
        if (ctx.alive()) showError(err);
      } finally {
        if (ctx.alive()) setInflight(false);
      }
    }

    async function captureAndInspect() {
      if (inflight || !stage.camera.isReady()) return;
      let blob;
      try {
        blob = await grabFrame(stage.video);
      } catch (err) {
        showError(err);
        return;
      }
      stage.flash();
      await inspect(blob, `frame_${Date.now()}.jpg`);
    }

    async function onFile(file) {
      inspect(await prepareFile(file), file.name || 'frame.jpg');
    }

    checkBtn.addEventListener('click', captureAndInspect);

    // --- auto mode: one capture every N s, never overlapping requests ---
    function scheduleAuto() {
      clearTimeout(autoTimer);
      if (!autoOn || !ctx.alive()) return;
      autoTimer = setTimeout(() => {
        if (!autoOn || !ctx.alive()) return;
        scheduleAuto(); // fixed cadence; a tick is skipped while the previous request still runs
        if (inflight) autoStatus.textContent = t('auto_skip');
        else if (!document.hidden) {
          autoStatus.textContent = t('auto_on', { n: autoEvery });
          captureAndInspect();
        }
      }, autoEvery * 1000);
    }

    function setAuto(on) {
      autoOn = on;
      autoSwitch.setAttribute('aria-checked', String(on));
      autoStatus.textContent = on ? t('auto_on', { n: autoEvery }) : '';
      if (on) scheduleAuto(); else clearTimeout(autoTimer);
    }
    autoSwitch.addEventListener('click', () => {
      if (!autoOn && !stage.camera.isReady()) {
        toast(t('err_no_frame'), 'info');
        return;
      }
      setAuto(!autoOn);
    });
    onLeave(() => { autoOn = false; clearTimeout(autoTimer); });

    compareSwitch.addEventListener('click', () => {
      compareOn = !compareOn;
      store.set(LS_COMPARE, compareOn ? '1' : '0');
      compareSwitch.setAttribute('aria-checked', String(compareOn));
      renderCompare();
    });
    renderCompare();

    // Keyboard shortcut for a laptop station: Space / Enter = check.
    const onKey = (e) => {
      if (e.target && /^(INPUT|TEXTAREA|SELECT|BUTTON|A)$/.test(e.target.tagName)) return;
      if (e.code === 'Space' || e.key === 'Enter') { e.preventDefault(); captureAndInspect(); }
    };
    document.addEventListener('keydown', onKey);
    onLeave(() => document.removeEventListener('keydown', onKey));

    mount(
      stationNav(sid, 'run'),
      el('div', { class: 'two-col' },
        el('div', { class: 'col-main' },
          stage.node,
          station.outline_url ? slider : null,
          el('div', { class: 'stack' },
            checkBtn,
            el('div', { class: 'auto-row' }, autoSwitch, el('div', { class: 'seg', role: 'group', 'aria-label': t('auto_every') }, everyBtns)),
            autoStatus,
            el('div', { class: 'stack-row' }, sysCamera))),
        el('div', { class: 'col-side stack' },
          resultBox,
          el('section', { class: 'card stack' }, compareSwitch, compareBox),
          qrCard(sid),
          el('div', { class: 'stack-row' },
            el('a', { class: 'btn btn-secondary', href: `#/s/${sid}/journal` }, icon('list'), t('nav_journal')),
            el('a', { class: 'btn btn-ghost', href: `#/s/${sid}/setup` }, icon('target'), t('nav_setup'))))));
  }

  // ---------------------------------------------------------------------------
  // Screen: Viewer (judge's phone)
  // ---------------------------------------------------------------------------

  async function renderViewer(ctx, sid) {
    let station = await getStation(sid);
    if (!ctx.alive()) return;
    setChrome({ title: t('view_title'), back: '' });

    const records = new Map(); // seq -> record
    let after = Math.max(0, (Number(station.last_seq) || 0) - PAGE_LIMIT);
    let timer = null;
    let lastShownSeq = null;
    let stationBusy = false;

    const live = el('span', { class: 'live' }, el('span', { class: 'live-dot' }), el('span', { class: 'live-text' }, t('live_connecting')));
    const latestBox = el('div', { class: 'result-box' },
      el('div', { class: 'verdict verdict-empty verdict-big' }, el('div', { class: 'spinner' }),
        el('div', { class: 'verdict-text' }, el('strong', { class: 'verdict-title' }, t('waiting_first')),
          el('span', { class: 'verdict-sub' }, t('waiting_first_sub')))));
    const compareBox = el('div', { class: 'compare-box' });
    const compareCard = el('section', { class: 'card stack', hidden: true },
      el('h3', { class: 'section-title' }, t('cmp_title')), compareBox);
    const tilesBox = el('div', null, statsTiles(station.stats));
    const list = el('ul', { class: 'rec-list' });
    const listEmpty = el('p', { class: 'muted small' }, t('journal_empty'));

    function setLive(ok) {
      live.classList.toggle('is-off', !ok);
      live.querySelector('.live-text').textContent = ok ? t('live_on') : t('live_off');
    }

    function update(rec) {
      records.set(rec.seq, rec);
      render();
    }

    function sorted() {
      return Array.from(records.values()).sort((a, b) => b.seq - a.seq);
    }

    function render() {
      const all = sorted();
      listEmpty.hidden = all.length > 0;
      replaceContent(list, all.slice(0, PAGE_LIMIT).map((r) => recordRow(sid, r, update)));
      const latest = all[0];
      if (!latest) return;
      replaceContent(latestBox,
        verdictBanner(latest, true),
        metaLine(latest),
        zoneHints(latest),
        reviewBar(sid, latest, update),
        resultFigure(latest));
      compareCard.hidden = !latest.baseline;
      if (latest.baseline) replaceContent(compareBox, recordCompare(latest));
      if (latest.seq !== lastShownSeq) {
        if (lastShownSeq !== null && latest.verdict === 'fail') vibrate(200);
        lastShownSeq = latest.seq;
      }
    }

    async function refreshStation() {
      if (stationBusy) return;
      stationBusy = true;
      try {
        station = await getStation(sid);
        if (ctx.alive()) replaceContent(tilesBox, statsTiles(station.stats));
      } catch (e) { /* next poll retries */ }
      stationBusy = false;
    }

    // Sequential polling (never overlapping); paused while the tab is hidden.
    async function poll() {
      if (!ctx.alive()) return;
      let again = POLL_MS;
      if (!document.hidden) {
        try {
          const data = await api(paths.inspections(sid, after, PAGE_LIMIT));
          if (!ctx.alive()) return;
          setLive(true);
          const items = (data && Array.isArray(data.items)) ? data.items : [];
          const lastSeq = data && typeof data.last_seq === 'number' ? data.last_seq : null;
          if (lastSeq !== null && lastSeq < after) {
            // The journal was reset on the server: start over.
            records.clear();
            after = 0;
            again = 0;
          } else if (items.length) {
            items.forEach((r) => records.set(r.seq, r));
            after = Math.max(after, ...items.map((r) => r.seq));
            render();
            refreshStation();
            if (items.length >= PAGE_LIMIT) again = 0; // catch up quickly
          }
        } catch (err) {
          if (!ctx.alive()) return;
          setLive(false);
        }
      }
      timer = setTimeout(poll, again);
    }
    onLeave(() => clearTimeout(timer));

    mount(
      el('div', { class: 'viewer-head' }, stationHeader(station), live),
      el('div', { class: 'two-col' },
        el('div', { class: 'col-main stack' }, latestBox, compareCard),
        el('div', { class: 'col-side stack' },
          tilesBox,
          el('section', { class: 'card stack' },
            el('h3', { class: 'section-title' }, t('journal_title')),
            listEmpty,
            list),
          el('div', { class: 'stack-row' },
            el('a', { class: 'btn btn-secondary', href: `#/s/${sid}/journal` }, icon('list'), t('nav_journal')),
            el('a', { class: 'btn btn-ghost', href: paths.report(sid), target: '_blank', rel: 'noopener' }, icon('doc'), t('shift_report'))))));
    poll();
  }

  // ---------------------------------------------------------------------------
  // Screen: Journal
  // ---------------------------------------------------------------------------

  async function renderJournal(ctx, sid) {
    const [station, items] = await Promise.all([getStation(sid), loadAllInspections(sid, ctx)]);
    if (!ctx.alive()) return;
    setChrome({ title: t('journal_title'), back: '' });

    const records = new Map(items.map((r) => [r.seq, r]));
    let filter = 'all';
    const list = el('ul', { class: 'rec-list' });
    const empty = el('p', { class: 'muted' }, t('journal_empty'));

    function update(rec) {
      records.set(rec.seq, rec);
      render();
    }

    function render() {
      const all = Array.from(records.values()).sort((a, b) => b.seq - a.seq)
        .filter((r) => filter === 'all' || verdictClass(r.verdict) === filter);
      empty.hidden = all.length > 0;
      replaceContent(list, all.map((r) => recordRow(sid, r, update)));
    }

    const filters = ['all', 'fail', 'pass', 'retake'];
    const filterBtns = filters.map((f) => el('button', {
      class: 'seg-btn', type: 'button', 'aria-pressed': String(f === filter),
      onclick: () => {
        filter = f;
        filterBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(filters[i] === f)));
        render();
      },
    }, t(`filter_${f}`)));

    mount(
      stationNav(sid, 'journal'),
      el('div', { class: 'viewer-head' }, stationHeader(station)),
      statsTiles(station.stats),
      el('div', { class: 'stack-row actions' },
        el('a', { class: 'btn btn-primary', href: paths.journalCsv(sid), download: `dalil_${sid}_journal.csv` }, icon('download'), t('download_csv')),
        el('a', { class: 'btn btn-secondary', href: paths.report(sid), target: '_blank', rel: 'noopener' }, icon('doc'), t('shift_report')),
        el('button', { class: 'btn btn-ghost', type: 'button', onclick: () => router() }, icon('refresh'), t('refresh'))),
      el('p', { class: 'chain' },
        el('span', { class: 'label-caps' }, t('chain_head')), ' ',
        el('code', { class: 'hash', title: station.chain_head || '' }, shortHash(station.chain_head) || '—')),
      el('div', { class: 'seg seg-wide', role: 'group', 'aria-label': t('filter') }, filterBtns),
      empty,
      list);
    render();
  }

  // ---------------------------------------------------------------------------
  // Screen: Quick compare (stateless)
  // ---------------------------------------------------------------------------

  function renderCompare(ctx) {
    setChrome({ title: t('quick_compare'), back: '' });
    const files = { reference: null, frame: null };
    const urls = [];
    onLeave(() => urls.forEach((u) => URL.revokeObjectURL(u)));

    const runBtn = el('button', { class: 'btn btn-accent btn-lg', type: 'button', disabled: true }, icon('compare'), t('compare_run'));
    const out = el('div', { class: 'compare-out stack' });

    function picker(key, title) {
      const input = el('input', { type: 'file', accept: 'image/*', class: 'visually-hidden' });
      const thumb = el('span', { class: 'picker-thumb' }, icon('image', 'icon-lg'));
      const caption = el('span', { class: 'picker-caption' }, t('pick_photo'));
      input.addEventListener('change', () => {
        const file = input.files && input.files[0];
        if (!file) return;
        files[key] = file;
        const url = URL.createObjectURL(file);
        urls.push(url);
        replaceContent(thumb, el('img', { src: url, alt: '' }));
        caption.textContent = file.name || t('photo_selected');
        runBtn.disabled = !(files.reference && files.frame);
        clearNode(out);
      });
      return el('label', { class: 'picker' }, input, thumb,
        el('span', { class: 'picker-text' }, el('strong', null, title), caption));
    }

    runBtn.addEventListener('click', async () => {
      if (!files.reference || !files.frame) return;
      try {
        const data = await withBusy(t('busy_comparing'), [runBtn], async () => {
          const [ref, frame] = await Promise.all([prepareFile(files.reference), prepareFile(files.frame)]);
          const form = new FormData();
          form.append('reference', ref, files.reference.name || 'reference.jpg');
          form.append('frame', frame, files.frame.name || 'frame.jpg');
          return api(paths.compare(), { method: 'POST', form });
        });
        if (!ctx.alive()) return;
        const results = (data && data.results) || {};
        const overlays = (data && data.overlays) || {};
        const src = (m) => (overlays[m] ? `data:image/jpeg;base64,${overlays[m]}` : null);
        const pbp = results.pbp || {};
        replaceContent(out,
          verdictBanner(pbp, true),
          metaLine(pbp),
          zoneHints(pbp),
          results.pixdiff
            ? el('section', { class: 'card stack' }, el('h3', { class: 'section-title' }, t('cmp_title')),
              compareGrid({ rec: pbp, src: src('pbp') }, { rec: results.pixdiff, src: src('pixdiff') }))
            : (src('pbp') ? el('figure', { class: 'figure' }, el('div', { class: 'figure-wrap' }, safeImg(src('pbp'), t('tab_zones')))) : null));
        out.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch (err) {
        showError(err);
      }
    });

    mount(
      el('p', { class: 'help strong' }, t('compare_help')),
      el('div', { class: 'pickers' }, picker('reference', t('pick_reference')), picker('frame', t('pick_frame'))),
      el('div', { class: 'stack actions' }, runBtn),
      out);
  }

  // ---------------------------------------------------------------------------
  // Language toggle + boot
  // ---------------------------------------------------------------------------

  function syncLangButtons() {
    const lang = I18N.getLang();
    document.querySelectorAll('[data-lang]').forEach((b) => {
      b.setAttribute('aria-pressed', String(b.getAttribute('data-lang') === lang));
    });
    $back.setAttribute('aria-label', t('back'));
    I18N.applyStatic(document);
  }

  document.querySelectorAll('[data-lang]').forEach((b) => {
    b.addEventListener('click', () => {
      const lang = b.getAttribute('data-lang');
      if (lang === I18N.getLang()) return;
      I18N.setLang(lang);
      syncLangButtons();
      router(); // re-render the current screen in the new language
    });
  });

  window.addEventListener('hashchange', router);
  window.addEventListener('pagehide', leaveScreen); // release the camera when the page goes away
  window.addEventListener('pageshow', (e) => { if (e.persisted) router(); }); // restored from bfcache

  syncLangButtons();
  router();
})();
