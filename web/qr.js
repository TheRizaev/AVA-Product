/*
 * Dalil QA — QR code as inline SVG.
 *
 * Thin wrapper around the vendored "qrcode-generator" 1.4.4 by Kazuhiko Arase (MIT licence,
 * see the header of web/qrcode.js; https://github.com/kazuhikoarase/qrcode-generator).
 * Exposes window.DalilQR = { render(container, text, opts) }.
 */
(function (global) {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const QUIET_ZONE = 4; // modules of white border required by the QR standard

  /**
   * Render `text` as a QR code SVG into `container` (its previous content is replaced).
   * opts: { ecc: 'L'|'M'|'Q'|'H' (default 'M'), label: accessible name }.
   * Returns the <svg> element, or null if the encoder is unavailable or the text does not fit.
   */
  function render(container, text, opts) {
    const o = opts || {};
    const factory = global.qrcode;
    while (container.firstChild) container.removeChild(container.firstChild);
    if (typeof factory !== 'function') return null;

    let qr;
    try {
      if (factory.stringToBytesFuncs && factory.stringToBytesFuncs['UTF-8']) {
        factory.stringToBytes = factory.stringToBytesFuncs['UTF-8'];
      }
      qr = factory(0, o.ecc || 'M'); // type 0 = smallest version that fits
      qr.addData(String(text), 'Byte');
      qr.make();
    } catch (e) {
      return null;
    }

    const n = qr.getModuleCount();
    const size = n + 2 * QUIET_ZONE;
    // One path of 1×1 squares: compact and crisp at any zoom.
    let d = '';
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        if (qr.isDark(r, c)) d += `M${c + QUIET_ZONE} ${r + QUIET_ZONE}h1v1h-1z`;
      }
    }

    const root = document.createElementNS(SVG_NS, 'svg');
    root.setAttribute('viewBox', `0 0 ${size} ${size}`);
    root.setAttribute('shape-rendering', 'crispEdges');
    root.setAttribute('role', 'img');
    root.setAttribute('aria-label', o.label || String(text));
    root.setAttribute('class', 'qr-svg');
    // Always dark-on-white, also in dark mode: many scanners cannot read inverted codes.
    const bg = document.createElementNS(SVG_NS, 'rect');
    bg.setAttribute('width', String(size));
    bg.setAttribute('height', String(size));
    bg.setAttribute('fill', '#ffffff');
    const fg = document.createElementNS(SVG_NS, 'path');
    fg.setAttribute('d', d);
    fg.setAttribute('fill', '#000000');
    root.appendChild(bg);
    root.appendChild(fg);
    container.appendChild(root);
    return root;
  }

  global.DalilQR = { render };
})(window);
