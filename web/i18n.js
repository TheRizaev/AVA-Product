/*
 * AVA — UI strings (English only; the site has no language switch) and a tiny t(key, vars) helper.
 * Exposes window.TEXT = { t, strings }. Static labels live directly in index.html.
 */
(function (global) {
  'use strict';

  const STRINGS = {
    // modes
    mode_surface: 'Surface check',
    mode_compare: 'Compare with reference',
    mode_surface_short: 'Surface',
    mode_compare_short: 'Compare',
    mode_hint_surface: 'When to use: paint, plastic, sheet or brushed metal, fabric, mesh — the same texture everywhere.',
    mode_hint_compare: 'When to use: parts with holes, ribs, logos or unique features — shoot a good part first.',

    // examples
    examples_label: 'Try an example:',
    ex_defect: 'Defect',
    ex_ok: 'No defect',
    ex_light: 'Brightness change',
    ex_tag: 'Example image (synthetic)',
    ex_failed: 'The example image is not available yet.',

    // photo slots
    slot_image: 'Surface photo',
    slot_image_sub: 'Fill the frame with the surface only',
    slot_reference: 'Reference',
    slot_reference_sub: 'a good part',
    slot_frame: 'Check',
    slot_frame_sub: 'the part to inspect, same angle',
    slot_empty: 'No photo',
    btn_camera: 'Camera',
    btn_gallery: 'Gallery',
    btn_retake: 'Retake',
    btn_replace: 'Replace',
    photo_size: '{w}×{h} px',

    // run button
    run_surface: 'Check surface',
    run_compare: 'Compare',
    busy_surface: 'Checking…',
    busy_compare: 'Comparing…',
    need_surface: 'Add a photo of the surface.',
    need_compare: 'Add both photos: the reference and the check.',

    // verdicts
    v_pass: 'PASS — no changes found',
    v_fail_one: 'FAIL — 1 defect zone',
    v_fail_many: 'FAIL — {n} defect zones',
    v_retake: 'RETAKE — {reason}',
    v_pass_short: 'PASS',
    v_fail_short: 'FAIL',
    v_retake_short: 'RETAKE',
    rr_alignment: 'could not align — shoot from the same angle',
    rr_glare: 'glare — change the angle by 15–30°',
    rr_blurry: 'photo is blurry',
    rr_dark: 'photo is too dark',
    rr_mismatch: 'different part or part missing',
    rr_low_coverage: 'too little of the image could be checked',
    rr_unknown: 'the photo is not usable',
    rh_alignment: 'Shoot the part from the same angle and distance as the reference.',
    rh_glare: 'Tilt the phone so the glare leaves the part.',
    rh_blurry: 'Hold the phone still and tap the screen to focus.',
    rh_dark: 'Add light or turn on the flashlight.',
    rh_low_coverage_surface: 'Fill the frame with the surface and avoid strong glare.',
    rh_low_coverage_compare: 'The part should fill most of the frame.',

    // result
    result_title_surface: 'AVA surface check',
    result_title_compare: 'AVA result',
    overlay_alt: 'Checked photo with defect zones',
    zones_title: 'Zones',
    zones_help: 'Tap a zone to highlight it on the image.',
    zone_name: 'Zone {n}',
    zone_area: '{p} % of the frame',
    zh_water_or_dirt: 'looks like dirt or a drop',
    zh_something_removed: 'missing feature?',
    zh_something_added: 'extra material or burr?',
    meta_ms: '{n} ms',
    meta_coverage: 'checked {n} %',
    meta_sha: 'SHA-256',
    sha_tap: 'Tap to show the full fingerprint',

    // baseline (plain reference subtraction)
    base_title: 'Plain reference subtraction',
    base_sub: 'Same aligned photo, per-pixel difference from the reference.',
    base_mismatch: 'FAIL — almost the whole frame changed',
    base_mismatch_why: 'This is how plain subtraction reacts to a lighting change.',
    base_alt: 'Plain reference subtraction result',
    base_none: 'No plain subtraction result.',

    // after a result
    new_check: 'New check',
    new_reference: 'New reference',
    new_photo: 'New photo',
    history_title: 'History (this session only)',

    // camera
    cam_title_image: 'Surface: fill the frame',
    cam_title_reference: 'Reference: shoot a good part',
    cam_title_frame: 'Check: match the reference',
    cam_fail_denied: 'Camera access was denied.',
    cam_fail_unavailable: 'The camera is unavailable.',
    cam_fallback: 'Pick a photo from the gallery.',

    // QR
    qr_failed: 'Could not build the QR code.',

    // errors
    err_decode: 'Could not open the photo (JPEG or PNG expected).',
    err_no_frame: 'The camera is not ready yet — try again.',
    err_network: 'Cannot reach the server. Check your connection.',
    err_timeout: 'The server did not answer in time. Please try again.',
    err_http: 'Server error ({status}).',
    err_bad_response: 'Unexpected response from the server.',
    err_generic: 'Something went wrong.',
    api_bad_image: 'Could not read the photo (JPEG or PNG expected).',
    api_empty_file: 'The file is empty.',
    api_image_too_small: 'The photo is too small.',
    api_too_large: 'The file is too large (15 MB max).',
    api_too_many_pixels: 'The photo resolution is too large.',
    api_file_required: 'A photo is missing.',
    api_internal: 'Internal server error. Please try again.',
  };

  /** Look up `key`, substituting {name} placeholders from `vars`; an unknown key returns the key itself. */
  function t(key, vars) {
    let s = Object.prototype.hasOwnProperty.call(STRINGS, key) ? STRINGS[key] : key;
    if (vars) {
      s = s.replace(/\{(\w+)\}/g, (m, name) => (Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : m));
    }
    return s;
  }

  global.TEXT = { t, strings: STRINGS };
})(window);
