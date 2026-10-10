/*
 * AVA — UI strings (RU default, EN) and a tiny t(key, vars) helper.
 * Exposes window.I18N = { t, setLang, getLang, applyStatic, langs, dict }.
 * The chosen language is persisted in localStorage (best effort, never throws).
 */
(function (global) {
  'use strict';

  const DICT = {
    ru: {
      // header / intro
      tagline: 'Сравнение с эталоном: показываем только то, что изменилось. Без обучения на дефектах, без светового бокса.',
      privacy: 'Фото не сохраняются на сервере.',
      noscript: 'Для работы AVA включите JavaScript.',
      insecure_note: 'Живая камера доступна только по HTTPS — кнопка «Камера» откроет системную камеру.',
      lang_label: 'Язык',
      close: 'Закрыть',

      // QR modal
      qr_btn_label: 'Показать QR-код ссылки на этот сайт',
      qr_title: 'Откройте AVA на телефоне',
      qr_help: 'Наведите камеру телефона на код.',
      qr_local: 'Это локальный адрес — с телефона он не откроется. Покажите публичную ссылку.',
      qr_failed: 'Не удалось построить QR-код.',

      // photo slots
      slot_reference: 'Эталон',
      slot_reference_sub: 'годная деталь',
      slot_frame: 'Проверка',
      slot_frame_sub: 'деталь, которую проверяем',
      slot_empty: 'Нет фото',
      btn_camera: 'Камера',
      btn_gallery: 'Из галереи',
      btn_retake: 'Переснять',
      btn_replace: 'Заменить',
      photo_size: '{w}×{h} px',

      // compare
      compare_run: 'Сравнить',
      compare_busy: 'Сравниваем…',
      compare_need: 'Добавьте оба фото: эталон и проверку.',

      // verdicts
      v_pass: 'ГОДНА — изменений нет',
      v_fail: 'БРАК — найдено зон: {n}',
      v_retake: 'ПЕРЕСНЯТЬ: {reason}',
      v_pass_short: 'ГОДНА',
      v_fail_short: 'БРАК',
      v_retake_short: 'ПЕРЕСНЯТЬ',
      rr_alignment: 'не удалось совместить: снимите с того же ракурса',
      rr_glare: 'блик: смените угол на 15–30°',
      rr_blurry: 'размыто',
      rr_dark: 'темно',
      rr_mismatch: 'не та деталь или деталь отсутствует',
      rr_low_coverage: 'мало проверенной площади',
      rr_unknown: 'снимок не подходит',
      rh_alignment: 'Снимите деталь с того же ракурса и расстояния, что и эталон.',
      rh_glare: 'Наклоните телефон, чтобы блик ушёл с детали.',
      rh_blurry: 'Держите телефон неподвижно и коснитесь экрана для фокуса.',
      rh_dark: 'Добавьте света или включите фонарик.',
      rh_low_coverage: 'Деталь должна занимать большую часть кадра.',

      // result
      result_title: 'Результат AVA',
      overlay_alt: 'Проверяемое фото, совмещённое с эталоном, с зонами изменений',
      zones_title: 'Зоны',
      zones_help: 'Нажмите на зону, чтобы выделить её на снимке.',
      zone_name: 'Зона {n}',
      zone_area: '{p} % кадра',
      zh_water_or_dirt: 'похоже на загрязнение/каплю',
      zh_something_removed: 'отсутствует элемент?',
      zh_something_added: 'лишний элемент/наплыв?',
      meta_ms: 'Обработка: {n} мс',
      meta_coverage: 'Проверено: {n} %',
      meta_sha: 'SHA-256',
      sha_tap: 'Нажмите, чтобы показать отпечаток полностью',

      // baseline (plain reference subtraction)
      base_title: 'Простое вычитание из эталона',
      base_sub: 'Тот же совмещённый кадр, попиксельная разница с эталоном.',
      base_mismatch: 'БРАК — изменился почти весь кадр',
      base_mismatch_why: 'Так простое вычитание реагирует на смену света.',
      base_alt: 'Результат простого вычитания из эталона',
      base_none: 'Нет результата простого вычитания.',

      // after a result
      new_check: 'Новая проверка',
      new_reference: 'Новый эталон',
      history_title: 'История (только эта сессия)',

      // camera
      cam_title_reference: 'Эталон: снимите годную деталь',
      cam_title_frame: 'Проверка: совместите с эталоном',
      cam_starting: 'Включаем камеру…',
      cam_tap: 'Нажмите, чтобы включить камеру',
      cam_shutter: 'Снять',
      ghost_opacity: 'Эталон',
      cam_fail_insecure: 'Камера доступна только по HTTPS.',
      cam_fail_denied: 'Нет доступа к камере.',
      cam_fail_unavailable: 'Камера недоступна.',
      cam_fallback: 'Выберите фото из галереи.',

      // errors
      err_decode: 'Не удалось открыть фото (нужен JPEG или PNG).',
      err_no_frame: 'Камера ещё не готова — попробуйте снова.',
      err_network: 'Нет связи с сервером. Проверьте интернет.',
      err_timeout: 'Сервер не ответил вовремя. Попробуйте ещё раз.',
      err_http: 'Ошибка сервера ({status}).',
      err_bad_response: 'Сервер вернул неожиданный ответ.',
      err_generic: 'Что-то пошло не так.',
      api_bad_image: 'Не удалось прочитать фото (нужен JPEG или PNG).',
      api_empty_file: 'Пустой файл.',
      api_image_too_small: 'Фото слишком маленькое.',
      api_too_large: 'Файл слишком большой (максимум 15 МБ).',
      api_too_many_pixels: 'Слишком большое разрешение фото.',
      api_file_required: 'Нужны оба фото: эталон и проверка.',
      api_internal: 'Внутренняя ошибка сервера. Попробуйте ещё раз.',
    },

    en: {
      // header / intro
      tagline: 'Compare with a reference: we show only what has changed. No defect training, no light box.',
      privacy: 'Photos are not stored on the server.',
      noscript: 'Please enable JavaScript to use AVA.',
      insecure_note: 'The live camera needs HTTPS — the "Camera" button will open the system camera.',
      lang_label: 'Language',
      close: 'Close',

      // QR modal
      qr_btn_label: 'Show a QR code linking to this site',
      qr_title: 'Open AVA on your phone',
      qr_help: 'Point your phone camera at the code.',
      qr_local: 'This is a local address — a phone cannot open it. Show the public link instead.',
      qr_failed: 'Could not build the QR code.',

      // photo slots
      slot_reference: 'Reference',
      slot_reference_sub: 'a good part',
      slot_frame: 'Check',
      slot_frame_sub: 'the part to inspect',
      slot_empty: 'No photo',
      btn_camera: 'Camera',
      btn_gallery: 'From gallery',
      btn_retake: 'Retake',
      btn_replace: 'Replace',
      photo_size: '{w}×{h} px',

      // compare
      compare_run: 'Compare',
      compare_busy: 'Comparing…',
      compare_need: 'Add both photos: the reference and the check.',

      // verdicts
      v_pass: 'PASS — no changes',
      v_fail: 'FAIL — zones found: {n}',
      v_retake: 'RETAKE: {reason}',
      v_pass_short: 'PASS',
      v_fail_short: 'FAIL',
      v_retake_short: 'RETAKE',
      rr_alignment: 'could not align: shoot from the same angle',
      rr_glare: 'glare: change the angle by 15–30°',
      rr_blurry: 'blurry',
      rr_dark: 'too dark',
      rr_mismatch: 'wrong part or the part is missing',
      rr_low_coverage: 'too little area checked',
      rr_unknown: 'the photo is not usable',
      rh_alignment: 'Shoot the part from the same angle and distance as the reference.',
      rh_glare: 'Tilt the phone so the glare leaves the part.',
      rh_blurry: 'Hold the phone still and tap the screen to focus.',
      rh_dark: 'Add light or turn on the flashlight.',
      rh_low_coverage: 'The part should fill most of the frame.',

      // result
      result_title: 'AVA result',
      overlay_alt: 'The checked photo aligned to the reference, with changed zones',
      zones_title: 'Zones',
      zones_help: 'Tap a zone to highlight it on the image.',
      zone_name: 'Zone {n}',
      zone_area: '{p} % of the frame',
      zh_water_or_dirt: 'looks like dirt / a drop',
      zh_something_removed: 'missing feature?',
      zh_something_added: 'extra feature / excess material?',
      meta_ms: 'Processing: {n} ms',
      meta_coverage: 'Checked: {n} %',
      meta_sha: 'SHA-256',
      sha_tap: 'Tap to show the full fingerprint',

      // baseline (plain reference subtraction)
      base_title: 'Plain reference subtraction',
      base_sub: 'Same aligned frame, per-pixel difference from the reference.',
      base_mismatch: 'FAIL — almost the whole frame changed',
      base_mismatch_why: 'This is how plain subtraction reacts to a change of lighting.',
      base_alt: 'Plain reference subtraction result',
      base_none: 'No plain subtraction result.',

      // after a result
      new_check: 'New check',
      new_reference: 'New reference',
      history_title: 'History (this session only)',

      // camera
      cam_title_reference: 'Reference: shoot a good part',
      cam_title_frame: 'Check: match the reference',
      cam_starting: 'Starting the camera…',
      cam_tap: 'Tap to start the camera',
      cam_shutter: 'Take photo',
      ghost_opacity: 'Reference',
      cam_fail_insecure: 'The camera works over HTTPS only.',
      cam_fail_denied: 'Camera access was denied.',
      cam_fail_unavailable: 'The camera is unavailable.',
      cam_fallback: 'Pick a photo from the gallery.',

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
      api_file_required: 'Both photos are required: the reference and the check.',
      api_internal: 'Internal server error. Please try again.',
    },
  };

  const STORAGE_KEY = 'ava.lang';
  const LANGS = Object.keys(DICT);
  let lang = 'ru';

  try {
    const saved = global.localStorage.getItem(STORAGE_KEY);
    if (saved && DICT[saved]) lang = saved;
  } catch (e) { /* storage unavailable: keep default */ }

  /** Translate `key`, substituting {name} placeholders from `vars`. Falls back to RU, then the key. */
  function t(key, vars) {
    let s = DICT[lang][key];
    if (s === undefined) s = DICT.ru[key];
    if (s === undefined) s = key;
    if (vars) {
      s = s.replace(/\{(\w+)\}/g, (m, name) => (Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : m));
    }
    return s;
  }

  function setLang(next) {
    if (!DICT[next]) return;
    lang = next;
    try { global.localStorage.setItem(STORAGE_KEY, next); } catch (e) { /* ignore */ }
    if (global.document) global.document.documentElement.lang = next;
  }

  /** Fill static markup: data-i18n="key" sets the text, data-i18n-aria="key" the aria-label. */
  function applyStatic(root) {
    const scope = root || global.document;
    if (!scope || !scope.querySelectorAll) return;
    scope.querySelectorAll('[data-i18n]').forEach((node) => {
      node.textContent = t(node.getAttribute('data-i18n'));
    });
    scope.querySelectorAll('[data-i18n-aria]').forEach((node) => {
      node.setAttribute('aria-label', t(node.getAttribute('data-i18n-aria')));
    });
  }

  if (global.document) global.document.documentElement.lang = lang;

  global.I18N = { t, setLang, getLang: () => lang, applyStatic, langs: LANGS, dict: DICT };
})(window);
