/*
 * Dalil QA — UI strings (RU default, EN) and a tiny t(key, vars) helper.
 * Exposes window.I18N = { t, plural, setLang, getLang, applyStatic, langs, dict }.
 * The chosen language is persisted in localStorage (best effort, never throws).
 */
(function (global) {
  'use strict';

  const DICT = {
    ru: {
      // chrome / common
      back: 'Назад',
      loading: 'Загрузка…',
      noscript: 'Для работы Dalil QA включите JavaScript.',
      optional: 'необязательно',
      retry: 'Повторить',
      refresh: 'Обновить',
      to_home: 'На главную',
      copied: 'Скопировано',
      copy_failed: 'Не удалось скопировать',
      copy_link: 'Копировать ссылку',
      station: 'Станция',
      station_code: 'Код станции',
      station_name: 'Название станции',
      sku: 'Артикул (SKU)',
      image_missing: 'Изображение недоступно',
      open_image: 'Открыть изображение',
      zone_forms: 'зона|зоны|зон',

      // home
      tagline: 'Контроль деталей по эталону — без обучения на дефектах, без светового бокса',
      new_station: 'Новая станция',
      continue_station: 'Продолжить: станция {code}',
      join_title: 'Подключиться к станции',
      join: 'Открыть',
      quick_compare: 'Сравнить два фото',
      step1_title: 'Эталон',
      step1_text: 'Один раз снимите годную деталь (1–5 кадров).',
      step2_title: 'Контроль',
      step2_text: 'Каждую деталь Dalil совмещает с эталоном и сравнивает PBP-подписями: ГОДНА, БРАК с зоной или ПЕРЕСНЯТЬ.',
      step3_title: 'Журнал смены',
      step3_text: 'Хэш-цепочка записей, подтверждение оператора, CSV и отчёт смены.',
      privacy_note: 'Снимайте только детали — без лиц и документов.',
      server_ok: 'Сервер на связи · версия {v}',
      server_down: 'Сервер недоступен',

      // station nav
      nav_setup: 'Эталон',
      nav_run: 'Контроль',
      nav_view: 'Наблюдатель',
      nav_journal: 'Журнал',

      // new station
      name_placeholder: 'Например, Участок штамповки №2',
      sku_placeholder: 'Например, BRK-1042',
      sku_help: 'Одна станция — одна деталь. Для новой детали создайте новую станцию.',
      create_station: 'Создать станцию',
      err_name_required: 'Введите название станции.',

      // setup
      setup_title: 'Эталон',
      setup_hint: 'Снимите годную деталь; можно 2–3 кадра с небольшим сдвигом.',
      shoot_reference: 'Снять эталон',
      use_system_camera: 'Камера телефона',
      from_gallery: 'Из галереи',
      ref_count: 'Кадров эталона: {n} из {max}',
      ref_none: 'пока нет',
      ref_alt: 'Эталон {n}',
      ref_ok: 'Кадр эталона сохранён, качество в порядке.',
      ref_warn_title: 'Кадр сохранён, но есть замечания:',
      ref_warn_hint: 'Лучше очистить эталон и переснять.',
      warn_blurry: 'смазано — закрепите телефон',
      warn_glare: 'блики — смените угол на 15–30°',
      warn_dark: 'слишком темно — добавьте света',
      clear_reference: 'Очистить эталон',
      clear_confirm: 'Удалить все кадры эталона этой станции?',
      start_control: 'Начать контроль',
      need_reference: 'Сначала снимите эталон годной детали.',

      // camera
      camera_starting: 'Включаем камеру…',
      camera_tap: 'Включить камеру',
      outline_opacity: 'Контур эталона',
      cam_fail_insecure: 'Камера в браузере работает только по HTTPS.',
      cam_fail_denied: 'Доступ к камере запрещён.',
      cam_fail_unavailable: 'Камера недоступна.',
      cam_fallback: 'Сделайте снимок камерой телефона кнопкой ниже.',

      // operator run
      run_title: 'Контроль',
      check: 'Проверить',
      checking: 'Проверяем…',
      auto: 'Авто',
      auto_every: 'Интервал автопроверки',
      every_s: '{n} с',
      auto_on: 'Авто: проверка каждые {n} с',
      auto_skip: 'Авто: предыдущая проверка ещё идёт — пропуск',
      waiting_check: 'Положите деталь',
      waiting_check_sub: 'Совместите с контуром эталона и нажмите «Проверить».',
      qr_label: 'QR-код: экран наблюдателя',
      qr_help: 'Наведите камеру телефона на QR — откроется экран наблюдателя с вердиктами.',
      qr_unavailable: 'QR-код недоступен — введите код станции на главной.',

      // verdicts
      v_pass: 'ГОДНА',
      v_pass_detail: 'отличий нет',
      v_fail: 'БРАК',
      v_fail_short: 'БРАК',
      v_retake: 'ПЕРЕСНЯТЬ',
      v_retake_short: 'ПЕРЕСНЯТЬ',
      rr_alignment: 'не совмещается с эталоном',
      rr_glare: 'блик',
      rr_blurry: 'смазанный кадр',
      rr_dark: 'слишком темно',
      rr_low_coverage: 'мало проверяемой площади',
      rr_mismatch: 'не та деталь или деталь отсутствует',
      rr_unknown: 'кадр не подходит',
      rh_alignment: 'Положите деталь так же, как на эталоне.',
      rh_blurry: 'Держите камеру неподвижно.',
      rh_dark: 'Добавьте света.',
      rh_low_coverage: 'Деталь целиком в кадре, без бликов.',
      hint_change_angle: 'Смените угол на 15–30°.',
      zh_water_or_dirt: 'похоже на загрязнение/каплю',
      zh_something_removed: 'отсутствует элемент?',
      zh_something_added: 'лишний элемент/наплыв?',
      coverage_pct: 'проверено {n} %',
      ms: '{n} мс',
      align_method: 'совмещение: {m}',
      tab_zones: 'Зоны',
      tab_frame: 'Кадр',

      // comparison with the baseline
      cmp_toggle: 'Сравнение с простым вычитанием',
      cmp_title: 'Сравнение на том же кадре',
      cmp_dalil: 'Dalil (PBP)',
      cmp_dalil_sub: 'структура мономов, устойчива к монотонному свету',
      cmp_baseline: 'Простое вычитание из эталона',
      cmp_baseline_sub: 'разность яркостей пикселей',
      cmp_differ: 'Решения методов расходятся.',
      cmp_none: 'Для этой проверки нет результата простого вычитания.',
      cmp_after_check: 'Появится после следующей проверки.',
      baseline_short: 'Вычитание',

      // review
      review_defect: 'Дефект подтверждён',
      review_false_alarm: 'Ложная тревога',
      review_ok: 'Верно: годна',
      review_missed: 'Пропущен дефект',
      review_saved: 'Отметка оператора сохранена',
      rv_defect: 'дефект подтверждён',
      rv_false_alarm: 'ложная тревога',
      rv_ok: 'подтверждено: годна',

      // viewer
      view_title: 'Наблюдатель',
      live_connecting: 'подключение…',
      live_on: 'в эфире',
      live_off: 'нет связи',
      waiting_first: 'Ждём проверку…',
      waiting_first_sub: 'Вердикт появится здесь сразу после проверки на станции.',

      // journal
      journal_title: 'Журнал смены',
      journal_empty: 'Проверок пока нет.',
      st_total: 'всего',
      st_pass: 'годна',
      st_fail: 'брак',
      st_retake: 'переснять',
      st_confirmed: 'брак подтверждён',
      st_false: 'ложные тревоги',
      download_csv: 'Скачать CSV',
      shift_report: 'Отчёт смены',
      chain_head: 'Хэш-цепочка:',
      filter: 'Фильтр',
      filter_all: 'Все',
      filter_fail: 'Брак',
      filter_pass: 'Годна',
      filter_retake: 'Переснять',

      // quick compare
      compare_help: 'Выберите фото годной детали (эталон) и проверяемой детали. Ничего не сохраняется.',
      pick_reference: 'Эталон (годная деталь)',
      pick_frame: 'Проверяемая деталь',
      pick_photo: 'Выбрать фото',
      photo_selected: 'Фото выбрано',
      compare_run: 'Сравнить',

      // busy / errors
      busy_default: 'Подождите…',
      busy_creating: 'Создаём станцию…',
      busy_connecting: 'Подключаемся…',
      busy_reference: 'Сохраняем эталон…',
      busy_comparing: 'Сравниваем…',
      err_title: 'Ошибка',
      err_generic: 'Что-то пошло не так.',
      err_network: 'Нет связи с сервером. Проверьте интернет.',
      err_http: 'Ошибка сервера ({status}).',
      err_not_found: 'Не найдено. Проверьте код станции.',
      err_too_large: 'Файл слишком большой (максимум 15 МБ).',
      err_no_frame: 'Камера ещё не готова — попробуйте снова.',
      err_bad_code: 'Код станции — 6 символов: латинские буквы и цифры 2–9.',
      // server error codes ({"error": code}) shown instead of the English server message
      api_no_reference: 'Сначала снимите эталон (годную деталь).',
      api_reference_mismatch: 'Кадр не совпадает с первым эталоном: та же деталь, та же позиция камеры.',
      api_too_many_references: 'Эталонов уже 5 — очистите и снимите заново.',
      api_station_not_found: 'Станция не найдена. Проверьте код.',
      api_bad_station_id: 'Код станции — 6 символов: латинские буквы и цифры 2–9.',
      api_inspection_not_found: 'Проверка не найдена.',
      api_bad_image: 'Не удалось прочитать изображение (нужен JPEG или PNG).',
      api_image_too_small: 'Изображение слишком маленькое.',
      api_too_large: 'Файл слишком большой (максимум 15 МБ).',
      api_too_many_pixels: 'Слишком большое разрешение изображения.',
      api_name_required: 'Введите название станции.',
      api_internal: 'Внутренняя ошибка сервера. Попробуйте ещё раз.',
    },

    en: {
      back: 'Back',
      loading: 'Loading…',
      noscript: 'Please enable JavaScript to use Dalil QA.',
      optional: 'optional',
      retry: 'Retry',
      refresh: 'Refresh',
      to_home: 'Home',
      copied: 'Copied',
      copy_failed: 'Could not copy',
      copy_link: 'Copy link',
      station: 'Station',
      station_code: 'Station code',
      station_name: 'Station name',
      sku: 'Part number (SKU)',
      image_missing: 'Image unavailable',
      open_image: 'Open image',
      zone_forms: 'zone|zones',

      tagline: 'Reference-based part inspection — no defect training data, no light box',
      new_station: 'New station',
      continue_station: 'Continue: station {code}',
      join_title: 'Connect to a station',
      join: 'Open',
      quick_compare: 'Compare two photos',
      step1_title: 'Reference',
      step1_text: 'Shoot a good part once (1–5 frames).',
      step2_title: 'Inspection',
      step2_text: 'Dalil aligns every part to the reference and compares PBP signatures: PASS, FAIL with zones, or RETAKE.',
      step3_title: 'Shift journal',
      step3_text: 'Hash-chained records, operator confirmation, CSV and shift report.',
      privacy_note: 'Photograph parts only — no faces or documents.',
      server_ok: 'Server online · version {v}',
      server_down: 'Server unavailable',

      nav_setup: 'Reference',
      nav_run: 'Inspect',
      nav_view: 'Viewer',
      nav_journal: 'Journal',

      name_placeholder: 'e.g. Stamping cell 2',
      sku_placeholder: 'e.g. BRK-1042',
      sku_help: 'One station — one part. Create a new station for a new part.',
      create_station: 'Create station',
      err_name_required: 'Enter a station name.',

      setup_title: 'Reference',
      setup_hint: 'Shoot a good part; 2–3 frames with a small shift are fine.',
      shoot_reference: 'Shoot reference',
      use_system_camera: 'Phone camera',
      from_gallery: 'From gallery',
      ref_count: 'Reference frames: {n} of {max}',
      ref_none: 'none yet',
      ref_alt: 'Reference {n}',
      ref_ok: 'Reference frame saved, quality is fine.',
      ref_warn_title: 'Frame saved, but:',
      ref_warn_hint: 'Better clear the reference and shoot again.',
      warn_blurry: 'blurry — fix the phone in place',
      warn_glare: 'glare — change the angle by 15–30°',
      warn_dark: 'too dark — add light',
      clear_reference: 'Clear reference',
      clear_confirm: 'Delete all reference frames of this station?',
      start_control: 'Start inspection',
      need_reference: 'Shoot a reference of a good part first.',

      camera_starting: 'Starting camera…',
      camera_tap: 'Start camera',
      outline_opacity: 'Reference outline',
      cam_fail_insecure: 'The browser camera needs HTTPS.',
      cam_fail_denied: 'Camera access denied.',
      cam_fail_unavailable: 'Camera unavailable.',
      cam_fallback: 'Use the phone camera button below.',

      run_title: 'Inspect',
      check: 'Check',
      checking: 'Checking…',
      auto: 'Auto',
      auto_every: 'Auto-check interval',
      every_s: '{n} s',
      auto_on: 'Auto: checking every {n} s',
      auto_skip: 'Auto: previous check still running — skipped',
      waiting_check: 'Place the part',
      waiting_check_sub: 'Match it to the reference outline and press “Check”.',
      qr_label: 'QR code: viewer screen',
      qr_help: 'Point a phone camera at the QR to open the live viewer.',
      qr_unavailable: 'QR unavailable — enter the station code on the home screen.',

      v_pass: 'PASS',
      v_pass_detail: 'no differences',
      v_fail: 'FAIL',
      v_fail_short: 'FAIL',
      v_retake: 'RETAKE',
      v_retake_short: 'RETAKE',
      rr_alignment: 'cannot align with the reference',
      rr_glare: 'glare',
      rr_blurry: 'blurry frame',
      rr_dark: 'too dark',
      rr_low_coverage: 'too little checkable area',
      rr_mismatch: 'wrong part or part missing',
      rr_unknown: 'frame not usable',
      rh_alignment: 'Place the part as in the reference.',
      rh_blurry: 'Keep the camera still.',
      rh_dark: 'Add light.',
      rh_low_coverage: 'Whole part in frame, no glare.',
      hint_change_angle: 'Change the angle by 15–30°.',
      zh_water_or_dirt: 'looks like dirt / a drop',
      zh_something_removed: 'missing feature?',
      zh_something_added: 'extra feature / flash?',
      coverage_pct: '{n} % checked',
      ms: '{n} ms',
      align_method: 'alignment: {m}',
      tab_zones: 'Zones',
      tab_frame: 'Frame',

      cmp_toggle: 'Compare with plain subtraction',
      cmp_title: 'Comparison on the same frame',
      cmp_dalil: 'Dalil (PBP)',
      cmp_dalil_sub: 'monomial structure, robust to monotone lighting',
      cmp_baseline: 'Plain reference subtraction',
      cmp_baseline_sub: 'pixel brightness difference',
      cmp_differ: 'The methods disagree.',
      cmp_none: 'No plain-subtraction result for this check.',
      cmp_after_check: 'Appears after the next check.',
      baseline_short: 'Subtraction',

      review_defect: 'Defect confirmed',
      review_false_alarm: 'False alarm',
      review_ok: 'Correct: pass',
      review_missed: 'Missed defect',
      review_saved: 'Operator label saved',
      rv_defect: 'defect confirmed',
      rv_false_alarm: 'false alarm',
      rv_ok: 'confirmed: pass',

      view_title: 'Viewer',
      live_connecting: 'connecting…',
      live_on: 'live',
      live_off: 'offline',
      waiting_first: 'Waiting for a check…',
      waiting_first_sub: 'The verdict appears here right after a check at the station.',

      journal_title: 'Shift journal',
      journal_empty: 'No checks yet.',
      st_total: 'total',
      st_pass: 'pass',
      st_fail: 'fail',
      st_retake: 'retake',
      st_confirmed: 'confirmed defects',
      st_false: 'false alarms',
      download_csv: 'Download CSV',
      shift_report: 'Shift report',
      chain_head: 'Hash chain:',
      filter: 'Filter',
      filter_all: 'All',
      filter_fail: 'Fail',
      filter_pass: 'Pass',
      filter_retake: 'Retake',

      compare_help: 'Pick a photo of a good part (reference) and of the part to check. Nothing is stored.',
      pick_reference: 'Reference (good part)',
      pick_frame: 'Part to check',
      pick_photo: 'Choose photo',
      photo_selected: 'Photo selected',
      compare_run: 'Compare',

      busy_default: 'Please wait…',
      busy_creating: 'Creating station…',
      busy_connecting: 'Connecting…',
      busy_reference: 'Saving reference…',
      busy_comparing: 'Comparing…',
      err_title: 'Error',
      err_generic: 'Something went wrong.',
      err_network: 'Cannot reach the server. Check your connection.',
      err_http: 'Server error ({status}).',
      err_not_found: 'Not found. Check the station code.',
      err_too_large: 'File is too large (15 MB max).',
      err_no_frame: 'Camera is not ready yet — try again.',
      err_bad_code: 'Station code is 6 characters: Latin letters and digits 2–9.',
      api_no_reference: 'Shoot the reference (a good part) first.',
      api_reference_mismatch: 'This shot does not match the first reference: same part, same camera position.',
      api_too_many_references: 'Already 5 reference shots — clear them and shoot again.',
      api_station_not_found: 'Station not found. Check the code.',
      api_bad_station_id: 'Station code is 6 characters: Latin letters and digits 2–9.',
      api_inspection_not_found: 'Inspection not found.',
      api_bad_image: 'Could not read the image (JPEG or PNG expected).',
      api_image_too_small: 'The image is too small.',
      api_too_large: 'File is too large (15 MB max).',
      api_too_many_pixels: 'Image resolution is too large.',
      api_name_required: 'Enter a station name.',
      api_internal: 'Internal server error. Please try again.',
    },
  };

  const STORAGE_KEY = 'dalil.lang';
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

  /** Plural form of a "one|few|many" (RU) or "one|other" (EN) dictionary entry for `n`. */
  function plural(n, key) {
    const forms = t(key).split('|');
    const abs = Math.abs(Math.trunc(Number(n) || 0));
    let idx;
    if (lang === 'ru') {
      const d10 = abs % 10;
      const d100 = abs % 100;
      if (d10 === 1 && d100 !== 11) idx = 0;
      else if (d10 >= 2 && d10 <= 4 && (d100 < 12 || d100 > 14)) idx = 1;
      else idx = 2;
    } else {
      idx = abs === 1 ? 0 : 1;
    }
    return forms[Math.min(idx, forms.length - 1)];
  }

  function setLang(next) {
    if (!DICT[next]) return;
    lang = next;
    try { global.localStorage.setItem(STORAGE_KEY, next); } catch (e) { /* ignore */ }
    if (global.document) global.document.documentElement.lang = next;
  }

  /** Fill elements marked with data-i18n="key" (static markup in index.html). */
  function applyStatic(root) {
    const scope = root || global.document;
    if (!scope || !scope.querySelectorAll) return;
    scope.querySelectorAll('[data-i18n]').forEach((node) => {
      node.textContent = t(node.getAttribute('data-i18n'));
    });
  }

  if (global.document) global.document.documentElement.lang = lang;

  global.I18N = { t, plural, setLang, getLang: () => lang, applyStatic, langs: LANGS, dict: DICT };
})(window);
