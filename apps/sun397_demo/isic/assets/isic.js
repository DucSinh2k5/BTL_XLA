// Each model owns its File objects, DOM, requests and results until a real reload.
function createISICView(root, model) {
  root.dataset.model = model;
  const $ = id => root.querySelector('#' + id);
  const textEnabled = model === 'mtacs';
  const title = textEnabled ? 'M-TACS' : 'TACS';
  const safeImage = value => typeof value === 'string' && /^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/.test(value);
  let selected = null, reference = null, preview = null, referenceURL = null;
  let metadata = null, output = null, view = 'overlay', controller = null, inspectController = null;
  let status = {can_predict: false}, timer = null, online = false, revision = 0, renderRevision = 0, imageInvalid = false;
  const imageCache = new Map();

  $('result-model').textContent = title;
  $('pipeline-title').textContent = textEnabled ? 'Bên trong M-TACS full_both' : 'Bên trong TACS visual';
  $('hero-description').textContent = textEnabled
    ? 'Kết hợp ảnh và caption của query lẫn context để khảo sát vùng phân đoạn.'
    : 'Khám phá cách TACS chọn một ảnh context để hỗ trợ xác định vùng tổn thương.';
  for (const element of root.querySelectorAll('.text-feature')) element.hidden = !textEnabled;
  for (const [id, active] of [['isic-tacs', !textEnabled], ['isic-mtacs', textEnabled]]) {
    if (active) $(id).setAttribute('aria-current', 'page'); else $(id).removeAttribute('aria-current');
  }
  $('query-input').value = $('reference-input').value = '';

  function message(heading, note, state = '') {
    root.querySelector('.run-bar').dataset.state = state;
    $('run-title').textContent = heading;
    $('run-note').textContent = note;
  }

  function updateActions() {
    $('segment-button').disabled = !selected || !metadata || !online || !status.can_predict || !!controller;
    $('choose-reference').disabled = !metadata || !!controller;
    $('reference-input').disabled = !!controller;
    $('clear-reference').disabled = !!controller;
    $('segment-button-label').textContent = controller ? 'Đang phân đoạn…' : 'Phân đoạn ảnh';
    $('opacity').disabled = !output || view !== 'overlay';
  }

  function clearResults() {
    output = null;
    renderRevision++;
    imageCache.clear();
    for (const id of ['dice-value', 'iou-value', 'area-value', 'time-value']) $(id).textContent = '—';
    for (const id of ['context-image', 'query-attention', 'context-attention']) { $(id).hidden = true; $(id).removeAttribute('src'); }
    for (const id of ['context-empty', 'query-attention-empty', 'context-attention-empty']) $(id).hidden = false;
    $('context-id').textContent = 'Chưa chọn context';
    $('context-note').textContent = 'Selector tìm một ảnh hữu ích từ pool cố định gồm 20% tập train.';
    $('query-caption').textContent = 'Caption sẽ được tạo từ ảnh bằng Florence-2.';
    $('query-caption-note').textContent = 'Vector CLIP hợp lệ tham gia selector và text fusion.';
    $('query-caption-state').textContent = $('context-caption-state').textContent = 'CHỜ SUY LUẬN';
    $('context-caption').textContent = 'Caption của chính ảnh được chọn sẽ xuất hiện ở đây.';
    $('gate-values').replaceChildren();
    $('download-mask').removeAttribute('href');
    $('download-mask').classList.add('disabled');
    $('download-mask').setAttribute('aria-disabled', 'true');
    $('comparison-note').textContent = 'Dice / IoU được tính khi có mask đối chiếu.';
    renderView();
    updateActions();
  }

  function image(url) {
    if (!imageCache.has(url)) imageCache.set(url, new Promise((resolve, reject) => {
      const element = new Image();
      element.onload = () => resolve(element);
      element.onerror = () => reject(new Error('Không đọc được ảnh kết quả.'));
      element.src = url;
    }));
    return imageCache.get(url);
  }

  async function renderView() {
    const version = ++renderRevision;
    const predicted = output?.prediction;
    const truth = output?.ground_truth;
    const url = view === 'truth' ? (safeImage(truth?.mask_url) ? truth.mask_url : referenceURL)
      : view === 'mask' ? predicted?.mask_url : view === 'probability' ? predicted?.probability_url : predicted?.mask_url;
    const available = (view === 'truth' && url === referenceURL && !!referenceURL) || safeImage(url);
    $('result-empty').hidden = !!available;
    $('result-canvas').hidden = !available;
    $('view-label').hidden = !available;
    $('probability-legend').hidden = !(available && view === 'probability');
    $('view-label').textContent = {overlay:'OVERLAY', mask:'BINARY MASK', probability:'XÁC SUẤT VÙNG', truth:'MASK ĐỐI CHIẾU'}[view];
    $('result-empty-note').textContent = view === 'truth'
      ? 'Thêm mask PNG hoặc chọn ảnh có annotation trong bundle.'
      : selected ? 'Kết quả thật sẽ xuất hiện sau khi chạy model.' : 'Chọn ảnh để bắt đầu thử nghiệm.';
    updateActions();
    if (!available) return;
    try {
      const mask = await image(url);
      const query = view === 'overlay' && preview ? await image(preview) : null;
      if (version !== renderRevision) return;
      const canvas = $('result-canvas');
      // Render at native dimensions; no browser resizing before mask alignment.
      canvas.width = mask.naturalWidth;
      canvas.height = mask.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (query) {
        ctx.drawImage(mask, 0, 0);
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
        for (let offset = 0; offset < data.data.length; offset += 4) {
          const foreground = data.data[offset] >= 128;
          data.data[offset] = 47; data.data[offset + 1] = 201; data.data[offset + 2] = 169;
          data.data[offset + 3] = foreground ? Math.round(Number($('opacity').value) * 2.55) : 0;
        }
        const layer = document.createElement('canvas');
        layer.width = canvas.width; layer.height = canvas.height;
        layer.getContext('2d').putImageData(data, 0, 0);
        ctx.drawImage(query, 0, 0, canvas.width, canvas.height);
        ctx.drawImage(layer, 0, 0);
      } else ctx.drawImage(mask, 0, 0);
    } catch (error) {
      if (version !== renderRevision) return;
      $('result-empty').hidden = false;
      $('result-canvas').hidden = true;
      $('result-empty-note').textContent = error.message;
    }
  }

  function pipeline(architecture) {
    $('pipeline').replaceChildren();
    architecture.steps.forEach((step, index) => {
      const node = document.createElement('div'); node.className = 'pipeline-step'; node.dataset.kind = step.kind;
      const number = document.createElement('span'); number.className = 'pipeline-number'; number.textContent = String(index + 1).padStart(2, '0');
      const heading = document.createElement('strong'); heading.textContent = step.title;
      const detail = document.createElement('small'); detail.textContent = step.detail;
      node.append(number, heading, detail); $('pipeline').append(node);
    });
  }

  async function loadStatus() {
    clearTimeout(timer);
    try {
      const response = await fetch(`/api/isic/status?model=${model}`, {cache:'no-store'});
      if (!response.ok) throw new Error('Không kết nối được API ISIC.');
      const data = await response.json();
      if (data.model !== model) throw new Error('Backend trả trạng thái mô hình khác.');
      online = true; status = data.inference;
      if (textEnabled && status.caption_experiment && !output) {
        $('query-caption-note').textContent = 'FirstValid v3 · CSV mới và CLIP cache mới; thử nghiệm với checkpoint segmentation hiện có.';
      }
      pipeline(data.architecture);
      const missing = ['bundle_missing', 'bundle_invalid'].includes(status.code);
      $('backend-status').dataset.state = missing ? 'missing' : status.available ? 'ready' : status.code === 'error' ? 'error' : 'loading';
      $('backend-status').querySelector('span').textContent = missing ? 'Chờ bộ model ISIC' : status.available ? 'Model sẵn sàng' : status.code === 'connected' ? 'Model đã kết nối' : 'Đang nạp model';
      if (!controller && !output && !inspectController && !imageInvalid) {
        message(missing ? 'Sẵn sàng xem ảnh · đang chờ checkpoint' : status.available ? 'Sẵn sàng phân đoạn' : 'Kết nối mô hình ISIC',
          missing ? 'Bộ model ISIC chưa được thêm vào máy chủ. Mask và attention sẽ để trống.'
            : selected ? 'Nhấn Phân đoạn ảnh để chạy model đã chọn.' : 'Chọn một ảnh ISIC để bắt đầu.', missing ? '' : 'ready');
      }
    } catch (error) {
      online = false;
      $('backend-status').dataset.state = 'error';
      $('backend-status').querySelector('span').textContent = 'Backend chưa kết nối';
      if (!controller && !output && !inspectController && !imageInvalid) message('Chưa kết nối API ISIC', 'Khởi động lại web để nạp phần segmentation mới.', 'error');
    } finally {
      updateActions();
      timer = setTimeout(loadStatus, status.code === 'loading' ? 2500 : 12000);
    }
  }

  function clearReference() {
    if (referenceURL) URL.revokeObjectURL(referenceURL);
    reference = referenceURL = null;
    $('reference-input').value = '';
    $('reference-file').hidden = true;
  }

  async function selectImage(file) {
    if (!file) return;
    const version = ++revision;
    if (controller) controller.abort();
    if (inspectController) inspectController.abort();
    controller = null;
    selected = file; metadata = null; imageInvalid = false;
    clearReference(); clearResults();
    if (preview) URL.revokeObjectURL(preview);
    preview = URL.createObjectURL(file);
    $('query-preview').src = preview; $('query-preview').hidden = false;
    $('preview-badge').hidden = false; $('upload-empty').hidden = true;
    $('image-name').textContent = file.name; $('clear-image').hidden = false;
    $('image-format').textContent = 'ĐANG KIỂM TRA';
    $('image-dimensions').textContent = 'Kích thước —'; $('image-size').textContent = 'Dung lượng —';
    message('Đang kiểm tra ảnh', 'Xác minh định dạng và kích thước ảnh query.');
    const request = new AbortController(); inspectController = request;
    const body = new FormData(); body.append('file', file);
    try {
      const response = await fetch('/api/isic/inspect', {method:'POST', body, signal:request.signal});
      const data = await response.json();
      if (version !== revision) return;
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Ảnh không hợp lệ.');
      metadata = data;
      $('image-format').textContent = data.format;
      $('image-name').textContent = data.filename;
      $('image-dimensions').textContent = `${data.width} × ${data.height} px`;
      $('image-size').textContent = `${(data.size_bytes / 1024 / 1024).toFixed(2)} MB`;
      message('Ảnh đã sẵn sàng', status.can_predict ? 'Bạn có thể thêm mask đối chiếu hoặc bắt đầu phân đoạn.' : 'Đã nhận ảnh. Cần bộ model ISIC để sinh kết quả thật.', status.can_predict ? 'ready' : '');
    } catch (error) {
      if (request.signal.aborted || version !== revision) return;
      imageInvalid = true;
      $('image-format').textContent = 'CHƯA HỢP LỆ';
      message('Chưa dùng được ảnh này', error.message, 'error');
    } finally { if (inspectController === request) inspectController = null; if (version === revision) updateActions(); }
  }

  function clearImage() {
    revision++;
    if (controller) controller.abort();
    if (inspectController) inspectController.abort();
    controller = null; selected = metadata = null; imageInvalid = false;
    if (preview) URL.revokeObjectURL(preview);
    preview = null; clearReference(); clearResults();
    $('query-input').value = ''; $('query-preview').hidden = true; $('query-preview').removeAttribute('src');
    $('upload-empty').hidden = false; $('preview-badge').hidden = true; $('clear-image').hidden = true;
    $('image-name').textContent = 'Chưa chọn ảnh'; $('image-format').textContent = 'JPG · PNG · WEBP';
    $('image-dimensions').textContent = 'Kích thước —'; $('image-size').textContent = 'Dung lượng —';
    message('Chọn một ảnh query', 'Ảnh và kết quả của mô hình còn lại vẫn được giữ.'); updateActions();
  }

  function showImage(id, emptyId, url) {
    const valid = safeImage(url);
    $(id).hidden = !valid; $(emptyId).hidden = valid;
    if (valid) $(id).src = url; else $(id).removeAttribute('src');
  }

  function renderResult(result) {
    output = result;
    const evidence = result.evidence || {};
    const candidate = evidence.candidate;
    showImage('context-image', 'context-empty', candidate?.image_url);
    $('context-id').textContent = candidate?.image_id || 'Chưa có context';
    $('context-note').textContent = candidate?.image_id ? 'Ảnh thật do selector chọn từ candidate pool train.' : 'Backend chưa trả ảnh context.';
    $('query-caption').textContent = evidence.query_caption?.text || 'Không có caption query.';
    $('query-caption-state').textContent = evidence.query_caption?.valid ? 'ĐANG SỬ DỤNG' : 'BỊ MASK / KHÔNG DÙNG';
    $('query-caption-note').textContent = evidence.query_caption?.valid ? 'Caption hợp lệ được encode bằng CLIP.' : 'Nhánh text này không đóng góp vào dự đoán.';
    if (result.caption_experiment) {
      $('query-caption-note').textContent += ' FirstValid v3 · thử caption mới với checkpoint hiện có; chưa train lại segmentation.';
    }
    $('context-caption').textContent = candidate?.caption || 'Không có caption context.';
    $('context-caption-state').textContent = candidate?.caption_valid ? 'ĐANG SỬ DỤNG' : 'BỊ MASK / KHÔNG DÙNG';
    showImage('query-attention', 'query-attention-empty', evidence.attention?.query_segmentor?.image_url);
    showImage('context-attention', 'context-attention-empty', evidence.attention?.candidate_selector?.image_url);
    const format = value => typeof value === 'number' && Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : '—';
    $('dice-value').textContent = format(result.comparison?.dice);
    $('iou-value').textContent = format(result.comparison?.iou);
    $('area-value').textContent = format(result.prediction.foreground_fraction);
    $('time-value').textContent = typeof result.timing?.inference_seconds === 'number' ? `${result.timing.inference_seconds.toFixed(1)}s` : '—';
    $('comparison-note').textContent = result.ground_truth?.verified ? 'Dice / IoU so với mask dataset đã xác minh theo pixel ảnh, ở kích thước gốc.'
      : result.ground_truth?.status === 'user_reference' ? 'Dice / IoU so với mask do bạn cung cấp; chưa xác minh annotation dataset.' : 'Chưa có mask đối chiếu để tính Dice / IoU.';
    if (safeImage(result.prediction.mask_url)) {
      $('download-mask').href = result.prediction.mask_url;
      $('download-mask').download = `isic-${model}-mask.png`;
      $('download-mask').classList.remove('disabled'); $('download-mask').setAttribute('aria-disabled', 'false');
    }
    $('gate-values').replaceChildren();
    if (textEnabled) for (const [key, label] of [['q_gate','Query'], ['c_gate','Context']]) {
      const value = evidence.fusion?.[key];
      if (typeof value === 'number') { const item = document.createElement('span'); item.textContent = `${label} gate ${value.toFixed(3)}`; $('gate-values').append(item); }
    }
    renderView();
  }

  $('choose-image').addEventListener('click', () => $('query-input').click());
  $('upload-zone').addEventListener('click', () => $('query-input').click());
  $('upload-zone').addEventListener('keydown', event => {if (['Enter',' '].includes(event.key)) {event.preventDefault(); $('query-input').click();}});
  $('query-input').addEventListener('change', () => selectImage($('query-input').files[0]));
  $('clear-image').addEventListener('click', clearImage);
  for (const type of ['dragenter','dragover']) $('upload-zone').addEventListener(type, event => {event.preventDefault(); $('upload-zone').classList.add('drag-over');});
  for (const type of ['dragleave','drop']) $('upload-zone').addEventListener(type, event => {event.preventDefault(); $('upload-zone').classList.remove('drag-over');});
  $('upload-zone').addEventListener('drop', event => selectImage(event.dataTransfer.files[0]));
  $('choose-reference').addEventListener('click', () => $('reference-input').click());
  $('reference-input').addEventListener('change', async () => {
    const file = $('reference-input').files[0];
    if (!file || !metadata || controller) return;
    const version = revision;
    const url = URL.createObjectURL(file);
    try {
      const mask = await image(url);
      if (version !== revision) {URL.revokeObjectURL(url); return;}
      if (file.type !== 'image/png' || file.size > 12 * 1024 * 1024 || mask.naturalWidth !== metadata.width || mask.naturalHeight !== metadata.height) throw new Error('Mask cần là PNG, cùng kích thước ảnh và không quá 12 MB.');
      clearReference(); reference = file; referenceURL = url;
      $('reference-name').textContent = file.name; $('reference-file').hidden = false;
      clearResults(); message('Đã thêm mask đối chiếu', 'Mask chỉ dùng để so sánh sau dự đoán; không đưa vào mô hình.');
    } catch (error) {URL.revokeObjectURL(url); message('Chưa dùng được mask', error.message, 'error');}
  });
  $('clear-reference').addEventListener('click', () => {clearReference(); clearResults();});
  for (const button of root.querySelectorAll('[data-view]')) button.addEventListener('click', () => {
    view = button.dataset.view;
    for (const tab of root.querySelectorAll('[data-view]')) tab.setAttribute('aria-selected', String(tab === button));
    renderView();
  });
  $('opacity').addEventListener('input', () => {$('opacity-value').textContent = $('opacity').value + '%'; renderView();});
  $('segment-button').addEventListener('click', async () => {
    if (!selected || !metadata || !online || !status.can_predict || controller) return;
    const request = new AbortController(); controller = request;
    const version = revision;
    const body = new FormData(); body.append('file', selected);
    if (reference) body.append('reference_mask', reference);
    updateActions(); message('Đang phân đoạn với ' + title, 'Đang chọn context, tạo mask và attention. Bạn có thể chuyển sang mô hình còn lại.', 'busy');
    try {
      const response = await fetch(`/api/isic/segment?model=${model}`, {method:'POST', body, signal:request.signal});
      const data = await response.json();
      if (version !== revision || controller !== request) return;
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : data.message || 'Chưa nhận được kết quả.');
      if (data.model !== model || data.task !== 'binary_segmentation' || !safeImage(data.prediction?.mask_url)) throw new Error('Backend chưa trả mask đúng mô hình.');
      renderResult(data); message('Đã có kết quả ' + title, 'Chuyển chế độ xem để đối chiếu mask, xác suất và ảnh gốc.', 'ready');
    } catch (error) {
      if (request.signal.aborted || version !== revision) return;
      message('Chưa phân đoạn được ảnh', error.message, 'error');
    } finally {if (controller === request) {controller = null; updateActions();}}
  });

  // Render architecture even while an older API is waiting to be restarted.
  pipeline({steps:[{title:'Ảnh query',detail:'DINOv3 frozen',kind:'visual'}, {title:'Candidate pool',detail:'20% tập train',kind:'visual'},
    {title:'Selector học',detail:textEnabled ? 'Ảnh + CLIP text' : 'Utility từ đặc trưng ảnh',kind:textEnabled?'text':'visual'},
    {title:'Context fusion',detail:'Cross-attention · 24 block',kind:'fusion'}, {title:'DPT nhẹ',detail:'Mask nhị phân',kind:'output'}]});
  loadStatus();
  return {root, activate() {document.body.dataset.model = model; document.title = `ISIC Studio · ${title}`; updateActions();}};
}

function createISICNavigation(browser, doc, createView) {
  const first = doc.getElementById('isic-page'), template = first.cloneNode(true), views = new Map();
  let active = null, activeModel = null;
  browser.history.scrollRestoration = 'manual';
  function show(model, push = false) {
    if (!['tacs','mtacs'].includes(model)) throw new Error('Unknown ISIC model');
    if (model === activeModel) return;
    if (!views.has(model)) {
      const root = views.size ? template.cloneNode(true) : first;
      const page = createView(root, model); page.scrollPosition = 0;
      for (const [id, target] of [['isic-tacs','tacs'], ['isic-mtacs','mtacs']]) root.querySelector('#' + id).addEventListener('click', event => {
        if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault(); show(target, true);
      });
      views.set(model, page);
    }
    const next = views.get(model);
    if (active) {active.scrollPosition = browser.scrollY || 0; active.root.replaceWith(next.root);}
    active = next; activeModel = model; active.activate();
    if (push) browser.history.pushState({isicModel:model}, '', model === 'mtacs' ? '/isic/mtacs' : '/isic');
    browser.scrollTo({top:next.scrollPosition, behavior:'auto'});
  }
  const fromURL = () => browser.location.pathname === '/isic/mtacs' ? 'mtacs' : 'tacs';
  browser.addEventListener('popstate', () => show(fromURL()));
  show(fromURL());
  return {show};
}
if (!document.body.dataset.studio) {
  createISICNavigation(window, document, createISICView);
}
