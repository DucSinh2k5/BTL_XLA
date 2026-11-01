// Each model owns its DOM, File object, requests, previews and results.
// A detached page keeps running; nothing is persisted across a real reload.
function createModelPage(root, modelId) {
  const elements = {
    badge: root.querySelector("#model-badge"),
    brandModel: root.querySelector("#brand-model"),
    brandMarkLetter: root.querySelector("#brand-mark-letter"),
    navTacs: root.querySelector("#nav-tacs"),
    navMtacs: root.querySelector("#nav-mtacs"),
    pageEyebrow: root.querySelector("#page-eyebrow"),
    pageHeading: root.querySelector("#page-heading"),
    pageDetail: root.querySelector("#page-detail"),
    architectureTitle: root.querySelector("#architecture-title"),
    architectureFooter: root.querySelector("#architecture-footer"),
    evidenceTitle: root.querySelector("#evidence-title"),
    lanes: root.querySelector("#architecture-lanes"),
    meta: root.querySelector("#pipeline-meta"),
    dropzone: root.querySelector("#dropzone"),
    input: root.querySelector("#file-input"),
    preview: root.querySelector("#preview-image"),
    previewLabel: root.querySelector("#preview-label"),
    previewFormat: root.querySelector("#preview-format"),
    choose: root.querySelector("#choose-button"),
    sample: root.querySelector("#sample-button"),
    clear: root.querySelector("#clear-button"),
    fileName: root.querySelector("#file-name"),
    dimensions: root.querySelector("#file-dimensions"),
    size: root.querySelector("#file-size"),
    fileStatus: root.querySelector("#file-status"),
    message: root.querySelector("#upload-message"),
    classify: root.querySelector("#classify-button"),
    resultHeading: root.querySelector("#result-heading"),
    resultDescription: root.querySelector("#result-description"),
    comparison: root.querySelector("#prediction-comparison"),
    predictionLabel: root.querySelector("#prediction-label"),
    predictionConfidence: root.querySelector("#prediction-confidence"),
    confidenceBar: root.querySelector("#confidence-bar"),
    resultGroundTruth: root.querySelector("#result-ground-truth"),
    groundTruthHeading: root.querySelector("#ground-truth-heading"),
    groundTruthNote: root.querySelector("#result-ground-truth-note"),
    comparisonStatus: root.querySelector("#comparison-status"),
    queryGroundTruth: root.querySelector("#query-ground-truth"),
    queryGroundTruthNote: root.querySelector("#query-ground-truth-note"),
    referenceClass: root.querySelector("#reference-class"),
    classifyReason: root.querySelector("#classify-reason"),
    evidence: root.querySelector("#evidence"),
    evidenceStatus: root.querySelector("#evidence-status"),
    queryEvidenceImage: root.querySelector("#query-evidence-image"),
    queryAttentionImage: root.querySelector("#query-attention-image"),
    queryAttentionEmpty: root.querySelector("#query-attention-empty"),
    queryAttentionNote: root.querySelector("#query-attention-note"),
    queryCaption: root.querySelector("#query-caption-text"),
    queryCaptionNote: root.querySelector("#query-caption-note"),
    candidateImage: root.querySelector("#candidate-image"),
    candidateImageEmpty: root.querySelector("#candidate-image-empty"),
    candidateImageId: root.querySelector("#candidate-image-id"),
    candidateClass: root.querySelector("#candidate-class-label"),
    candidateAttentionImage: root.querySelector("#candidate-attention-image"),
    candidateAttentionEmpty: root.querySelector("#candidate-attention-empty"),
    candidateAttentionNote: root.querySelector("#candidate-attention-note"),
    candidateCaption: root.querySelector("#candidate-caption-text"),
    candidateCaptionNote: root.querySelector("#candidate-caption-note"),
  };
  
  const page = modelId === "tacs" ? {
    title: "SUN397 | TACS gốc",
    brand: "TACS",
    eyebrow: "SUN397 / TACS GỐC",
    heading: "Phân loại cảnh với TACS",
    detail: "Selector học chọn một ảnh context hữu ích từ candidate pool; classifier xử lý cặp ảnh query và context.",
    intro: "Chọn ảnh rồi nhấn Dự đoán để xem nhãn, ảnh context được chọn và attention từ model.",
    evidence: "Cặp ảnh dùng để dự đoán",
    architecture: "Cấu trúc TACS gốc",
    footer: "TACS visual của dự án dùng DINOv3 ViT-L/16, ảnh 384 × 384 và một context. Heatmap biểu diễn visual self-attention.",
    waiting: "Model đang được nạp. Bạn có thể chọn ảnh trong lúc chờ.",
    missingCandidate: "Chưa có ảnh context được truy hồi",
    missingReason: "Cần checkpoint TACS và candidate pool để chọn ảnh context.",
    queryAttentionMissing: "Cần checkpoint và ảnh context được chọn để xuất classifier attention trên query.",
    candidateAttentionMissing: "Cần checkpoint và candidate pool để xuất selector attention trên candidate.",
  } : {
    title: "SUN397 | M-TACS + CLIP",
    brand: "M-TACS",
    eyebrow: "SUN397 / M-TACS + CLIP",
    heading: "Phân loại cảnh đa phương thức",
    detail: "Query ảnh và caption được đối chiếu với candidate pool bằng tín hiệu thị giác và vector văn bản CLIP.",
    intro: "Chọn ảnh để phân loại với context ảnh và caption. Caption hợp lệ mới được đưa vào nhánh text của model.",
    evidence: "Ảnh và caption dùng để dự đoán",
    architecture: "Cấu trúc M-TACS + CLIP",
    footer: "M-TACS full_both dùng caption query/candidate cho selector và fusion khi hợp lệ. Heatmap chỉ mô tả nhánh ảnh, không phải đóng góp của từng từ.",
    waiting: "Đang nạp model phân loại, Florence-2 và CLIP. Bạn có thể chọn ảnh trong lúc chờ.",
    missingCandidate: "Chưa có ảnh candidate được truy hồi",
    missingReason: "Cần checkpoint và candidate pool để chọn ảnh cùng caption tương ứng.",
    queryAttentionMissing: "Cần checkpoint và context để xuất classifier attention trên query.",
    candidateAttentionMissing: "Cần checkpoint và candidate pool để xuất attention của nhánh ảnh trong selector; heatmap này không biểu diễn điểm CLIP.",
  };
  
  const sampleUrl = "/assets/sample-library.png";
  let selectedFile = null;
  let previewUrl = null;
  let inspectController = null;
  let inferenceAvailable = false;
  let backendOnline = false;
  let classifyController = null;
  let statusTimer = null;
  let hasPrediction = false;
  let selectedGroundTruth = null;
  let classesLoaded = false;
  
  function applyPage() {
    elements.brandModel.textContent = page.brand;
    elements.brandMarkLetter.textContent = modelId === "tacs" ? "T" : "M";
    elements.pageEyebrow.textContent = page.eyebrow;
    elements.pageHeading.textContent = page.heading;
    elements.pageDetail.textContent = page.detail;
    elements.resultDescription.textContent = page.intro;
    elements.evidenceTitle.textContent = page.evidence;
    elements.architectureTitle.textContent = page.architecture;
    elements.architectureFooter.textContent = page.footer;
    elements.candidateImageEmpty.textContent = page.missingCandidate;
    elements.queryAttentionNote.textContent = page.queryAttentionMissing;
    elements.candidateAttentionNote.textContent = page.candidateAttentionMissing;
    for (const [link, active] of [[elements.navTacs, modelId === "tacs"], [elements.navMtacs, modelId === "mtacs"]]) {
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    }
  }
  
  function setMessage(message, isError = false) {
    elements.message.textContent = message;
    elements.message.classList.toggle("is-error", isError);
  }
  
  function setFileStatus(message, state) {
    elements.fileStatus.textContent = message;
    elements.fileStatus.dataset.state = state;
  }
  
  function updateClassifyButton() {
    elements.classify.disabled = !backendOnline || !inferenceAvailable || !selectedFile ||
      elements.fileStatus.dataset.state !== "ready" || classifyController !== null;
    elements.referenceClass.disabled = !classesLoaded || !selectedFile ||
      elements.fileStatus.dataset.state !== "ready" || selectedGroundTruth?.verified === true || classifyController !== null;
  }
  
  function resetPrediction() {
    hasPrediction = false;
    elements.comparison.hidden = true;
    elements.candidateClass.textContent = "Chưa chọn context";
    elements.evidence.hidden = true;
    elements.queryEvidenceImage.removeAttribute("src");
    renderAttention(null);
    elements.resultHeading.textContent = inferenceAvailable ? "Sẵn sàng phân loại" : "Đang chuẩn bị mô hình";
    elements.resultDescription.textContent = page.intro;
    elements.classifyReason.textContent = "Chọn một ảnh để xem trạng thái dự đoán và dữ liệu context.";
  }
  
  function renderQueryGroundTruth(groundTruth) {
    selectedGroundTruth = groundTruth;
    const verified = groundTruth?.verified === true;
    elements.queryGroundTruth.textContent = verified ? groundTruth.label : "Chưa xác định";
    elements.queryGroundTruthNote.textContent = verified
      ? `Nhãn SUN397 đã xác minh theo nội dung ảnh${groundTruth.split ? ` · tập ${groundTruth.split}` : ""}.`
      : groundTruth?.status === "ambiguous" ? "Ảnh khớp nhiều annotation khác nhãn; chưa thể xác định ground truth."
        : "Chưa tìm thấy annotation khớp nội dung ảnh. Tên file và kết quả dự đoán không được dùng làm nhãn thật.";
    elements.referenceClass.value = verified ? String(groundTruth.class_id) : "";
    updateClassifyButton();
  }
  
  function renderComparison(result) {
    const prediction = result.prediction;
    const truth = result.ground_truth;
    const state = result.comparison?.status || "unavailable";
    elements.comparison.hidden = false;
    elements.predictionLabel.textContent = prediction.label;
    elements.predictionConfidence.textContent = `${(prediction.confidence * 100).toFixed(2)}%`;
    elements.confidenceBar.value = prediction.confidence;
    elements.resultGroundTruth.textContent = truth?.label || "Chưa xác định";
    elements.groundTruthHeading.textContent = truth?.source === "user_provided" ? "Nhãn đối chiếu (người dùng)" : "Nhãn thật (ground truth)";
    elements.groundTruthNote.textContent = truth?.verified
      ? `Annotation SUN397 đã xác minh${truth.image_id ? ` · ${truth.image_id}` : ""}${truth.split ? ` · tập ${truth.split}` : ""}.`
      : truth?.source === "user_provided" ? "Nhãn do người dùng cung cấp, chưa được đối chiếu với annotation dataset."
        : "Chưa có ground truth nên chưa kết luận mô hình dự đoán đúng hay sai.";
    const messages = {correct: "Đúng · Dự đoán khớp nhãn thật", incorrect: "Sai · Dự đoán khác nhãn thật",
      matches_reference: "Khớp nhãn đối chiếu do bạn cung cấp", differs_from_reference: "Khác nhãn đối chiếu do bạn cung cấp",
      unavailable: "Chưa có nhãn thật để đối chiếu"};
    elements.comparisonStatus.textContent = messages[state] || messages.unavailable;
    elements.comparisonStatus.dataset.state = ["correct", "incorrect"].includes(state) ? state
      : state.includes("reference") ? "reference" : "unknown";
  }
  
  async function loadClasses() {
    try {
      const response = await fetch("/api/classes", { cache: "no-store" });
      if (!response.ok) throw new Error("Không tải được danh sách nhãn.");
      const data = await response.json();
      for (const item of data.classes) {
        const option = document.createElement("option");
        option.value = String(item.class_id);
        option.textContent = item.label;
        elements.referenceClass.append(option);
      }
      classesLoaded = data.classes.length > 0;
      if (selectedGroundTruth?.verified) elements.referenceClass.value = String(selectedGroundTruth.class_id);
      updateClassifyButton();
    } catch (error) {
      elements.queryGroundTruthNote.textContent = error.message;
    }
  }
  
  function isSafeEvidenceImageUrl(url, pathPrefix) {
    return typeof url === "string" &&
      (url.startsWith(pathPrefix) || /^data:image\/(jpeg|png|webp);base64,/i.test(url));
  }
  
  function renderAttention(attention) {
    for (const [kind, image, empty, note, missingMessage] of [
      ["query_classifier", elements.queryAttentionImage, elements.queryAttentionEmpty,
        elements.queryAttentionNote, page.queryAttentionMissing],
      ["candidate_selector", elements.candidateAttentionImage, elements.candidateAttentionEmpty,
        elements.candidateAttentionNote, page.candidateAttentionMissing],
    ]) {
      const url = attention?.[kind]?.image_url;
      const available = isSafeEvidenceImageUrl(url, "/api/attention/");
      image.hidden = !available;
      empty.hidden = available;
      if (available) image.src = url;
      else image.removeAttribute("src");
      note.textContent = available
        ? "Self-attention từ lần suy luận này; không phải giải thích nhân quả hoặc Grad-CAM."
        : missingMessage;
    }
  }
  
  function renderEvidence(evidence, available) {
    const queryCaption = evidence?.query_caption?.text;
    const candidate = evidence?.candidate;
    const candidateCaption = candidate?.caption;
    const candidateUrl = candidate?.image_url;
    const safeCandidateUrl = isSafeEvidenceImageUrl(candidateUrl, "/api/candidates/");
  
    elements.evidence.hidden = false;
    elements.evidenceStatus.dataset.state = available ? "ready" : "missing";
    elements.evidenceStatus.textContent = available ? "Đã truy hồi context" : "Chưa có dữ liệu suy luận";
    elements.queryEvidenceImage.src = previewUrl;
    renderAttention(evidence?.attention);
    const queryValid = evidence?.query_caption?.valid === true;
    const queryUnused = evidence?.query_caption?.status === "not_used";
    elements.queryCaption.textContent = queryUnused ? "Không dùng caption" : (queryCaption || "Chưa tạo caption query");
    elements.queryCaption.dataset.state = queryUnused ? "missing" : queryValid ? "ready" : queryCaption ? "masked" : "missing";
    elements.queryCaptionNote.textContent = queryUnused ? "TACS visual chỉ sử dụng ảnh."
      : queryCaption ? (queryValid ? "Caption hợp lệ, được sử dụng trong nhánh text."
        : "Caption bị bộ lọc loại; model không sử dụng vector text này.")
      : "Florence-2 sẽ tạo caption khi suy luận ảnh mới.";
    elements.candidateImage.hidden = !safeCandidateUrl;
    elements.candidateImageEmpty.hidden = safeCandidateUrl;
    if (safeCandidateUrl) elements.candidateImage.src = candidateUrl;
    else elements.candidateImage.removeAttribute("src");
    elements.candidateImageId.hidden = !candidate?.image_id;
    elements.candidateImageId.textContent = candidate?.image_id || "";
    elements.candidateClass.textContent = candidate?.label || candidate?.class_name || "Chưa có nhãn context";
    const candidateValid = candidate?.caption_valid === true;
    elements.candidateCaption.textContent = modelId === "tacs" ? "Không dùng caption" : (candidateCaption || "Chưa có caption candidate");
    elements.candidateCaption.dataset.state = candidateCaption ? candidateValid ? "ready" : "masked" : "missing";
    elements.candidateCaptionNote.textContent = candidateCaption
      ? candidateValid ? "Caption hợp lệ của chính ảnh candidate được chọn."
        : "Caption của candidate đã chọn bị mask; ảnh context vẫn được sử dụng."
      : modelId === "tacs" ? "TACS visual chỉ sử dụng ảnh context." : page.missingReason;
  }
  
  function renderArchitecture(architecture) {
    elements.meta.replaceChildren();
    for (const label of architecture.chips) {
      const chip = document.createElement("span");
      chip.className = "meta-chip";
      chip.textContent = label;
      elements.meta.append(chip);
    }
  
    elements.lanes.replaceChildren();
    for (const [index, lane] of architecture.lanes.entries()) {
      const section = document.createElement("section");
      section.className = "lane";
      const heading = document.createElement("div");
      heading.className = "lane-head";
      const number = document.createElement("span");
      number.className = "lane-index";
      number.textContent = String(index + 1).padStart(2, "0");
      const title = document.createElement("h3");
      title.textContent = lane.title;
      heading.append(number, title);
      const nodes = document.createElement("div");
      nodes.className = "lane-nodes";
      for (const node of lane.nodes) {
        const item = document.createElement("div");
        item.className = "pipeline-node";
        item.dataset.kind = node.kind;
        const label = document.createElement("strong");
        label.textContent = node.label;
        const detail = document.createElement("small");
        detail.textContent = node.detail;
        item.append(label, detail);
        nodes.append(item);
      }
      section.append(heading, nodes);
      elements.lanes.append(section);
    }
  }
  
  async function loadStatus() {
    clearTimeout(statusTimer);
    try {
      const response = await fetch(`/api/status?model=${modelId}`, { cache: "no-store" });
      if (!response.ok) throw new Error("Không đọc được trạng thái backend.");
      const status = await response.json();
      if (status.model !== modelId) throw new Error("Backend trả cấu hình của mô hình khác.");
      renderArchitecture(status.architecture);
      backendOnline = true;
      inferenceAvailable = status.inference.available === true;
      const loading = status.inference.code === "loading";
      elements.badge.dataset.state = inferenceAvailable ? "ready" : loading ? "loading" : "error";
      elements.badge.lastChild.textContent = inferenceAvailable ? "Sẵn sàng" : loading ? "Đang nạp model" : "Model chưa sẵn sàng";
      updateClassifyButton();
      if (!hasPrediction && !classifyController) {
        elements.resultHeading.textContent = inferenceAvailable ? "Sẵn sàng phân loại" : loading ? "Đang nạp mô hình" : "Không nạp được mô hình";
        elements.resultDescription.textContent = inferenceAvailable ? page.intro : status.inference.message;
        elements.classifyReason.textContent = inferenceAvailable
          ? "Chọn ảnh rồi nhấn Dự đoán."
          : status.inference.message;
      }
    } catch (error) {
      backendOnline = false;
      elements.badge.dataset.state = "error";
      elements.badge.lastChild.textContent = "Backend không khả dụng";
      elements.meta.textContent = "Không tải được cấu hình mô hình.";
      if (!hasPrediction && !classifyController) {
        elements.resultDescription.textContent = "Không thể kết nối backend để kiểm tra trạng thái mô hình.";
      }
      updateClassifyButton();
    } finally {
      statusTimer = setTimeout(loadStatus, inferenceAvailable ? 15000 : 2500);
    }
  }
  
  function resetImage() {
    if (inspectController) inspectController.abort();
    inspectController = null;
    if (classifyController) classifyController.abort();
    classifyController = null;
    selectedFile = null;
    selectedGroundTruth = null;
    elements.referenceClass.value = "";
    elements.queryGroundTruth.textContent = "Chưa chọn ảnh";
    elements.queryGroundTruthNote.textContent = "Nhãn SUN397 được đối chiếu khi có ảnh phù hợp.";
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    elements.input.value = "";
    elements.preview.src = sampleUrl;
    elements.preview.alt = "Ảnh minh họa thư viện, không thuộc bộ SUN397";
    elements.previewLabel.textContent = "ẢNH MINH HỌA";
    elements.previewFormat.textContent = "CHƯA CHỌN ẢNH";
    elements.fileName.textContent = "Chưa chọn ảnh";
    elements.dimensions.textContent = "—";
    elements.size.textContent = "—";
    setFileStatus("Chưa kiểm tra", "idle");
    setMessage("Ảnh không được lưu trên máy chủ sau khi kiểm tra.");
    elements.clear.hidden = true;
    resetPrediction();
    updateClassifyButton();
  }
  
  async function useImage(file, illustrative = false) {
    if (!file) return;
    if (inspectController) inspectController.abort();
    if (classifyController) classifyController.abort();
    classifyController = null;
    resetPrediction();
    selectedGroundTruth = null;
    elements.referenceClass.value = "";
    elements.queryGroundTruth.textContent = "Đang kiểm tra nhãn…";
    elements.queryGroundTruthNote.textContent = "Đang đối chiếu annotation SUN397…";
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const controller = new AbortController();
    inspectController = controller;
    selectedFile = file;
    previewUrl = URL.createObjectURL(file);
    elements.preview.src = previewUrl;
    elements.preview.alt = illustrative ? "Ảnh thư viện minh họa đã chọn" : `Ảnh đã chọn: ${file.name}`;
    elements.previewLabel.textContent = illustrative ? "ẢNH MINH HỌA" : "ẢNH ĐÃ CHỌN";
    elements.previewFormat.textContent = "ĐANG KIỂM TRA";
    elements.fileName.textContent = file.name;
    elements.dimensions.textContent = "—";
    elements.size.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
    setFileStatus("Đang kiểm tra", "loading");
    setMessage("Đang kiểm tra định dạng, kích thước và tính toàn vẹn của ảnh…");
    elements.clear.hidden = false;
    updateClassifyButton();
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch("/api/images/inspect", { method: "POST", body, signal: controller.signal });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Không thể kiểm tra ảnh.");
      if (inspectController !== controller) return;
      elements.fileName.textContent = data.filename;
      elements.dimensions.textContent = `${data.width} × ${data.height} px`;
      elements.size.textContent = `${(data.size_bytes / 1024 / 1024).toFixed(2)} MB`;
      elements.previewFormat.textContent = data.format;
      setFileStatus("Ảnh hợp lệ", "ready");
      renderQueryGroundTruth(data.ground_truth);
      setMessage(modelId === "tacs"
        ? "Ảnh đã được kiểm tra. Chưa chạy selector hoặc classifier TACS."
        : "Ảnh đã được kiểm tra. Chưa chạy caption hoặc mô hình phân loại.");
      elements.classifyReason.textContent = inferenceAvailable
        ? "Nhấn Dự đoán để xem kết quả và context được chọn."
        : page.waiting;
      updateClassifyButton();
    } catch (error) {
      if (error.name === "AbortError" || inspectController !== controller) return;
      elements.previewFormat.textContent = "KHÔNG HỢP LỆ";
      setFileStatus("Không hợp lệ", "error");
      elements.queryGroundTruth.textContent = "Chưa xác định";
      elements.queryGroundTruthNote.textContent = "Chỉ đối chiếu nhãn khi ảnh hợp lệ.";
      setMessage(error.message, true);
      updateClassifyButton();
    }
  }
  
  elements.choose.addEventListener("click", () => elements.input.click());
  elements.dropzone.addEventListener("click", () => elements.input.click());
  elements.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      elements.input.click();
    }
  });
  elements.input.addEventListener("change", () => useImage(elements.input.files[0]));
  elements.clear.addEventListener("click", resetImage);
  elements.sample.addEventListener("click", async () => {
    try {
      const response = await fetch(sampleUrl);
      if (!response.ok) throw new Error("Không thể mở ảnh minh họa.");
      const blob = await response.blob();
      useImage(new File([blob], "anh-minh-hoa-thu-vien.png", { type: "image/png" }), true);
    } catch (error) {
      setMessage(error.message, true);
    }
  });
  
  for (const name of ["dragenter", "dragover"]) {
    elements.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      elements.dropzone.classList.add("is-dragover");
    });
  }
  for (const name of ["dragleave", "drop"]) {
    elements.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      elements.dropzone.classList.remove("is-dragover");
    });
  }
  elements.dropzone.addEventListener("drop", (event) => useImage(event.dataTransfer.files[0]));
  elements.classify.addEventListener("click", async () => {
    if (!backendOnline || !inferenceAvailable || !selectedFile || elements.fileStatus.dataset.state !== "ready") return;
    const requestFile = selectedFile;
    const controller = new AbortController();
    classifyController = controller;
    const body = new FormData();
    body.append("file", requestFile);
    if (!selectedGroundTruth?.verified && elements.referenceClass.value !== "") {
      body.append("reference_class_id", elements.referenceClass.value);
    }
    updateClassifyButton();
    elements.classifyReason.textContent = modelId === "mtacs"
      ? "Đang tạo/kiểm tra caption, chọn context và phân loại…"
      : "Đang chọn context, phân loại và tạo heatmap…";
    try {
      const response = await fetch(`/api/classify?model=${modelId}`, { method: "POST", body, signal: controller.signal });
      const result = await response.json();
      if (classifyController !== controller || selectedFile !== requestFile) return;
      if (response.ok && result.model !== modelId) throw new Error("Backend trả kết quả của mô hình khác.");
      renderEvidence(result.evidence, response.ok && Boolean(result.prediction));
      if (response.ok && result.prediction) {
        hasPrediction = true;
        elements.resultHeading.textContent = "Đã có kết quả phân loại";
        renderComparison(result);
        const duration = result.timing?.inference_seconds;
        elements.resultDescription.textContent = typeof duration === "number" ? `Thời gian suy luận: ${duration.toFixed(1)} giây` : "Kết quả từ mô hình đã nạp.";
        elements.classifyReason.textContent = modelId === "tacs"
          ? "Ảnh context được chọn nằm ở phần dữ liệu bên dưới."
          : "Caption query và candidate context nằm ở phần dữ liệu bên dưới.";
      } else {
        elements.resultHeading.textContent = "Chưa thể dự đoán";
        elements.comparison.hidden = true;
        elements.resultDescription.textContent = result.message || (typeof result.detail === "string" ? result.detail : "Không thể xử lý yêu cầu.");
        elements.classifyReason.textContent = modelId === "tacs"
          ? "Chưa chọn context; không có kết quả giả."
          : "Chưa tạo caption hay chọn candidate; không có kết quả giả.";
      }
      if (root.isConnected) elements.evidence.scrollIntoView({
        behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
        block: "start",
      });
    } catch (error) {
      if (error.name === "AbortError" || classifyController !== controller) return;
      renderEvidence(null, false);
      elements.comparison.hidden = true;
      elements.resultHeading.textContent = "Chưa thể phân loại";
      elements.resultDescription.textContent = error.message;
      elements.classifyReason.textContent = "Không nhận được dữ liệu suy luận từ backend.";
    } finally {
      if (classifyController === controller) {
        classifyController = null;
        updateClassifyButton();
      }
    }
  });
  
  elements.input.value = "";
  elements.referenceClass.value = "";
  applyPage();
  loadStatus();
  loadClasses();
  elements.referenceClass.addEventListener("change", resetPrediction);
  return {
    root,
    activate() {
      document.body.dataset.mode = modelId;
      document.title = page.title;
      updateClassifyButton();
    },
    renderComparison, renderEvidence, resetPrediction,
  };
}
