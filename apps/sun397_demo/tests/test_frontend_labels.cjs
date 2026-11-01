// DOM contract checks for both pages; no synthetic prediction is served by the app.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../assets/page.js"), "utf8");

for (const mode of ["tacs", "mtacs"]) {
  const nodes = new Map();
  function element(id) {
    if (!nodes.has(id)) nodes.set(id, { textContent: "", value: "", hidden: false, dataset: {},
      lastChild: {}, classList: { add() {}, remove() {}, toggle() {} },
      setAttribute() {}, removeAttribute() {}, addEventListener() {}, append() {}, replaceChildren() {},
      scrollIntoView() {} });
    return nodes.get(id);
  }
  const context = vm.createContext({
    document: { getElementById: element, body: {dataset: {}}, title: "", createElement: () => element(Symbol()) },
    root: { querySelector: selector => element(selector.slice(1)), isConnected: true },
    mode,
    window: { location: { pathname: mode === "mtacs" ? "/mtacs" : "/" } },
    fetch: () => new Promise(() => {}), setTimeout: () => 0, clearTimeout() {},
  });
  vm.runInContext(source, context);
  vm.runInContext("const view = createModelPage(root, mode); view.activate();", context);
  assert.equal(context.document.body.dataset.mode, mode);
  const result = { prediction: {class_id: 85, label: "c/cathedral/outdoor", confidence: 0.34247},
    ground_truth: {class_id: 0, label: "a/abbey", image_id: "a/abbey/query.jpg", split: "val", verified: true},
    comparison: {status: "incorrect"} };
  context.result = result;
  vm.runInContext("view.renderComparison(result)", context);
  assert.equal(element("prediction-label").textContent, "c/cathedral/outdoor");
  assert.equal(element("prediction-confidence").textContent, "34.25%");
  assert.equal(element("result-ground-truth").textContent, "a/abbey");
  assert.equal(element("comparison-status").dataset.state, "incorrect");
  assert.equal(element("confidence-bar").value, 0.34247);
  vm.runInContext('view.renderEvidence({candidate:{image_id:"p/palace/query.jpg",label:"p/palace"}},true)', context);
  assert.equal(element("candidate-class-label").textContent, "p/palace");
  vm.runInContext("view.resetPrediction()", context);
  assert.equal(element("prediction-comparison").hidden, true);
  assert.equal(element("candidate-class-label").textContent, "Chưa chọn context");

  context.result = {...result, ground_truth: {label: "a/abbey", source: "user_provided", verified: false},
    comparison: {status: "differs_from_reference"}};
  vm.runInContext("view.renderComparison(result)", context);
  assert.equal(element("comparison-status").dataset.state, "reference");
  assert.match(element("ground-truth-heading").textContent, /người dùng/);

  context.result = {...result, ground_truth: {label: null, verified: false}, comparison: {status: "unavailable"}};
  vm.runInContext("view.renderComparison(result)", context);
  assert.equal(element("result-ground-truth").textContent, "Chưa xác định");
  assert.equal(element("comparison-status").dataset.state, "unknown");
  assert.doesNotMatch(element("comparison-status").textContent, /^Đúng|^Sai/);
}
console.log("Frontend label/comparison checks passed for TACS and M-TACS.");
