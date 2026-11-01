// Exercise production event handlers with isolated DOM/network fixtures.
// Fixtures never enter the running app or call a real model.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");
const directory = path.join(__dirname, "..");
const html = fs.readFileSync(path.join(directory, "index.html"), "utf8");
const pageSource = fs.readFileSync(path.join(directory, "assets/page.js"), "utf8");
const navigationSource = fs.readFileSync(path.join(directory, "assets/app.js"), "utf8");
const tags = [...html.matchAll(/<\w+\b([^>]*\bid="([^"]+)"[^>]*)>/g)];
const tick = () => new Promise(resolve => setImmediate(resolve));

class Element {
  constructor(attributes = "") {
    this.textContent = this.value = this.src = "";
    this.hidden = /\bhidden\b/.test(attributes);
    this.disabled = /\bdisabled\b/.test(attributes);
    this.dataset = {};
    this.files = [];
    this.children = [];
    this.lastChild = {};
    this.listeners = new Map();
    this.attributes = new Map();
    this.classList = { add() {}, remove() {}, toggle() {} };
    this.scrolls = 0;
  }
  addEventListener(type, handler) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(handler);
  }
  emit(type, overrides = {}) {
    const event = { button: 0, defaultPrevented: false,
      preventDefault() { this.defaultPrevented = true; }, ...overrides };
    const promise = Promise.all((this.listeners.get(type) || []).map(handler => handler(event)));
    return { event, promise };
  }
  click() { return this.emit("click"); }
  setAttribute(name, value) { this.attributes.set(name, value); }
  removeAttribute(name) { this.attributes.delete(name); if (name === "src") this.src = ""; }
  append(...children) { this.children.push(...children); }
  replaceChildren() { this.children = []; }
  scrollIntoView() { this.scrolls++; }
}

function fixture(start = "/") {
  const document = { body: { dataset: {} }, title: "", createElement: () => new Element() };
  class Root {
    constructor() {
      this.nodes = new Map(tags.filter(tag => tag[2] !== "demo-page").map(tag => [tag[2], new Element(tag[1])]));
    }
    querySelector(selector) {
      const element = this.nodes.get(selector.slice(1));
      assert.ok(element, `Missing HTML element: ${selector}`);
      return element;
    }
    cloneNode() { return new Root(); }
    get isConnected() { return document.root === this; }
    replaceWith(next) {
      assert.equal(document.root, this, "Only the attached page can be replaced");
      document.root = next;
    }
  }
  document.root = new Root();
  document.getElementById = id => id === "demo-page" ? document.root : document.root.querySelector("#" + id);
  const location = { pathname: start, href: "http://localhost" + start };
  const listeners = new Map();
  const browser = { location, scrollY: 0,
    addEventListener: (name, handler) => listeners.set(name, handler),
    scrollTo: ({ top }) => { browser.scrollY = top; } };
  const entries = [location.href];
  let cursor = 0;
  function visit(url) {
    const parsed = new URL(url, location.href);
    location.href = parsed.href;
    location.pathname = parsed.pathname;
  }
  browser.history = {
    state: null,
    pushState(state, _, url) {
      this.state = state;
      visit(url);
      entries.splice(++cursor, entries.length, location.href);
    },
    replaceState(state, _, url) { this.state = state; visit(url); entries[cursor] = location.href; },
    back() { if (cursor > 0) { visit(entries[--cursor]); listeners.get("popstate")(); } },
    forward() { if (cursor + 1 < entries.length) { visit(entries[++cursor]); listeners.get("popstate")(); } },
  };
  const requests = [];
  const revoked = [];
  const timers = new Map();
  let nextTimer = 0;
  let nextUrl = 0;
  let offline = false;
  const response = data => ({ ok: true, json: async () => data });
  function fetch(url, options = {}) {
    if (url.startsWith("/api/status")) {
      if (offline) return Promise.reject(new Error("offline"));
      return Promise.resolve(response({ model: url.split("=")[1], architecture: { chips: [], lanes: [] },
        inference: { available: true, code: "ready" } }));
    }
    if (url === "/api/classes") return Promise.resolve(response({ classes: [{class_id: 0, label: "a/abbey"}] }));
    if (url === "/api/images/inspect") return Promise.resolve(response({ filename: options.body.get("file").name,
      width: 384, height: 384, size_bytes: 100, format: "PNG", ground_truth: {verified: false} }));
    assert.ok(url.startsWith("/api/classify?model="), `Unexpected fetch: ${url}`);
    return new Promise(resolve => requests.push({ model: url.split("=")[1], ...options,
      finish: data => resolve(response(data)) }));
  }
  const context = vm.createContext({ document, window: browser, fetch, FormData, File, AbortController,
    URL: { createObjectURL: () => `blob:fixture-${++nextUrl}`, revokeObjectURL: url => revoked.push(url) },
    matchMedia: () => ({matches: true}),
    setTimeout: callback => { timers.set(++nextTimer, callback); return nextTimer; },
    clearTimeout: id => timers.delete(id),
  });
  vm.runInContext(pageSource, context);
  vm.runInContext(navigationSource, context);
  return { document, browser, requests, revoked, timers, context, entries,
    offline: () => { offline = true; }, node: id => document.getElementById(id) };
}

function prediction(model, label) {
  const image = `data:image/png;base64,${Buffer.from(model).toString("base64")}`;
  return { model, prediction: { label, class_id: 0, confidence: model === "tacs" ? 0.8 : 0.9 },
    ground_truth: { label: "a/abbey", source: "user_provided", verified: false },
    comparison: { status: "matches_reference" }, timing: { inference_seconds: 1 },
    evidence: { query_caption: { text: `${model} query caption`, valid: true },
      candidate: { image_id: `${model}/context.jpg`, image_url: image, label: `${model} context`,
        caption: `${model} context caption`, caption_valid: true },
      attention: { query_classifier: {image_url: image}, candidate_selector: {image_url: image} } } };
}

async function upload(f, name) {
  const file = new File(["image fixture"], name, {type: "image/png"});
  f.node("file-input").files = [file];
  await f.node("file-input").emit("change").promise;
  assert.equal(f.node("file-status").dataset.state, "ready");
  assert.equal(f.node("classify-button").disabled, false);
  return file;
}

test("each model retains its upload, reference, result, captions, heatmaps and scroll", async () => {
  const f = fixture();
  await tick();
  const original = f.document.root;
  const tacsFile = await upload(f, "tacs.png");
  const tacsUrl = f.node("preview-image").src;
  f.node("reference-class").value = "0";
  const first = f.node("classify-button").click();
  f.requests[0].finish(prediction("tacs", "TACS result"));
  await first.promise;
  f.browser.scrollY = 480;
  assert.equal(f.node("nav-mtacs").click().event.defaultPrevented, true);
  await tick();
  const improved = f.document.root;
  assert.notEqual(improved, original);
  assert.equal(f.browser.location.pathname, "/mtacs");
  assert.equal(f.document.body.dataset.mode, "mtacs");
  assert.equal(f.node("prediction-comparison").hidden, true);
  const mtacsFile = await upload(f, "mtacs.png");
  const mtacsUrl = f.node("preview-image").src;
  f.node("reference-class").value = "0";
  const second = f.node("classify-button").click();
  f.requests[1].finish(prediction("mtacs", "M-TACS result"));
  await second.promise;
  f.browser.scrollY = 720;
  f.node("nav-tacs").click();
  assert.equal(f.document.root, original);
  assert.equal(f.browser.scrollY, 480);
  assert.equal(f.node("preview-image").src, tacsUrl);
  assert.equal(f.node("file-input").files[0], tacsFile);
  assert.equal(f.node("prediction-label").textContent, "TACS result");
  assert.equal(f.node("prediction-confidence").textContent, "80.00%");
  assert.equal(f.node("reference-class").value, "0");
  assert.equal(f.node("result-ground-truth").textContent, "a/abbey");
  f.browser.history.back();
  assert.equal(f.document.root, improved);
  assert.equal(f.browser.scrollY, 720);
  assert.equal(f.node("preview-image").src, mtacsUrl);
  assert.equal(f.node("file-input").files[0], mtacsFile);
  assert.equal(f.node("prediction-label").textContent, "M-TACS result");
  assert.equal(f.node("prediction-confidence").textContent, "90.00%");
  assert.equal(f.node("query-caption-text").textContent, "mtacs query caption");
  assert.equal(f.node("candidate-caption-text").textContent, "mtacs context caption");
  assert.equal(f.node("candidate-class-label").textContent, "mtacs context");
  assert.equal(f.node("candidate-image-id").textContent, "mtacs/context.jpg");
  for (const id of ["candidate-image", "query-attention-image", "candidate-attention-image"])
    assert.equal(f.node(id).src, prediction("mtacs").evidence.candidate.image_url);
  f.browser.history.forward();
  assert.equal(f.document.root, original);
  f.node("nav-mtacs").click();
  const entries = f.entries.length;
  f.node("nav-mtacs").click();
  assert.equal(f.entries.length, entries, "Clicking the active tab does not reset or add history");
  assert.equal(f.requests.length, 2, "Navigation never reruns inference");
  assert.ok(f.requests.every(request => !request.signal.aborted));
  assert.deepEqual(f.requests.map(request => request.body.get("file")), [tacsFile, mtacsFile]);
  assert.deepEqual(f.requests.map(request => request.body.get("reference_class_id")), ["0", "0"]);
  assert.equal(f.revoked.length, 0, "Navigation must not revoke either preview");
  f.offline();
  for (const poll of [...f.timers.values()]) await poll();
  assert.equal(f.node("prediction-label").textContent, "M-TACS result");
  assert.match(f.node("result-description").textContent, /1\.0/);
  f.node("clear-button").click();
  assert.equal(f.node("prediction-comparison").hidden, true);
  f.node("nav-tacs").click();
  assert.equal(f.node("prediction-label").textContent, "TACS result", "Clear only resets its own page");
});

test("inference completes on its detached page without affecting the active model", async () => {
  const f = fixture("/mtacs");
  await tick();
  const mtacs = f.document.root;
  await upload(f, "background.png");
  const pending = f.node("classify-button").click();
  assert.equal(f.node("classify-button").disabled, true);
  f.node("nav-tacs").click();
  await tick();
  f.requests[0].finish(prediction("mtacs", "Background result"));
  await pending.promise;
  assert.equal(f.document.body.dataset.mode, "tacs");
  assert.equal(f.node("prediction-comparison").hidden, true);
  assert.equal(mtacs.querySelector("#prediction-label").textContent, "Background result");
  assert.equal(mtacs.querySelector("#evidence").scrolls, 0);
  assert.equal(f.requests[0].signal.aborted, false);
  f.node("nav-mtacs").click();
  assert.equal(f.document.root, mtacs);
  assert.equal(f.node("prediction-label").textContent, "Background result");
  assert.equal(f.node("classify-button").disabled, false);
});

test("modifier clicks keep native navigation; a real new document starts clean", async () => {
  const f = fixture();
  await tick();
  await upload(f, "temporary.png");
  for (const modifiers of [{ctrlKey:true}, {metaKey:true}, {shiftKey:true}, {altKey:true}, {button:1}]) {
    assert.equal(f.node("nav-mtacs").emit("click", modifiers).event.defaultPrevented, false);
    assert.equal(f.browser.location.pathname, "/");
  }
  const refreshed = fixture("/mtacs");
  await tick();
  assert.equal(refreshed.node("file-input").files.length, 0);
  assert.equal(refreshed.node("prediction-comparison").hidden, true);
  assert.equal(refreshed.node("classify-button").disabled, true);
  refreshed.node("nav-tacs").click();
  await tick();
  assert.equal(refreshed.node("file-input").files.length, 0);
  assert.equal(refreshed.node("prediction-comparison").hidden, true);
  assert.equal(refreshed.requests.length, 0);
});
