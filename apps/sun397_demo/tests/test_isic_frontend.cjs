// Browser-independent checks of production ISIC event handlers.
// Fixture masks/results stay in this test process and never enter the web app.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const directory = path.join(__dirname, '../isic');
const html = fs.readFileSync(path.join(directory, 'index.html'), 'utf8');
const source = fs.readFileSync(path.join(directory, 'assets/isic.js'), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));
const tags = [...html.matchAll(/<(\w+)\b([^>]*)>/g)];

class Element {
  constructor(attributes = '') {
    this.value = '';
    this.textContent = '';
    this.hidden = /\bhidden\b/.test(attributes);
    this.disabled = /\bdisabled\b/.test(attributes);
    this.dataset = {};
    for (const [, key, value] of attributes.matchAll(/data-(\w+)="([^"]*)"/g)) this.dataset[key] = value;
    this.classes = (attributes.match(/class="([^"]*)"/)?.[1] || '').split(/\s+/);
    this.attributes = new Map();
    this.listeners = new Map();
    this.files = [];
    this.children = [];
    this.draws = [];
    this.classList = {add() {}, remove() {}};
    this.width = 30; this.height = 20;
  }
  querySelector() { return this.child ||= new Element(); }
  addEventListener(type, callback) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(callback);
  }
  emit(type, overrides = {}) {
    const event = {button:0, defaultPrevented:false, preventDefault() {this.defaultPrevented = true;}, ...overrides};
    return {event, promise:Promise.all((this.listeners.get(type) || []).map(callback => callback(event)))};
  }
  click() { return this.emit('click'); }
  setAttribute(key, value) {this.attributes.set(key, value);}
  removeAttribute(key) {this.attributes.delete(key);}
  append(...children) {this.children.push(...children);}
  replaceChildren() {this.children = [];}
  getContext() {
    return {clearRect() {}, drawImage: image => this.draws.push(image.src),
      getImageData: () => ({data:new Uint8ClampedArray(this.width * this.height * 4).fill(255)}),
      putImageData() {}};
  }
}
function fixture({start='/isic', ready=true} = {}) {
  const document = {body:{dataset:{}}, createElement:() => new Element(), title:''};
  class Root {
    constructor() {
      this.dataset = {};
      this.all = tags.map(tag => {
        const node = new Element(tag[2]);
        node.id = tag[2].match(/\bid="([^"]+)"/)?.[1];
        return node;
      });
    }
    querySelector(selector) {
      const node = selector.startsWith('#') ? this.all.find(node => node.id === selector.slice(1))
        : this.all.find(node => node.classes.includes(selector.slice(1)));
      assert.ok(node, selector);
      return node;
    }
    querySelectorAll(selector) {
      return selector === '[data-view]' ? this.all.filter(node => node.dataset.view)
        : this.all.filter(node => node.classes.includes(selector.slice(1)));
    }
    cloneNode() {return new Root();}
    replaceWith(next) {assert.equal(document.root, this); document.root = next;}
  }
  document.root = new Root();
  document.getElementById = id => id === 'isic-page' ? document.root : document.root.querySelector('#' + id);
  const location = {pathname:start};
  const events = {};
  const browser = {location, scrollY:0, addEventListener:(name, handler) => events[name] = handler,
    scrollTo:({top}) => browser.scrollY = top,
    history:{pushState(_, _title, target) {location.pathname = target;}}};
  const requests = [], timers = [], revoked = [];
  let blob = 0;
  const response = data => ({ok:true, json:async () => data});
  const fetch = (url, options = {}) => {
    if (url.startsWith('/api/isic/status')) return Promise.resolve(response({model:url.split('=')[1],
      architecture:{steps:[]}, inference:{available:ready, can_predict:ready, code:ready?'ready':'bundle_missing'}}));
    if (url === '/api/isic/inspect') return Promise.resolve(response({filename:options.body.get('file').name,
      width:30, height:20, size_bytes:100, format:'PNG'}));
    assert.ok(url.startsWith('/api/isic/segment?model='));
    return new Promise(resolve => requests.push({model:url.split('=')[1], ...options, finish:data => resolve(response(data))}));
  };
  class LoadedImage {
    constructor() {this.naturalWidth = 30; this.naturalHeight = 20;}
    set src(value) {this.url = value; queueMicrotask(() => this.onload());}
    get src() {return this.url;}
  }
  const context = vm.createContext({document, window:browser, Image:LoadedImage, fetch, FormData, File,
    AbortController, Uint8ClampedArray, URL:{createObjectURL:() => 'blob:fixture-' + ++blob, revokeObjectURL:url => revoked.push(url)},
    setTimeout:callback => {timers.push(callback); return timers.length;}, clearTimeout() {}});
  vm.runInContext(source, context);
  return {document, browser, context, requests, timers, revoked,
    node:id => document.getElementById(id), switch:model => document.root.querySelector(model === 'tacs' ? '#isic-tacs' : '#isic-mtacs').click(),
    tab:view => document.root.querySelectorAll('[data-view]').find(node => node.dataset.view === view)};
}
async function upload(f, name) {
  const file = new File(['fixture'], name, {type:'image/png'});
  f.node('query-input').files = [file];
  await f.node('query-input').emit('change').promise;
  return file;
}
const dataURL = 'data:image/png;base64,aXNpYw==';
function result(model) {
  return {model, task:'binary_segmentation', prediction:{mask_url:dataURL, probability_url:dataURL, foreground_fraction:.25},
    comparison:{dice:.8, iou:.7}, ground_truth:{verified:true}, timing:{inference_seconds:2},
    evidence:{candidate:{image_id:model + '_context', image_url:dataURL, caption:'matching context caption', caption_valid:true},
      query_caption:{text:'query caption', valid:true}, fusion:{q_gate:.1, c_gate:.2},
      attention:{query_segmentor:{image_url:dataURL}, candidate_selector:{image_url:dataURL}}}};
}
test('model views retain independent uploads and completed/pending segmentation evidence', async () => {
  const f = fixture();
  await tick();
  const firstRoot = f.document.root;
  const file = await upload(f, 'tacs.png');
  const preview = f.node('query-preview').src;
  const pending = f.node('segment-button').click();
  assert.equal(f.requests.length, 1);
  f.switch('mtacs');
  await tick();
  const improvedRoot = f.document.root;
  assert.equal(f.node('query-input').files.length, 0);
  await upload(f, 'mtacs.png');
  f.requests[0].finish(result('tacs'));
  await pending.promise;
  await tick();
  assert.equal(f.document.body.dataset.model, 'mtacs');
  assert.equal(f.node('context-image').hidden, true);
  assert.equal(firstRoot.querySelector('#context-id').textContent, 'tacs_context');
  assert.equal(f.requests[0].signal.aborted, false);
  const mask = new File(['reference fixture'], 'reference.png', {type:'image/png'});
  f.node('reference-input').files = [mask];
  await f.node('reference-input').emit('change').promise;
  assert.equal(f.node('reference-file').hidden, false);
  const second = f.node('segment-button').click();
  assert.equal(f.requests[1].body.get('reference_mask'), mask);
  f.requests[1].finish(result('mtacs'));
  await second.promise;
  await tick();
  assert.equal(f.node('query-caption').textContent, 'query caption');
  assert.equal(f.node('context-caption').textContent, 'matching context caption');
  assert.equal(f.node('dice-value').textContent, '80.0%');
  assert.equal(f.node('iou-value').textContent, '70.0%');
  assert.equal(f.node('area-value').textContent, '25.0%');
  assert.equal(f.node('download-mask').download, 'isic-mtacs-mask.png');
  assert.equal(f.node('query-attention').hidden, false);
  f.tab('probability').click();
  await tick();
  assert.equal(f.node('probability-legend').hidden, false);
  f.switch('tacs');
  await tick();
  assert.equal(f.document.root, firstRoot);
  assert.equal(f.node('query-input').files[0], file);
  assert.equal(f.node('query-preview').src, preview);
  assert.equal(f.node('context-id').textContent, 'tacs_context');
  assert.equal(f.node('dice-value').textContent, '80.0%');
  f.switch('mtacs');
  assert.equal(f.document.root, improvedRoot);
  assert.equal(f.node('reference-name').textContent, 'reference.png');
  assert.equal(f.node('probability-legend').hidden, false);
  assert.equal(f.requests.length, 2);
  assert.equal(f.requests[0].body.get('file'), file);
  f.node('clear-image').click();
  f.switch('tacs');
  assert.equal(f.node('context-id').textContent, 'tacs_context');
});
test('missing models allow input/reference previews but never fabricate a result', async () => {
  const f = fixture({ready:false});
  await tick();
  await upload(f, 'input.png');
  assert.equal(f.node('segment-button').disabled, true);
  assert.equal(f.node('choose-reference').disabled, false);
  f.node('reference-input').files = [new File(['mask'], 'truth.png', {type:'image/png'})];
  await f.node('reference-input').emit('change').promise;
  f.tab('truth').click();
  await tick();
  assert.equal(f.node('result-canvas').hidden, false, 'Manual reference preview is distinct from prediction');
  assert.equal(f.node('dice-value').textContent, '—');
  assert.equal(f.node('query-attention').hidden, true);
  f.node('segment-button').click();
  assert.equal(f.requests.length, 0);
  assert.equal(f.node('download-mask').attributes.get('aria-disabled'), 'true');
  const fresh = fixture({ready:false});
  await tick();
  assert.equal(fresh.node('query-input').files.length, 0);
});
test('replacing a query aborts its request and ignores the stale mask response', async () => {
  const f = fixture();
  await tick();
  await upload(f, 'old.png');
  const pending = f.node('segment-button').click();
  await upload(f, 'new.png');
  assert.equal(f.requests[0].signal.aborted, true);
  f.requests[0].finish(result('tacs'));
  await pending.promise;
  assert.equal(f.node('image-name').textContent, 'new.png');
  assert.equal(f.node('context-id').textContent, 'Chưa chọn context');
  assert.equal(f.node('dice-value').textContent, '—');
});

