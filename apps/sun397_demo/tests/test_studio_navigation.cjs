// Task orchestration checks. Model view handlers are exercised separately by
// test_frontend_navigation / test_isic_frontend; nothing calls a real model.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const source = fs.readFileSync(path.join(__dirname, '../assets/studio.js'), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));

function fixture(start = '/') {
  const created = [], fetches = [], events = new Map();
  const location = {pathname:start, href:'http://localhost' + start};
  const document = {body:{dataset:{}, append(node) {this.notice = node;}}, title:''};
  class Element {
    constructor(href, dataset = {}) {this.href = href; this.dataset = dataset; this.attrs = {}; this.events = []; this.focused = 0;}
    setAttribute(k, v) {this.attrs[k] = v; if (k === 'href') this.href = v;}
    removeAttribute(k) {delete this.attrs[k];}
    getAttribute(k) {return k === 'href' ? this.href : this.attrs[k];}
    addEventListener(_, callback) {this.events.push(callback);}
    click(overrides = {}) {
      const event = {defaultPrevented:false, button:0, preventDefault() {this.defaultPrevented = true;}, ...overrides};
      this.events.forEach(cb => cb(event)); return event;
    }
    focus() {this.focused++;}
    scrollIntoView() {this.scrolled = true;}
  }
  class Root {
    constructor(task) {
      this.task = task; this.dataset = {};
      this.classList = {contains:name => name === 'studio-page'};
      this.heading = new Element(); this.section = new Element();
      this.tasks = ['sun397','isic'].map(task => new Element(task === 'isic'?'/isic':'/',{studioTask:task}));
      this.models = task === 'isic' ? [new Element('/isic'), new Element('/isic/mtacs')] : [new Element('/'), new Element('/mtacs')];
      this.hash = new Element('#workspace');
    }
    cloneNode() {return new Root(this.task);}
    querySelectorAll(s) {
      if (s === '[data-studio-task]') return this.tasks;
      if (s === '[data-studio-route], [data-studio-task]') return [...this.tasks, ...this.models];
      if (s === 'a[href^="#"]') return [this.hash];
      throw new Error(s);
    }
    querySelector(s) {return s === 'h1' ? this.heading : s === '#workspace' ? this.section : null;}
    replaceWith(next) {assert.equal(document.root, this); document.root = next;}
  }
  document.root = new Root(start.startsWith('/isic') ? 'isic':'sun397');
  document.getElementById = () => document.root;
  document.createElement = () => new Element();
  const entries = [start]; let position = 0;
  function visit(value) {location.pathname = value; location.href = 'http://localhost' + value;}
  const browser = {location, scrollY:0, addEventListener:(name, cb) => events.set(name, cb), scrollTo:({top}) => browser.scrollY = top};
  browser.history = {state:null,
    pushState(state, _, value) {this.state = state; visit(value); entries.splice(++position, entries.length, value);},
    replaceState(state, _, value) {this.state = state; const pathname = new URL(value, location.href).pathname; visit(pathname); entries[position] = pathname;},
    back() {if (position > 0) {visit(entries[--position]); events.get('popstate')();}},
    forward() {if (position + 1 < entries.length) {visit(entries[++position]); events.get('popstate')();}},
  };
  function factory(task) {return (root, model) => {
    const view = {root, task, model, files:[], result:null, activate() {document.title = task + '/' + model;}};
    created.push(view); return view;
  };}
  const factories = {sun397:factory('sun397'), isic:factory('isic')};
  let pendingFetch = false, failingFetch = false;
  const fetch = url => {
    fetches.push(url);
    const response = () => ({ok:!failingFetch, text:async () => url === '/isic' ? 'isic':'sun397'});
    if (pendingFetch) return new Promise(resolve => {fetch.finish = () => resolve(response());});
    return Promise.resolve(response());
  };
  class DOMParser {parseFromString(task) {return {getElementById:() => new Root(task)};}}
  const context = vm.createContext({window:browser, document, factories, fetch, DOMParser, matchMedia:() => ({matches:true})});
  vm.runInContext(source, context);
  const navigation = vm.runInContext('createStudioNavigation(window, document, factories)', context);
  return {document, browser, navigation, created, fetches, entries, fetch,
    defer:() => pendingFetch = true, fail:() => failingFetch = true,
    task:name => document.root.tasks.find(link => link.dataset.studioTask === name),
    model: index => document.root.models[index]};
}

test('all four views retain identity, files, results, scroll and background completion across tasks', async () => {
  const f = fixture('/mtacs'); await f.navigation.ready;
  const sunM = f.created[0]; const file = new File(['query'], 'sun.png');
  sunM.files.push(file); sunM.result = {label:'abbey', confidence:.8};
  f.browser.scrollY = 420;
  assert.equal(f.task('isic').href, '/isic/mtacs');
  f.task('isic').click(); await tick();
  const isicM = f.created[1]; isicM.files.push(new File(['isic'], 'skin.png')); isicM.result = {mask:'actual returned mask'};
  assert.equal(f.navigation.currentPath(), '/isic/mtacs');
  assert.equal(f.task('sun397').href, '/mtacs');
  f.model(0).click(); await tick();
  const isicT = f.created[2]; f.browser.scrollY = 720;
  f.task('sun397').click(); await tick();
  const sunT = f.created[3];
  assert.equal(f.navigation.currentPath(), '/');
  sunM.result.caption = 'Background response retained';
  f.model(1).click(); await tick();
  assert.equal(f.document.root, sunM.root);
  assert.equal(f.browser.scrollY, 420);
  assert.equal(sunM.files[0], file);
  assert.equal(sunM.result.caption, 'Background response retained');
  f.browser.history.back(); await tick(); assert.equal(f.document.root, sunT.root);
  f.browser.history.back(); await tick(); assert.equal(f.document.root, isicT.root); assert.equal(f.browser.scrollY, 720);
  f.browser.history.back(); await tick(); assert.equal(f.document.root, isicM.root); assert.ok(isicM.result.mask);
  f.browser.history.forward(); await tick(); assert.equal(f.document.root, isicT.root);
  assert.equal(f.created.length, 4);
  assert.deepEqual(f.fetches, ['/isic'], 'Only the other task template is fetched once');
});

test('a slow template cannot override a newer choice and repeated clicks share the same view', async () => {
  const f = fixture(); await f.navigation.ready; f.defer();
  f.task('isic').click(); f.task('isic').click(); await tick();
  assert.equal(f.fetches.length, 1);
  f.task('sun397').click();
  f.fetch.finish(); await tick();
  assert.equal(f.navigation.currentPath(), '/');
  assert.equal(f.browser.location.pathname, '/');
  assert.equal(f.created.filter(view => view.task === 'isic').length, 1);
  f.task('isic').click(); await tick();
  assert.equal(f.navigation.currentPath(), '/isic');
  assert.equal(f.created.length, 2);
});

test('failed task loading keeps the current upload and view, with a retryable message', async () => {
  const f = fixture(); await f.navigation.ready;
  f.created[0].files.push('keep file'); f.fail();
  f.task('isic').click(); await tick();
  assert.equal(f.navigation.currentPath(), '/');
  assert.equal(f.document.root, f.created[0].root);
  assert.deepEqual(f.created[0].files, ['keep file']);
  assert.equal(f.document.body.notice.hidden, false);
  assert.match(f.document.body.notice.textContent, /Không tải được/);
});

test('direct ISIC entry, modifier clicks, section anchors and a fresh document behave normally', async () => {
  const f = fixture('/isic/mtacs'); await f.navigation.ready;
  for (const modifiers of [{ctrlKey:true},{metaKey:true},{shiftKey:true},{altKey:true},{button:1}]) {
    assert.equal(f.task('sun397').click(modifiers).defaultPrevented, false);
  }
  const count = f.entries.length;
  assert.equal(f.document.root.hash.click().defaultPrevented, true);
  assert.equal(f.document.root.section.scrolled, true);
  assert.equal(f.entries.length, count);
  f.task('sun397').click(); await tick();
  assert.equal(f.navigation.currentPath(), '/mtacs');
  assert.deepEqual(f.fetches, ['/']);
  const clean = fixture('/isic/mtacs'); await clean.navigation.ready;
  assert.equal(clean.created[0].files.length, 0);
  assert.equal(clean.created[0].result, null);
});
