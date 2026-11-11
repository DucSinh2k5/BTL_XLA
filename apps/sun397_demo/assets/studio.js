// One document, four independent views. Uploaded Files, pending requests and
// results stay in memory until reload; changing task never reruns inference.
function createStudioNavigation(browser, doc, factories) {
  const routes = {
    '/': {task:'sun397', model:'tacs', root:'demo-page'},
    '/mtacs': {task:'sun397', model:'mtacs', root:'demo-page'},
    '/isic': {task:'isic', model:'tacs', root:'isic-page'},
    '/isic/mtacs': {task:'isic', model:'mtacs', root:'isic-page'},
  };
  const pathFor = (task, model) => task === 'isic'
    ? model === 'mtacs' ? '/isic/mtacs' : '/isic'
    : model === 'mtacs' ? '/mtacs' : '/';
  const initialPath = browser.location.pathname;
  const initialRoute = routes[initialPath];
  if (!initialRoute) throw new Error('Unknown studio route');
  const initialRoot = doc.getElementById(initialRoute.root);
  const templates = new Map([[initialRoute.task, initialRoot.cloneNode(true)]]);
  const templateRequests = new Map(), pages = new Map(), viewRequests = new Map();
  let active = null, activePath = null, revision = 0;
  browser.history.scrollRestoration = 'manual';

  const notification = doc.createElement('div');
  notification.className = 'studio-navigation-notice';
  notification.setAttribute('role', 'status');
  notification.hidden = true;
  doc.body.append(notification);

  async function templateFor(task) {
    if (templates.has(task)) return templates.get(task);
    if (!templateRequests.has(task)) {
      templateRequests.set(task, (async () => {
        const response = await fetch(pathFor(task, 'tacs'), {headers:{Accept:'text/html'}});
        if (!response.ok) throw new Error('Không tải được giao diện tác vụ. Hãy thử chuyển lại.');
        const document = new DOMParser().parseFromString(await response.text(), 'text/html');
        const node = document.getElementById(task === 'isic' ? 'isic-page' : 'demo-page');
        if (!node?.classList.contains('studio-page')) throw new Error('Giao diện đã thay đổi. Hãy tải lại trang sau khi lưu kết quả cần thiết.');
        templates.set(task, node);
        return node;
      })().finally(() => templateRequests.delete(task)));
    }
    return templateRequests.get(task);
  }

  async function viewFor(path) {
    if (pages.has(path)) return pages.get(path);
    if (viewRequests.has(path)) return viewRequests.get(path);
    const pending = buildView(path).finally(() => viewRequests.delete(path));
    viewRequests.set(path, pending);
    return pending;
  }

  async function buildView(path) {
    const route = routes[path];
    const root = path === initialPath ? initialRoot : (await templateFor(route.task)).cloneNode(true);
    const view = factories[route.task](root, route.model);
    view.scrollPosition = 0;
    for (const link of root.querySelectorAll('[data-studio-task]')) {
      link.setAttribute('href', pathFor(link.dataset.studioTask, route.model));
      if (link.dataset.studioTask === route.task) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
    for (const link of root.querySelectorAll('[data-studio-route], [data-studio-task]')) {
      link.addEventListener('click', event => {
        if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        const target = link.getAttribute('href');
        if (!routes[target]) return;
        event.preventDefault();
        show(target, true, true);
      });
    }
    // Hash links must not create model history entries (Back returns to a model).
    for (const link of root.querySelectorAll('a[href^="#"]')) link.addEventListener('click', event => {
      if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      const section = root.querySelector(link.getAttribute('href'));
      if (!section) return;
      event.preventDefault();
      section.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth', block:'start'});
    });
    pages.set(path, view);
    return view;
  }

  async function show(path, push = false, focus = false) {
    if (!routes[path]) return;
    const version = ++revision;
    // Even a click on the active task cancels a slower pending transition.
    if (active?.root) active.root.dataset.navigationLoading = 'false';
    if (path === activePath) return;
    if (active?.root) active.root.dataset.navigationLoading = 'true';
    try {
      const next = await viewFor(path);
      if (version !== revision) return;
      if (active) {
        active.scrollPosition = browser.scrollY || 0;
        active.root.replaceWith(next.root);
      }
      active = next;
      activePath = path;
      next.root.dataset.navigationLoading = 'false';
      next.activate();
      doc.body.dataset.task = routes[path].task;
      if (push) browser.history.pushState({studioPath:path}, '', path);
      else browser.history.replaceState({...(browser.history.state || {}),studioPath:path}, '', browser.location.href);
      notification.hidden = true;
      if (focus) next.root.querySelector('h1')?.focus({preventScroll:true});
      browser.scrollTo({top:next.scrollPosition, behavior:'auto'});
    } catch (error) {
      if (version !== revision) return;
      notification.textContent = error.message || 'Chưa chuyển được tác vụ. Ảnh và kết quả hiện tại vẫn được giữ.';
      notification.hidden = false;
      if (!push && activePath) browser.history.replaceState({studioPath:activePath}, '', activePath);
    } finally {
      if (version === revision && active?.root) active.root.dataset.navigationLoading = 'false';
    }
  }

  const ready = show(initialPath);
  browser.addEventListener('popstate', () => show(browser.location.pathname));
  return {show, ready, currentPath:() => activePath};
}

if (document.body.dataset.studio) {
  createStudioNavigation(window, document, {sun397:createModelPage, isic:createISICView});
}
