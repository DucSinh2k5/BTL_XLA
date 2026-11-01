// Switch views without navigating away from the document. In-memory pages
// preserve uploaded File objects and pending/completed inference until reload.
function createSessionNavigation(browser, doc, createPage) {
  const initialRoot = doc.getElementById("demo-page");
  const template = initialRoot.cloneNode(true);
  const pages = new Map();
  let active = null;
  let activeModel = null;
  const modelFromLocation = () => browser.location.pathname === "/mtacs" ? "mtacs" : "tacs";
  browser.history.scrollRestoration = "manual";

  function getPage(model) {
    if (!pages.has(model)) {
      const root = pages.size === 0 ? initialRoot : template.cloneNode(true);
      const view = createPage(root, model);
      view.scrollPosition = 0;
      for (const [id, target] of [["nav-tacs", "tacs"], ["nav-mtacs", "mtacs"]]) {
        root.querySelector("#" + id).addEventListener("click", (event) => {
          if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
          event.preventDefault();
          show(target, true);
        });
      }
      pages.set(model, view);
    }
    return pages.get(model);
  }

  function show(model, pushHistory = false) {
    if (!["tacs", "mtacs"].includes(model)) throw new Error("Unknown model page");
    if (model === activeModel) return;
    const next = getPage(model);
    if (active) {
      active.scrollPosition = browser.scrollY || 0;
      active.root.replaceWith(next.root);
    }
    active = next;
    activeModel = model;
    next.activate();
    if (pushHistory) browser.history.pushState({ sun397Model: model }, "", model === "mtacs" ? "/mtacs" : "/");
    browser.scrollTo({ top: next.scrollPosition, behavior: "auto" });
  }

  show(modelFromLocation());
  browser.history.replaceState({ ...(browser.history.state || {}), sun397Model: activeModel }, "", browser.location.href);
  browser.addEventListener("popstate", () => show(modelFromLocation()));
  return { show, currentModel: () => activeModel };
}

const navigation = createSessionNavigation(window, document, createModelPage);
