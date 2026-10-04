#!/usr/bin/env node
// Long-lived browser owner for control-openhands.
//
// One daemon per verification run keeps a single Chromium profile and page
// alive between CLI invocations, records page errors / console errors /
// failed requests continuously, and executes one command per HTTP request.
// It listens on 127.0.0.1 only and requires the per-run token stored in
// <run>/private/browser.json (mode 0600). Start it with
// `control-openhands browser start`, never by hand.

import { createServer } from "node:http";
import { X509Certificate, createHash, randomBytes } from "node:crypto";
import {
  appendFileSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { buildLocator, toCss } from "./lib/selectors.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../../../..");
const require = createRequire(join(repoRoot, "package.json"));
const { chromium } = require("playwright");

const runDir = process.env.OH_VERIFY_RUN;
if (!runDir) {
  console.error("browser-daemon: OH_VERIFY_RUN is required");
  process.exit(2);
}
const privateDir = join(runDir, "private");
const evidenceDir = join(runDir, "evidence");
const baseUrl = new URL(process.env.OH_VERIFY_BASE_URL);
const eventsPath = join(privateDir, "browser-events.jsonl");
const VIEWPORTS = {
  desktop: { width: 1440, height: 1000 },
  phone: { width: 390, height: 844 },
  narrow: { width: 320, height: 700 },
  tablet: { width: 820, height: 1180 },
};

// Behind a TLS-intercepting proxy, pin its CA certificates by SPKI hash so the
// page can reach external origins the way an ordinary browser would there.
function spkiPins(files) {
  const pins = [];
  for (const file of files.split(/[,:]/).filter(Boolean)) {
    const pem = readFileSync(file, "utf8");
    for (const block of pem.match(
      /-----BEGIN CERTIFICATE-----[\s\S]+?-----END CERTIFICATE-----/g,
    ) ?? []) {
      const der = new X509Certificate(block).publicKey.export({
        type: "spki",
        format: "der",
      });
      pins.push(createHash("sha256").update(der).digest("base64"));
    }
  }
  return pins;
}

const trustPins = process.env.CONTROL_OPENHANDS_TRUST_CA
  ? spkiPins(process.env.CONTROL_OPENHANDS_TRUST_CA)
  : [];
const extraArgs = [
  "--disable-background-networking",
  "--disable-component-update",
  "--no-first-run",
  ...(trustPins.length
    ? [`--ignore-certificate-errors-spki-list=${trustPins.join(",")}`]
    : []),
  ...(process.env.CONTROL_OPENHANDS_BROWSER_ARGS || "")
    .split(/\s+/)
    .filter(Boolean),
];
const executablePath =
  process.env.CONTROL_OPENHANDS_BROWSER || process.env.QA_BROWSER_EXECUTABLE;

const context = await chromium.launchPersistentContext(
  join(privateDir, "browser-profile"),
  {
    headless: process.env.CONTROL_OPENHANDS_HEADED !== "1",
    executablePath: executablePath || undefined,
    viewport: VIEWPORTS.desktop,
    acceptDownloads: true,
    // Opt-in for sandboxes whose TLS interception presents leaf-only chains
    // that SPKI pins cannot match. Never set it in CI.
    ignoreHTTPSErrors:
      process.env.CONTROL_OPENHANDS_IGNORE_HTTPS_ERRORS === "1",
    args: extraArgs,
  },
);

// Instrumentation, not mocking: record media playback so sound features can
// be observed (`browser media`); playback itself still happens.
await context.addInitScript(() => {
  const original = HTMLMediaElement.prototype.play;
  window.__ohMediaPlays = [];
  HTMLMediaElement.prototype.play = function play(...args) {
    window.__ohMediaPlays.push({
      src: this.currentSrc || this.src || "",
      ts: new Date().toISOString(),
    });
    return original.apply(this, args);
  };
});

let dialogPolicy = "dismiss";
const events = [];
// Request log by origin (no query strings or bodies), for privacy and
// telemetry checks: which hosts did the page talk to?
const requests = [];
let markIndex = 0;
let activePage = context.pages()[0] ?? (await context.newPage());

function originOf(url) {
  try {
    return new URL(url).origin;
  } catch {
    return "";
  }
}

function record(kind, page, detail) {
  const entry = {
    ts: new Date().toISOString(),
    kind,
    page: page?.url?.() ?? "",
    appOrigin: false,
    ...detail,
  };
  const subject = entry.url || entry.page;
  entry.appOrigin = originOf(subject) === baseUrl.origin;
  events.push(entry);
  try {
    appendFileSync(eventsPath, `${JSON.stringify(entry)}\n`, { mode: 0o600 });
  } catch {
    // Event persistence is best effort; the in-memory copy still answers.
  }
}

function watch(page) {
  page.on("pageerror", (error) =>
    record("pageerror", page, {
      message: String(error?.message ?? error).slice(0, 2000),
    }),
  );
  page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning") {
      record(`console.${msg.type()}`, page, {
        message: msg.text().slice(0, 2000),
        url: msg.location()?.url || "",
      });
    }
  });
  page.on("request", (request) => {
    let origin = "";
    let path = "";
    try {
      const url = new URL(request.url());
      origin = url.origin;
      path = url.pathname.slice(0, 120);
    } catch {
      return;
    }
    if (!origin.startsWith("http")) return;
    requests.push({
      ts: new Date().toISOString(),
      origin,
      path,
      method: request.method(),
      type: request.resourceType(),
      app: origin === baseUrl.origin,
    });
    if (requests.length > 5000) requests.splice(0, 1000);
  });
  page.on("requestfailed", (request) => {
    const failure = request.failure()?.errorText ?? "failed";
    if (failure === "net::ERR_ABORTED") return;
    record("requestfailed", page, {
      url: request.url().slice(0, 500),
      method: request.method(),
      message: failure,
    });
  });
  page.on("response", (response) => {
    if (response.status() >= 400) {
      record("http-error", page, {
        url: response.url().slice(0, 500),
        method: response.request().method(),
        status: response.status(),
      });
    }
  });
  page.on("dialog", async (dialog) => {
    record("dialog", page, {
      message: `${dialog.type()}: ${dialog.message()}`.slice(0, 500),
      policy: dialogPolicy,
    });
    if (dialogPolicy === "accept") await dialog.accept();
    else await dialog.dismiss();
  });
  page.on("download", async (download) => {
    const dir = join(privateDir, "downloads");
    mkdirSync(dir, { recursive: true, mode: 0o700 });
    const target = join(
      dir,
      `${Date.now()}-${download.suggestedFilename().replace(/[^\w.-]+/g, "_")}`,
    );
    try {
      await download.saveAs(target);
      record("download", page, { path: target });
    } catch (error) {
      record("download", page, { message: String(error) });
    }
  });
}

for (const page of context.pages()) watch(page);
context.on("page", (page) => {
  watch(page);
  record("page-opened", page, { url: page.url() });
});

function locate(selector) {
  return buildLocator(activePage, selector);
}

function evidencePath(feature, name, ext) {
  const safeFeature = String(feature || "_misc").replace(/[^\w.-]+/g, "_");
  const safeName = String(name || `capture-${Date.now()}`).replace(
    /[^\w.-]+/g,
    "_",
  );
  const dir = join(evidenceDir, safeFeature);
  mkdirSync(dir, { recursive: true });
  return join(dir, `${safeName}${ext}`);
}

function assertAppUrl(target, allowExternal) {
  const url = new URL(target, baseUrl);
  if (!allowExternal && url.origin !== baseUrl.origin) {
    throw new Error(
      `Refusing to navigate outside this run (${baseUrl.origin}); pass --allow-external for a deliberate external check`,
    );
  }
  return url.toString();
}

async function failureShot() {
  try {
    const path = evidencePath("_failures", `${Date.now()}`, ".png");
    await activePage.screenshot({ path });
    return path;
  } catch {
    return undefined;
  }
}

async function collectTestids(scopeSelector, includeHidden) {
  const scope = scopeSelector
    ? locate(scopeSelector)
    : activePage.locator(":root");
  return scope.first().evaluate((root, wantHidden) => {
    const seen = new Map();
    const nodes = [root, ...root.querySelectorAll("[data-testid]")];
    for (const el of nodes) {
      const id = el.getAttribute && el.getAttribute("data-testid");
      if (!id) continue;
      const rect = el.getBoundingClientRect();
      const style = getComputedStyle(el);
      const visible =
        rect.width > 0 &&
        rect.height > 0 &&
        style.visibility !== "hidden" &&
        style.display !== "none";
      // Carousel slides and drawers parked beside the viewport are rendered
      // but not on screen; content below the fold still counts (scrollable).
      let offscreen = rect.right <= 0 || rect.left >= window.innerWidth;
      // Also off screen when an overflow-clipping ancestor (a slide rail,
      // a collapsed drawer) hides it entirely.
      for (let a = el.parentElement; a && !offscreen; a = a.parentElement) {
        const ov = getComputedStyle(a);
        if (/hidden|clip/.test(ov.overflowX + ov.overflowY)) {
          const r = a.getBoundingClientRect();
          offscreen =
            rect.right <= r.left ||
            rect.left >= r.right ||
            rect.bottom <= r.top ||
            rect.top >= r.bottom;
        }
      }
      if ((!visible || offscreen) && !wantHidden) continue;
      const label =
        el.getAttribute("aria-label") ||
        el.getAttribute("title") ||
        el.getAttribute("placeholder") ||
        (el.innerText || el.value || "").trim().split("\n")[0];
      const entry = seen.get(id) || {
        testid: id,
        tag: el.tagName.toLowerCase(),
        role: el.getAttribute("role") || undefined,
        label: (label || "").slice(0, 60) || undefined,
        count: 0,
        visible,
      };
      entry.count += 1;
      seen.set(id, entry);
    }
    return [...seen.values()];
  }, includeHidden);
}

const handlers = {
  async ping() {
    return { url: activePage.url(), pages: context.pages().length };
  },
  async goto({ target, allowExternal }) {
    const url = assertAppUrl(target ?? "/", allowExternal);
    const response = await activePage.goto(url, {
      waitUntil: "domcontentloaded",
    });
    await activePage
      .waitForLoadState("networkidle", { timeout: 10_000 })
      .catch(() => {});
    return { url: activePage.url(), status: response?.status() };
  },
  async reload() {
    await activePage.reload({ waitUntil: "domcontentloaded" });
    await activePage
      .waitForLoadState("networkidle", { timeout: 10_000 })
      .catch(() => {});
    return { url: activePage.url() };
  },
  async back() {
    await activePage.goBack({ waitUntil: "domcontentloaded" });
    return { url: activePage.url() };
  },
  async forward() {
    await activePage.goForward({ waitUntil: "domcontentloaded" });
    return { url: activePage.url() };
  },
  async url() {
    return {
      url: activePage.url(),
      title: await activePage.title(),
      viewport: activePage.viewportSize(),
    };
  },
  async click({
    selector,
    timeout,
    force,
    button,
    modifiers,
    position,
    expectUrl,
    observe,
    observeMs,
  }) {
    // Record transient states (labels such as "Saving...", skeletons) of the
    // observed elements while the click's effects play out: an in-page
    // MutationObserver when the selector is plain CSS/testid, else polling.
    let observer;
    let poller;
    const css = observe ? toCss(observe) : null;
    if (css) {
      observer = await activePage
        .evaluateHandle((sel) => {
          document.querySelectorAll(sel);
          const log = [];
          const snap = () => {
            const els = [...document.querySelectorAll(sel)];
            const entry = els.length
              ? els.map((e) =>
                  (e.innerText || e.value || "").trim().slice(0, 80),
                )
              : ["<absent>"];
            const key = JSON.stringify(entry);
            if (log[log.length - 1]?.key !== key) {
              log.push({ key, t: Math.round(performance.now()) });
            }
          };
          snap();
          const mo = new MutationObserver(snap);
          mo.observe(document.body, {
            subtree: true,
            childList: true,
            characterData: true,
            attributes: true,
          });
          return { log, mo };
        }, css)
        .catch(() => undefined);
    }
    if (observe && !observer) {
      const log = [];
      const t0 = Date.now();
      poller = { log, running: true };
      poller.done = (async () => {
        while (poller.running) {
          let entry;
          try {
            const loc = locate(observe);
            entry = (await loc.count())
              ? (await loc.allInnerTexts()).map((s) => s.trim().slice(0, 80))
              : ["<absent>"];
          } catch {
            entry = ["<unreadable>"];
          }
          const key = JSON.stringify(entry);
          if (log[log.length - 1]?.key !== key)
            log.push({ key, t: Date.now() - t0 });
          await new Promise((r) => setTimeout(r, 20));
        }
      })();
    }
    await locate(selector).click({
      timeout,
      force,
      button,
      modifiers,
      position,
    });
    if (expectUrl) {
      await activePage.waitForURL(new RegExp(expectUrl), { timeout });
    }
    let observed;
    if (observer || poller) {
      await activePage.waitForTimeout(Number(observeMs ?? 3000));
    }
    if (observer) {
      observed = await observer.evaluate(({ log, mo }) => {
        mo.disconnect();
        const start = log[0]?.t ?? 0;
        return log.map((l) => ({ ms: l.t - start, state: JSON.parse(l.key) }));
      });
    } else if (poller) {
      poller.running = false;
      await poller.done;
      observed = poller.log.map((l) => ({
        ms: l.t,
        state: JSON.parse(l.key),
      }));
    }
    return {
      url: activePage.url(),
      observed,
      observedBy: observe ? (observer ? "mutation" : "poll-20ms") : undefined,
    };
  },
  async dblclick({ selector, timeout, modifiers }) {
    await locate(selector).dblclick({ timeout, modifiers });
    return { url: activePage.url() };
  },
  async "mouse-click"({ x, y, button }) {
    await activePage.mouse.click(Number(x), Number(y), { button });
    return { url: activePage.url() };
  },
  async hover({ selector, timeout }) {
    await locate(selector).hover({ timeout });
    return {};
  },
  async tooltip({ selector, timeout }) {
    // Tooltips often ignore the first hover after a navigation: move away,
    // hover, then wait for role=tooltip.
    await activePage.mouse.move(0, 0);
    await locate(selector).hover({ timeout });
    const tip = activePage.getByRole("tooltip").first();
    await tip.waitFor({ state: "visible", timeout: timeout ?? 5000 });
    return { text: (await tip.innerText()).trim() };
  },
  async focus({ selector, timeout }) {
    await locate(selector).focus({ timeout });
    return {};
  },
  async fill({ selector, value, timeout }) {
    await locate(selector).fill(value, { timeout });
    return { length: value.length };
  },
  async type({ selector, value, timeout, delay }) {
    const loc = locate(selector);
    await loc.click({ timeout });
    await loc.pressSequentially(value, { delay: delay ?? 10 });
    return { length: value.length };
  },
  async press({ key, selector, timeout }) {
    if (selector) await locate(selector).press(key, { timeout });
    else await activePage.keyboard.press(key);
    return { url: activePage.url() };
  },
  async check({ selector, timeout }) {
    await locate(selector).check({ timeout });
    return { checked: true };
  },
  async uncheck({ selector, timeout }) {
    await locate(selector).uncheck({ timeout });
    return { checked: false };
  },
  async select({ selector, value, timeout }) {
    const selected = await locate(selector).selectOption(value, { timeout });
    return { selected };
  },
  async upload({ selector, files, timeout }) {
    await locate(selector).setInputFiles(files, { timeout });
    return { files };
  },
  async scroll({ selector, by, timeout }) {
    if (selector && by !== undefined) {
      // Scroll the element's nearest scrollable container (settings and
      // panels scroll inside a container, not the window).
      return locate(selector)
        .first()
        .evaluate((el, dy) => {
          let node = el;
          while (
            node &&
            !(
              node.scrollHeight > node.clientHeight &&
              /auto|scroll/.test(getComputedStyle(node).overflowY)
            )
          ) {
            node = node.parentElement;
          }
          const target = node || document.scrollingElement;
          target.scrollBy(0, dy);
          return {
            scrolled:
              target === document.scrollingElement ? "page" : "container",
            scrollTop: Math.round(target.scrollTop),
            scrollHeight: target.scrollHeight,
          };
        }, Number(by));
    }
    if (selector) {
      await locate(selector).scrollIntoViewIfNeeded({ timeout });
      return { scrolled: "into-view" };
    }
    await activePage.mouse.wheel(0, Number(by ?? 600));
    return { scrolled: "wheel at mouse position" };
  },
  async wait({ selector, state, timeout }) {
    await locate(selector)
      .first()
      .waitFor({ state: state ?? "visible", timeout });
    return { state: state ?? "visible" };
  },
  async "wait-url"({ pattern, timeout }) {
    await activePage.waitForURL(new RegExp(pattern), { timeout });
    return { url: activePage.url() };
  },
  async "wait-text"({ text, timeout }) {
    await activePage
      .getByText(text)
      .first()
      .waitFor({ state: "visible", timeout });
    return { text };
  },
  async text({ selector, timeout }) {
    const loc = locate(selector);
    const count = await loc.count();
    if (count > 1) {
      return { count, texts: (await loc.allInnerTexts()).map((t) => t.trim()) };
    }
    return { text: (await loc.innerText({ timeout })).trim() };
  },
  async value({ selector, timeout }) {
    return { value: await locate(selector).inputValue({ timeout }) };
  },
  async attr({ selector, name, timeout }) {
    return { [name]: await locate(selector).getAttribute(name, { timeout }) };
  },
  async count({ selector }) {
    return { count: await locate(selector).count() };
  },
  async visible({ selector }) {
    return { visible: await locate(selector).first().isVisible() };
  },
  async enabled({ selector, timeout }) {
    return { enabled: await locate(selector).isEnabled({ timeout }) };
  },
  async bbox({ selector, timeout }) {
    const loc = locate(selector);
    const box = await loc.boundingBox({ timeout });
    const viewport = activePage.viewportSize();
    const overflow = await loc.evaluate((el) => ({
      scrollWidth: el.scrollWidth,
      clientWidth: el.clientWidth,
      scrollHeight: el.scrollHeight,
      clientHeight: el.clientHeight,
    }));
    const docOverflowX = await activePage.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth,
    );
    return {
      box,
      viewport,
      insideViewport: Boolean(
        box &&
        viewport &&
        box.x >= 0 &&
        box.y >= 0 &&
        box.x + box.width <= viewport.width + 0.5 &&
        box.y + box.height <= viewport.height + 0.5,
      ),
      overflow,
      pageHorizontalOverflow: docOverflowX,
    };
  },
  async snapshot({ selector, maxLines, feature, name }) {
    const loc = selector
      ? locate(selector).first()
      : activePage.locator("body");
    const tree = await loc.ariaSnapshot();
    let saved;
    if (feature || name) {
      saved = evidencePath(
        feature,
        String(name ?? "").replace(/(\.aria)?\.txt$/, ""),
        ".aria.txt",
      );
      writeFileSync(saved, `${activePage.url()}\n${tree}\n`);
    }
    const lines = tree.split("\n");
    const limit = Number(maxLines ?? 120);
    return {
      url: activePage.url(),
      lines: lines.length,
      truncated: lines.length > limit,
      snapshot: lines.slice(0, limit).join("\n"),
      saved,
    };
  },
  async testids({ selector, includeHidden }) {
    const items = await collectTestids(selector, Boolean(includeHidden));
    return { url: activePage.url(), count: items.length, testids: items };
  },
  async screenshot({ feature, name, selector, fullPage }) {
    const path = evidencePath(feature, name, ".png");
    if (selector) await locate(selector).first().screenshot({ path });
    else await activePage.screenshot({ path, fullPage: Boolean(fullPage) });
    return { path, url: activePage.url(), viewport: activePage.viewportSize() };
  },
  async viewport({ size }) {
    const preset = VIEWPORTS[size];
    let next = preset;
    if (!next) {
      const match = /^(\d+)x(\d+)$/.exec(size ?? "");
      if (!match)
        throw new Error("viewport needs desktop|phone|narrow|tablet|WxH");
      next = { width: Number(match[1]), height: Number(match[2]) };
    }
    await activePage.setViewportSize(next);
    return { viewport: next };
  },
  async errors({ clear, all, appOnly }) {
    const slice = all ? events : events.slice(markIndex);
    const relevant = slice.filter(
      (e) => !["page-opened", "download", "dialog"].includes(e.kind),
    );
    const filtered = appOnly ? relevant.filter((e) => e.appOrigin) : relevant;
    const summary = {};
    for (const e of filtered) {
      const key = `${e.appOrigin ? "app" : "external"}:${e.kind}`;
      summary[key] = (summary[key] ?? 0) + 1;
    }
    const result = {
      since: all ? "daemon start" : "last clear",
      summary,
      pageErrors: filtered.filter((e) => e.kind === "pageerror").length,
      appErrors: filtered.filter((e) => e.appOrigin).length,
      events: filtered.slice(-40),
    };
    if (clear) markIndex = events.length;
    return result;
  },
  async events({ kinds, last }) {
    const wanted = kinds ? kinds.split(",") : null;
    const list = events.filter((e) => !wanted || wanted.includes(e.kind));
    return { events: list.slice(-Number(last ?? 20)) };
  },
  async eval({ expression }) {
    // Inspection only: the CLI documents that state must not be mutated here.
    const value = await activePage.evaluate((source) => {
      // eslint-disable-next-line no-new-func
      const result = new Function(`return (${source});`)();
      return result instanceof Promise ? result : Promise.resolve(result);
    }, expression);
    return { value };
  },
  async tabs() {
    return {
      active: context.pages().indexOf(activePage),
      pages: await Promise.all(
        context.pages().map(async (p, index) => ({
          index,
          url: p.url(),
          title: await p.title().catch(() => ""),
        })),
      ),
    };
  },
  async tab({ index }) {
    const page = context.pages()[Number(index)];
    if (!page) throw new Error(`No tab ${index}`);
    activePage = page;
    await page.bringToFront();
    return { active: Number(index), url: page.url() };
  },
  async "close-tab"({ index }) {
    const pages = context.pages();
    const page = pages[Number(index)];
    if (!page) throw new Error(`No tab ${index}`);
    if (pages.length === 1) throw new Error("Refusing to close the last tab");
    await page.close();
    if (page === activePage) activePage = context.pages()[0];
    return {
      closed: Number(index),
      active: context.pages().indexOf(activePage),
    };
  },
  async dialogs({ policy }) {
    if (policy) {
      if (!["accept", "dismiss"].includes(policy)) {
        throw new Error("dialog policy must be accept or dismiss");
      }
      dialogPolicy = policy;
    }
    return {
      policy: dialogPolicy,
      dialogs: events.filter((e) => e.kind === "dialog").slice(-10),
    };
  },
  async "pick-folder"({ path: want, timeout }) {
    // The Open Workspace folder browser has no path field; walk it with
    // folder-browser-up / folder-browser-entry-<name> until PATH is current.
    const label = activePage.getByTestId("folder-browser-current-path");
    const current = async () => {
      const text = (await label.innerText()).trim();
      return text.length > 1 ? text.replace(/\/+$/, "") : text;
    };
    await label.waitFor({ timeout });
    const deadline = Date.now() + (timeout ?? 30_000);
    let cur = await current();
    while (!cur && Date.now() < deadline) {
      await activePage.waitForTimeout(100);
      cur = await current();
    }
    const steps = [];
    for (let i = 0; i < 60; i += 1) {
      if (cur === want) return { path: cur, steps };
      const prefix = cur === "/" ? "/" : `${cur}/`;
      if (want.startsWith(prefix)) {
        const seg = want.slice(prefix.length).split("/")[0];
        await activePage
          .getByTestId(`folder-browser-entry-${seg}`)
          .click({ timeout: 15_000 });
        steps.push(seg);
      } else {
        await activePage.getByTestId("folder-browser-up").click({
          timeout: 15_000,
        });
        steps.push("..");
      }
      const before = cur;
      const t0 = Date.now();
      while ((cur = await current()) === before && Date.now() - t0 < 10_000) {
        await activePage.waitForTimeout(100);
      }
      if (cur === before) throw new Error(`Folder browser stayed at ${cur}`);
    }
    throw new Error(`Could not reach ${want}; stopped at ${cur}`);
  },
  async downloads() {
    return {
      downloads: events.filter((e) => e.kind === "download").slice(-10),
    };
  },
  async storage({ keysOnly, session }) {
    // Lists storage keys (values only when explicitly requested) so that
    // persisted UI state can be checked without printing secrets by default.
    const data = await activePage.evaluate((useSession) => {
      const store = useSession ? window.sessionStorage : window.localStorage;
      return Object.fromEntries(
        Object.keys(store).map((k) => [k, store.getItem(k)]),
      );
    }, Boolean(session));
    const area = session ? "sessionStorage" : "localStorage";
    if (keysOnly !== false) return { area, keys: Object.keys(data).sort() };
    return { area, storage: data };
  },
  async network({ clear, external, last }) {
    const list = external ? requests.filter((r) => !r.app) : requests;
    const byOrigin = {};
    for (const r of list) byOrigin[r.origin] = (byOrigin[r.origin] ?? 0) + 1;
    const result = {
      total: list.length,
      byOrigin,
      recent: list.slice(-Number(last ?? 15)),
    };
    if (clear) requests.length = 0;
    return result;
  },
  async toasts() {
    const toasts = await activePage.evaluate(() =>
      [...document.querySelectorAll('[role="status"], [role="alert"]')]
        .filter((el) => {
          const r = el.getBoundingClientRect();
          return r.width > 0 && r.height > 0 && el.innerText.trim();
        })
        .map((el) => ({
          role: el.getAttribute("role"),
          text: el.innerText.trim().slice(0, 300),
        })),
    );
    return { toasts };
  },
  async media({ clear }) {
    const plays = await activePage.evaluate((reset) => {
      const list = window.__ohMediaPlays ?? [];
      if (reset) window.__ohMediaPlays = [];
      return list;
    }, Boolean(clear));
    return { plays };
  },
  async uiprobe({ target, timeout }) {
    // Read-only health probe in a throwaway tab: the driving tab keeps its URL.
    const probe = await context.newPage();
    const errors = [];
    probe.on("pageerror", (e) => errors.push(String(e?.message ?? e)));
    try {
      await probe.goto(assertAppUrl(target ?? "/", false), {
        waitUntil: "domcontentloaded",
        timeout,
      });
      await probe
        .waitForLoadState("networkidle", { timeout: 15_000 })
        .catch(() => {});
      await probe.waitForTimeout(1500);
      const ids = await probe.evaluate(() =>
        [...document.querySelectorAll("[data-testid]")].map((e) =>
          e.getAttribute("data-testid"),
        ),
      );
      const markers = [
        "api-key-entry-screen",
        "first-run-onboarding-screen",
        "onboarding-modal",
        "telemetry-consent-form",
        "home-chat-launcher",
        "interactive-chat-box",
        "loading-spinner",
      ].filter((m) => ids.includes(m));
      return {
        url: probe.url(),
        title: await probe.title(),
        testids: ids.length,
        markers,
        pageErrors: errors,
      };
    } finally {
      await probe.close();
    }
  },
  async shutdown() {
    setTimeout(async () => {
      await context.close().catch(() => {});
      process.exit(0);
    }, 50);
    return { stopping: true };
  },
};

const token = randomBytes(24).toString("hex");
const server = createServer(async (req, res) => {
  if (req.method !== "POST" || req.headers["x-control-token"] !== token) {
    res.writeHead(403).end();
    return;
  }
  let body = "";
  for await (const chunk of req) body += chunk;
  let payload;
  try {
    payload = JSON.parse(body || "{}");
  } catch {
    res.writeHead(400).end(JSON.stringify({ ok: false, error: "bad json" }));
    return;
  }
  const handler = handlers[payload.cmd];
  res.setHeader("content-type", "application/json");
  if (!handler) {
    res.end(
      JSON.stringify({ ok: false, error: `unknown command ${payload.cmd}` }),
    );
    return;
  }
  try {
    const result = await handler(payload.args ?? {});
    res.end(JSON.stringify({ ok: true, ...result }));
  } catch (error) {
    const message = String(error?.message ?? error)
      .split("\n")
      .slice(0, 14)
      .join("\n");
    let hint;
    if (/strict mode violation/.test(message)) {
      hint =
        'Several elements match. Scope it (`testid=dialog >> role=button[name="Save"]`) or add `>> nth=0` after checking `browser testids`.';
    } else if (/Timeout/.test(message)) {
      hint =
        "Not found in time. Inspect the current page with `browser snapshot` or `browser testids`, then retry; check `browser errors` for crashes.";
    }
    res.end(
      JSON.stringify({
        ok: false,
        error: message,
        hint,
        url: activePage.url(),
        failureScreenshot: await failureShot(),
      }),
    );
  }
});

server.listen(0, "127.0.0.1", () => {
  const { port } = server.address();
  writeFileSync(
    join(privateDir, "browser.json"),
    JSON.stringify({
      pid: process.pid,
      port,
      token,
      startedAt: new Date().toISOString(),
    }),
    { mode: 0o600 },
  );
  console.log(`browser-daemon ready on 127.0.0.1:${port}`);
});

for (const signal of ["SIGTERM", "SIGINT"]) {
  process.on(signal, async () => {
    await context.close().catch(() => {});
    process.exit(0);
  });
}
