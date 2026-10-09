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
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { createRequire } from "node:module";
import { basename, dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { redactStorage } from "./lib/redact-storage.mjs";
import { SECRET_KEY, SECRET_PATH, redactBody } from "./lib/network-bodies.mjs";
import { buildLocator, toCss } from "./lib/selectors.mjs";
import {
  RECORD_DEFAULTS,
  canvasSize,
  encodeRecording,
  findFfmpeg,
  frameDurations,
  videoToGif,
  withoutPauses,
} from "./lib/recording.mjs";

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

// Copy buttons await navigator.clipboard; grant it so `browser clipboard`
// can read what they wrote.
await context
  .grantPermissions(["clipboard-read", "clipboard-write"], {
    origin: baseUrl.origin,
  })
  .catch(() => {});

// Toasts vanish within seconds: keep a per-document history of the texts that
// appeared in status/alert regions (`browser toasts --history`).
await context.addInitScript(() => {
  window.__ohToasts = [];
  const last = new WeakMap();
  const scan = () => {
    for (const el of document.querySelectorAll(
      '[role="status"], [role="alert"]',
    )) {
      const text = (el.innerText || "").trim();
      if (!text || last.get(el) === text) continue;
      last.set(el, text);
      window.__ohToasts.push({
        ts: new Date().toISOString(),
        role: el.getAttribute("role"),
        text: text.slice(0, 300),
      });
    }
  };
  // Observe the document node itself: hydration can replace <html>.
  new MutationObserver(scan).observe(document, {
    subtree: true,
    childList: true,
    characterData: true,
  });
});

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
// Events survive a daemon restart (`restart`, `browser reset`): reload the
// ones recorded since the last `errors --clear` mark.
const events = [];
if (existsSync(eventsPath)) {
  const lines = readFileSync(eventsPath, "utf8").trim().split("\n");
  let start = 0;
  lines.forEach((line, i) => {
    if (line.includes('"kind":"mark"')) start = i + 1;
  });
  for (const line of lines.slice(start).slice(-3000)) {
    try {
      events.push(JSON.parse(line));
    } catch {
      // A torn line from a killed daemon; skip it.
    }
  }
}
// Request log by origin, for privacy and telemetry checks: which hosts did
// the page talk to? Query values that look like credentials are redacted;
// bodies are never kept.
const requests = [];
const requestEntries = new WeakMap();

function redactQuery(search) {
  if (!search) return "";
  const params = new URLSearchParams(search);
  for (const key of [...params.keys()]) {
    if (SECRET_KEY.test(key)) params.set(key, "<redacted>");
  }
  const text = params.toString();
  return text ? `?${text.slice(0, 200)}` : "";
}
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
    let query = "";
    try {
      const url = new URL(request.url());
      origin = url.origin;
      path = url.pathname.slice(0, 120);
      query = redactQuery(url.search);
    } catch {
      return;
    }
    if (!origin.startsWith("http") && !origin.startsWith("ws")) return;
    // The body of an app-origin write, redacted (see lib/network-bodies.mjs);
    // `network --bodies` shows it, so a recipe can prove what the page sent.
    let body;
    if (
      origin === baseUrl.origin &&
      /^(POST|PUT|PATCH|DELETE)$/.test(request.method())
    ) {
      try {
        body = redactBody(request.postData(), { all: SECRET_PATH.test(path) });
      } catch {
        body = undefined;
      }
    }
    const entry = {
      ts: new Date().toISOString(),
      origin,
      path,
      query: query || undefined,
      method: request.method(),
      type: request.resourceType(),
      app: origin === baseUrl.origin,
      body,
    };
    requestEntries.set(request, entry);
    requests.push(entry);
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
    const entry = requestEntries.get(response.request());
    if (entry) entry.status = response.status();
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

// Transient states (labels such as "Saving...", a loading row) of the
// elements an `--observe` selector names, while an action's effects play
// out: an in-page MutationObserver when the selector is plain CSS/testid,
// else polling every 20 ms. `startObserver` before the action,
// `finishObserver` after it.
async function startObserver(observe) {
  if (!observe) return undefined;
  let observer;
  let poller;
  const css = toCss(observe);
  if (css) {
    observer = await activePage
      .evaluateHandle((sel) => {
        document.querySelectorAll(sel);
        const log = [];
        const snap = () => {
          const els = [...document.querySelectorAll(sel)];
          // Text plus disabled/busy markers, so pending states show up.
          const entry = els.length
            ? els.map(
                (e) =>
                  (e.innerText || e.value || "").trim().slice(0, 80) +
                  (e.disabled || e.getAttribute("aria-disabled") === "true"
                    ? " [disabled]"
                    : "") +
                  (e.getAttribute("aria-busy") === "true" ? " [busy]" : ""),
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
  if (!observer) {
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
  return { observer, poller };
}

// The action failed: stop watching without a report, so the poller does not
// outlive the request and the page's MutationObserver is disconnected.
async function abandonObserver(watching) {
  if (!watching) return;
  const { observer, poller } = watching;
  if (observer)
    await observer.evaluate(({ mo }) => mo.disconnect()).catch(() => {});
  if (poller) {
    poller.running = false;
    await poller.done.catch(() => {});
  }
}

async function finishObserver(watching, observeMs) {
  if (!watching) return {};
  const { observer, poller } = watching;
  await activePage.waitForTimeout(Number(observeMs ?? 3000));
  let observed;
  if (observer) {
    observed = await observer.evaluate(({ log, mo }) => {
      mo.disconnect();
      const start = log[0]?.t ?? 0;
      return log.map((l) => ({ ms: l.t - start, state: JSON.parse(l.key) }));
    });
  } else {
    poller.running = false;
    await poller.done;
    observed = poller.log.map((l) => ({
      ms: l.t,
      state: JSON.parse(l.key),
    }));
  }
  return { observed, observedBy: observer ? "mutation" : "poll-20ms" };
}

// One scroll, as `browser scroll` asked for it: a horizontal rail, the
// nearest scrollable container of an element, an element into view, or the
// wheel at the pointer.
async function scrollOnce({ selector, by, x, timeout }) {
  if (selector && x !== undefined) {
    // Horizontal rails: scroll the nearest horizontally scrollable box.
    return locate(selector)
      .first()
      .evaluate((el, dx) => {
        let node = el;
        while (
          node &&
          !(
            node.scrollWidth > node.clientWidth &&
            /auto|scroll/.test(getComputedStyle(node).overflowX)
          )
        ) {
          node = node.parentElement;
        }
        const target = node || document.scrollingElement;
        target.scrollBy(dx, 0);
        return {
          scrolled: target === document.scrollingElement ? "page" : "container",
          scrollLeft: Math.round(target.scrollLeft),
          scrollWidth: target.scrollWidth,
        };
      }, Number(x));
  }
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
          scrolled: target === document.scrollingElement ? "page" : "container",
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
}

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
  // Evidence is never replaced: a repeated name gets -2, -3, ... and the
  // caller reports the path it actually wrote.
  let path = join(dir, `${safeName}${ext}`);
  for (let n = 2; existsSync(path); n += 1)
    path = join(dir, `${safeName}-${n}${ext}`);
  return path;
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

// One recording at a time (`browser record`). The loop captures whichever tab
// is active, so it follows `browser tab`; see lib/recording.mjs for why it
// polls instead of using a screencast. The text caret is hidden in each
// capture, as in screenshots: its blink would make an idle page look busy.
let recording = null;

async function captureLoop(rec) {
  while (!rec.stopping && Date.now() - rec.startedAt < rec.maxMs) {
    const t = Date.now();
    const page = activePage;
    if (rec.pausedAt) {
      await new Promise((r) => setTimeout(r, rec.intervalMs));
      continue;
    }
    try {
      const data = await page.screenshot({
        type: "jpeg",
        quality: 85,
        caret: "hide",
        timeout: 5_000,
      });
      rec.captures += 1;
      const hash = createHash("sha1").update(data).digest("hex");
      if (hash !== rec.lastHash) {
        rec.lastHash = hash;
        const file = join(
          rec.dir,
          `frame-${String(rec.frames.length).padStart(5, "0")}.jpg`,
        );
        writeFileSync(file, data);
        const size = page.viewportSize() ?? {};
        rec.frames.push({ file, t, width: size.width, height: size.height });
      }
    } catch {
      // A navigation or a closed tab; the next tick captures the active page.
      rec.misses += 1;
    }
    const wait = rec.intervalMs - (Date.now() - t);
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  }
  rec.endedAt = Date.now();
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
        (el.innerText || (el.type === "password" ? "" : el.value) || "")
          .trim()
          .split("\n")[0];
      const entry = seen.get(id) || {
        testid: id,
        tag: el.tagName.toLowerCase(),
        role: el.getAttribute("role") || undefined,
        label: (label || "").slice(0, 60),
        count: 0,
        domCount: document.querySelectorAll(`[data-testid="${CSS.escape(id)}"]`)
          .length,
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
    expectNewUrl,
    hoverFirst,
    observe,
    observeMs,
  }) {
    const before = activePage.url();
    const watching = await startObserver(observe);
    try {
      if (hoverFirst) {
        // Hover-driven controls re-render on pointerenter and swallow a click
        // that moves and presses at once.
        await locate(selector).hover({ timeout });
        await activePage.waitForTimeout(Number(hoverFirst) || 150);
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
      if (expectNewUrl) {
        const re = new RegExp(expectNewUrl);
        await activePage.waitForURL(
          (u) => u.toString() !== before && re.test(u.toString()),
          { timeout },
        );
      }
    } catch (error) {
      await abandonObserver(watching);
      throw error;
    }
    return {
      url: activePage.url(),
      ...(await finishObserver(watching, observeMs)),
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
    const target = locate(selector).first();
    await target.hover({ timeout });
    const tip = activePage.getByRole("tooltip").first();
    try {
      await tip.waitFor({ state: "visible", timeout: timeout ?? 5000 });
      return { source: "role=tooltip", text: (await tip.innerText()).trim() };
    } catch (error) {
      // Native title tooltips are not in the DOM; report the attribute.
      const title = await target.evaluate(
        (el) =>
          el.getAttribute("title") ||
          el.querySelector("[title]")?.getAttribute("title") ||
          el.closest("[title]")?.getAttribute("title") ||
          null,
      );
      if (title) return { source: "title attribute", text: title };
      throw error;
    }
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
  async scroll({ selector, by, x, timeout, observe, observeMs }) {
    // `--observe SEL` records what SEL shows while the scroll's effects
    // play out (a loading row, a fetched page), as `click --observe` does.
    const watching = await startObserver(observe);
    let result;
    try {
      result = await scrollOnce({ selector, by, x, timeout });
    } catch (error) {
      await abandonObserver(watching);
      throw error;
    }
    return { ...result, ...(await finishObserver(watching, observeMs)) };
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
  async value({ selector, timeout, reveal }) {
    const loc = locate(selector);
    const value = await loc.inputValue({ timeout });
    const type = await loc
      .first()
      .getAttribute("type")
      .catch(() => null);
    if (type === "password" && !reveal)
      return {
        value: value ? "********" : "",
        length: value.length,
        masked: true,
      };
    return { value };
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
    // ARIA snapshots include textbox values: mask what password fields hold
    // so keys never land in output or saved evidence.
    const secrets = await activePage.evaluate(() =>
      [...document.querySelectorAll('input[type="password"]')]
        .map((i) => i.value)
        .filter((v) => v && v.length >= 4),
    );
    let tree = await loc.ariaSnapshot();
    for (const s of secrets) tree = tree.split(s).join("********");
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
    const ambiguous = items.filter((i) => i.domCount > 1).map((i) => i.testid);
    return {
      url: activePage.url(),
      count: items.length,
      testids: items,
      hint: ambiguous.length
        ? `Several DOM nodes share ${ambiguous.slice(0, 5).join(", ")} (domCount > 1): scope with ' >> ' or add '>> visible'.`
        : undefined,
    };
  },
  async screenshot({ feature, name, selector, fullPage }) {
    const path = evidencePath(feature, name, ".png");
    if (selector) await locate(selector).first().screenshot({ path });
    else await activePage.screenshot({ path, fullPage: Boolean(fullPage) });
    return { path, url: activePage.url(), viewport: activePage.viewportSize() };
  },
  async clock({ offsetMs, system, fixed }) {
    // The page's clock, for states that depend on the browser's time
    // differing from the server's (a message stamped in the future). The
    // app's own timers keep running; install before the navigation whose
    // page should see the skewed time.
    const when = (value) =>
      /^[+-]\d+$/.test(String(value))
        ? Date.now() + Number(value)
        : new Date(value).getTime();
    if (fixed !== undefined) {
      await activePage.clock.setFixedTime(when(fixed));
      return { clock: "fixed", time: new Date(when(fixed)).toISOString() };
    }
    if (system !== undefined) {
      await activePage.clock.setSystemTime(when(system));
      return { clock: "system", time: new Date(when(system)).toISOString() };
    }
    const time = Date.now() + Number(offsetMs ?? 0);
    await activePage.clock.install({ time });
    return {
      clock: "installed",
      offsetMs: Number(offsetMs ?? 0),
      time: new Date(time).toISOString(),
    };
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
  async errors({ clear, all, appOnly, noWarnings }) {
    const slice = all ? events : events.slice(markIndex);
    const relevant = slice.filter(
      (e) =>
        !["page-opened", "download", "dialog", "mark"].includes(e.kind) &&
        !(noWarnings && e.kind === "console.warning"),
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
      appErrors: filtered.filter(
        (e) => e.appOrigin && e.kind !== "console.warning",
      ).length,
      warnings: filtered.filter((e) => e.kind === "console.warning").length,
      events: filtered.slice(-40),
    };
    if (clear) {
      markIndex = events.length;
      try {
        appendFileSync(
          eventsPath,
          `${JSON.stringify({ ts: new Date().toISOString(), kind: "mark" })}\n`,
        );
      } catch {
        // Best effort, as for record().
      }
    }
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
  async "wait-tab"({ pattern, timeout, excludeActive }) {
    // window.open pages appear a moment after the click. With
    // `--new` the page the daemon is on does not count, so a second tab at
    // the current URL can be waited for.
    const re = new RegExp(pattern);
    const skip = excludeActive ? activePage : null;
    const deadline = Date.now() + (timeout ?? 10_000);
    while (Date.now() < deadline) {
      const pages = context.pages();
      const index = pages.findIndex((p) => p !== skip && re.test(p.url()));
      if (index >= 0) return { index, url: pages[index].url() };
      await new Promise((r) => setTimeout(r, 100));
    }
    const open = context.pages().map((p, i) => `${i}: ${p.url()}`);
    throw new Error(
      `No tab matching ${pattern} within the timeout. Open tabs: ${open.join(" | ")} (a pop-up that stays about:blank could not load its URL)`,
    );
  },
  async "new-tab"({ target, allowExternal }) {
    // A plain new tab, as a user opening the app in a second tab: no
    // opener, no sessionStorage, the localStorage of the same profile.
    // Resolve the target first: a refused URL must not leave a blank tab
    // open and active.
    const url = target ? assertAppUrl(target, allowExternal) : undefined;
    const previous = activePage;
    const page = await context.newPage();
    activePage = page;
    if (url) {
      try {
        await page.goto(url, { waitUntil: "domcontentloaded" });
      } catch (error) {
        // A navigation that fails (the stack is down) closes the tab it
        // opened and leaves the daemon on the tab it was on.
        activePage = previous;
        await page.close().catch(() => {});
        throw error;
      }
      await page
        .waitForLoadState("networkidle", { timeout: 10_000 })
        .catch(() => {});
    }
    return { index: context.pages().indexOf(page), url: page.url() };
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
    return { area, storage: redactStorage(data) };
  },
  async network({ clear, external, last, filter, bodies }) {
    const re = filter ? new RegExp(filter) : null;
    const list = requests.filter(
      (r) =>
        (!external || !r.app) &&
        (!re || re.test(`${r.origin}${r.path}${r.query ?? ""}`)),
    );
    const byOrigin = {};
    for (const r of list) byOrigin[r.origin] = (byOrigin[r.origin] ?? 0) + 1;
    const result = {
      total: list.length,
      byOrigin,
      recent: list
        .slice(-Number(last ?? 15))
        .map(({ body, ...rest }) => (bodies ? { ...rest, body } : rest)),
    };
    if (clear) requests.length = 0;
    return result;
  },
  async toasts({ history, clear }) {
    const toasts = await activePage.evaluate(() =>
      [...document.querySelectorAll('[role="status"], [role="alert"]')]
        .filter((el) => {
          const r = el.getBoundingClientRect();
          return r.width > 0 && r.height > 0 && el.innerText.trim();
        })
        .map((el) => ({
          role: el.getAttribute("role"),
          text: el.innerText.trim().slice(0, 300),
          links: [...el.querySelectorAll("a[href]")].map((a) => ({
            text: a.innerText.trim(),
            href: a.getAttribute("href"),
          })),
        })),
    );
    if (!history && !clear) return { toasts };
    const seen = await activePage.evaluate((reset) => {
      const list = window.__ohToasts ?? [];
      if (reset) window.__ohToasts = [];
      return list;
    }, Boolean(clear));
    return { toasts, history: seen };
  },
  async clipboard({ write }) {
    if (write !== undefined) {
      await activePage.evaluate((t) => navigator.clipboard.writeText(t), write);
      return { written: write.length };
    }
    const text = await activePage.evaluate(() =>
      Promise.race([
        navigator.clipboard.readText(),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error("clipboard read timed out")), 5000),
        ),
      ]),
    );
    return { text };
  },
  async drag({ source, target, by, steps, position }) {
    const from = locate(source).first();
    if (target) {
      // HTML5 drag and drop with stepped pointer moves: React handlers that
      // set state in dragover need intermediate events before the drop.
      const from0 = await from.boundingBox();
      const box = await locate(target).first().boundingBox();
      if (!from0 || !box) throw new Error("source or target has no box");
      const tx = box.x + box.width / 2;
      const ty =
        position === "before"
          ? box.y + 3
          : position === "after"
            ? box.y + box.height - 3
            : box.y + box.height / 2;
      await activePage.mouse.move(
        from0.x + from0.width / 2,
        from0.y + from0.height / 2,
      );
      await activePage.mouse.down();
      await activePage.mouse.move(tx, ty, { steps: Number(steps ?? 20) });
      await activePage.waitForTimeout(150);
      await activePage.mouse.move(tx, ty + 1);
      await activePage.mouse.up();
      return { dragged: source, to: target, position: position ?? "center" };
    }
    // Pointer drag by an offset (resize grips, dividers).
    const box = await from.boundingBox();
    if (!box) throw new Error(`${source} has no box to drag`);
    const [dx, dy] = String(by ?? "0,0")
      .split(",")
      .map(Number);
    const x = box.x + box.width / 2;
    const y = box.y + box.height / 2;
    await activePage.mouse.move(x, y);
    await activePage.mouse.down();
    await activePage.mouse.move(x + dx, y + (dy || 0), {
      steps: Number(steps ?? 12),
    });
    await activePage.mouse.up();
    return { dragged: source, by: { dx, dy: dy || 0 } };
  },
  async "upload-via"({ trigger, files, timeout }) {
    // The real file chooser opened by a menu item or button.
    const [chooser] = await Promise.all([
      activePage.waitForEvent("filechooser", { timeout: timeout ?? 15_000 }),
      locate(trigger).first().click({ timeout }),
    ]);
    await chooser.setFiles(files);
    return { files, multiple: chooser.isMultiple() };
  },
  async "drop-files"({ selector, files, stage }) {
    // Files dragged in from the desktop: dragenter, dragover, drop with a
    // DataTransfer holding the files (--stage enter|over stops early).
    const MIME = {
      ".png": "image/png",
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".gif": "image/gif",
      ".txt": "text/plain",
      ".md": "text/markdown",
      ".json": "application/json",
      ".pdf": "application/pdf",
    };
    const payload = files.map((f) => ({
      name: basename(f),
      type: MIME[extname(f).toLowerCase()] ?? "application/octet-stream",
      b64: readFileSync(f).toString("base64"),
    }));
    const dataTransfer = await activePage.evaluateHandle((items) => {
      const dt = new DataTransfer();
      for (const it of items) {
        const bin = atob(it.b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
        dt.items.add(new File([bytes], it.name, { type: it.type }));
      }
      return dt;
    }, payload);
    const loc = locate(selector).first();
    const order = ["dragenter", "dragover", "drop"];
    const last = { enter: 0, over: 1 }[stage] ?? 2;
    for (const type of order.slice(0, last + 1)) {
      await loc.dispatchEvent(type, { dataTransfer });
    }
    return { files: payload.map((p) => p.name), stage: order[last] };
  },
  async paste({ selector, text, files }) {
    // A paste event carrying text or files, as the browser delivers it.
    const payload = (files ?? []).map((f) => ({
      name: basename(f),
      type: /\.png$/i.test(f)
        ? "image/png"
        : /\.jpe?g$/i.test(f)
          ? "image/jpeg"
          : "application/octet-stream",
      b64: readFileSync(f).toString("base64"),
    }));
    const loc = locate(selector).first();
    await loc.focus();
    const delivered = await loc.evaluate(
      (el, { text: t, items }) => {
        const dt = new DataTransfer();
        if (t !== undefined && t !== null) dt.setData("text/plain", t);
        for (const it of items) {
          const bin = atob(it.b64);
          const bytes = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
          dt.items.add(new File([bytes], it.name, { type: it.type }));
        }
        const event = new ClipboardEvent("paste", {
          clipboardData: dt,
          bubbles: true,
          cancelable: true,
        });
        el.dispatchEvent(event);
        return { defaultPrevented: event.defaultPrevented };
      },
      { text, items: payload },
    );
    return {
      pasted: { text: text?.length, files: payload.map((p) => p.name) },
      ...delivered,
    };
  },
  async choose({ selector, option, timeout }) {
    // Autocomplete/combobox: open, filter by the label, click the option.
    const input = locate(selector).first();
    await input.click({ timeout });
    const opt = activePage.getByRole("option", { name: option, exact: true });
    await input.fill(option).catch(() => {});
    try {
      await opt.first().waitFor({ state: "visible", timeout: 3000 });
    } catch {
      await input.fill("").catch(() => {});
      await input.press("ArrowDown").catch(() => {});
      await opt
        .first()
        .waitFor({ state: "visible", timeout: timeout ?? 10_000 });
    }
    await opt.first().click({ timeout });
    return {
      chosen: option,
      value: await input.inputValue().catch(() => undefined),
    };
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
  async "record-start"({ feature, name, fps, maxSeconds }) {
    if (recording)
      throw new Error(
        "A recording is already running: end it with `browser record stop`.",
      );
    const rate = Number(fps ?? RECORD_DEFAULTS.fps);
    if (!(rate >= 1 && rate <= 25))
      throw new Error("--fps must be between 1 and 25");
    const limit = Number(maxSeconds ?? RECORD_DEFAULTS.maxSeconds);
    if (!(limit > 0 && limit <= 3600))
      throw new Error("--max-seconds must be between 1 and 3600");
    const ffmpeg = findFfmpeg();
    if (!ffmpeg)
      throw new Error(
        "browser record needs ffmpeg: install it (MP4, --gif), or run `npx playwright install ffmpeg` (WebM). CONTROL_OPENHANDS_FFMPEG=/path picks one.",
      );
    const dir = join(privateDir, "recordings", String(Date.now()));
    mkdirSync(dir, { recursive: true });
    recording = {
      feature,
      name,
      dir,
      ffmpeg,
      fps: rate,
      intervalMs: 1000 / rate,
      maxMs: limit * 1000,
      startedAt: Date.now(),
      frames: [],
      cuts: [],
      captures: 0,
      misses: 0,
    };
    recording.loop = captureLoop(recording);
    return {
      recording: true,
      fps: rate,
      maxSeconds: limit,
      format: ffmpeg.format,
      ffmpeg: ffmpeg.path,
      url: activePage.url(),
    };
  },
  async "record-pause"() {
    if (!recording || recording.endedAt)
      throw new Error("No recording is running.");
    if (recording.pausedAt) throw new Error("The recording is already paused.");
    recording.pausedAt = Date.now();
    return { paused: true };
  },
  async "record-resume"() {
    if (!recording?.pausedAt) throw new Error("No recording is paused.");
    if (recording.endedAt)
      throw new Error(
        "The recording reached --max-seconds while paused; stop it.",
      );
    const ms = Date.now() - recording.pausedAt;
    recording.cuts.push({ at: recording.pausedAt, ms });
    recording.pausedAt = undefined;
    // The first capture after the cut is kept even if nothing changed.
    recording.lastHash = undefined;
    return { resumed: true, cutSeconds: Number((ms / 1000).toFixed(1)) };
  },
  async "record-status"() {
    if (!recording) return { recording: false };
    return {
      recording: !recording.endedAt,
      paused: Boolean(recording.pausedAt),
      capped: Boolean(recording.endedAt),
      seconds: ((recording.endedAt ?? Date.now()) - recording.startedAt) / 1000,
      captures: recording.captures,
      changes: recording.frames.length,
      misses: recording.misses,
    };
  },
  async "record-stop"({ gif, keepFrames, discard }) {
    const rec = recording;
    if (!rec)
      throw new Error(
        "No recording is running: start one with `browser record start`.",
      );
    rec.stopping = true;
    await rec.loop;
    recording = null;
    if (discard) {
      rmSync(rec.dir, { recursive: true, force: true });
      return { discarded: true, captures: rec.captures };
    }
    if (!rec.frames.length)
      throw new Error(
        `No frame was captured (${rec.misses} failed captures); frames dir ${rec.dir}`,
      );
    // Stopped while paused: the video ends where the pause began.
    const timeline = withoutPauses(
      rec.frames,
      rec.pausedAt ?? rec.endedAt,
      rec.cuts,
    );
    const durations = frameDurations(timeline.frames, timeline.endT);
    const { width, height } = canvasSize(rec.frames);
    const path = evidencePath(rec.feature, rec.name, `.${rec.ffmpeg.format}`);
    const written = await encodeRecording({
      ffmpeg: rec.ffmpeg,
      frames: rec.frames,
      durations,
      fps: rec.fps,
      width,
      height,
      out: path,
    }).catch((error) => {
      throw new Error(`${error.message} (frames kept in ${rec.dir})`);
    });
    const result = {
      path,
      seconds: Number((written / rec.fps).toFixed(1)),
      recordedSeconds: Number(
        ((rec.endedAt - rec.startedAt) / 1000).toFixed(1),
      ),
      size: `${width}x${height}`,
      fps: rec.fps,
      captures: rec.captures,
      changes: rec.frames.length,
      misses: rec.misses,
      capped: rec.endedAt - rec.startedAt >= rec.maxMs,
      cutSeconds: Number(
        (rec.cuts.reduce((sum, c) => sum + c.ms, 0) / 1000).toFixed(1),
      ),
      url: activePage.url(),
    };
    if (gif && !rec.ffmpeg.gif) {
      result.gifError = `${rec.ffmpeg.path} cannot write GIF; install ffmpeg for --gif`;
    } else if (gif) {
      const out = evidencePath(rec.feature, rec.name, ".gif");
      await videoToGif({
        ffmpeg: rec.ffmpeg,
        video: path,
        out,
        fps: Math.min(rec.fps, 12),
        width: RECORD_DEFAULTS.gifWidth,
      }).then(
        () => (result.gif = out),
        (error) => (result.gifError = error.message),
      );
    }
    if (keepFrames) result.framesDir = rec.dir;
    else rmSync(rec.dir, { recursive: true, force: true });
    return result;
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
    let hint = error?.hint;
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
        code: Number.isInteger(error?.code) ? error.code : undefined,
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
