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
import { randomBytes } from "node:crypto";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { buildLocator } from "./lib/selectors.mjs";

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

const extraArgs = [
  "--disable-background-networking",
  "--disable-component-update",
  "--no-first-run",
  ...(process.env.CONTROL_OPENHANDS_BROWSER_ARGS || "").split(/\s+/).filter(Boolean),
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
    args: extraArgs,
  },
);

let dialogPolicy = "dismiss";
const events = [];
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
  const scope = scopeSelector ? locate(scopeSelector) : activePage.locator(":root");
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
      const offscreen = rect.right <= 0 || rect.left >= window.innerWidth;
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
    const response = await activePage.goto(url, { waitUntil: "domcontentloaded" });
    await activePage.waitForLoadState("networkidle", { timeout: 10_000 }).catch(() => {});
    return { url: activePage.url(), status: response?.status() };
  },
  async reload() {
    await activePage.reload({ waitUntil: "domcontentloaded" });
    await activePage.waitForLoadState("networkidle", { timeout: 10_000 }).catch(() => {});
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
    return { url: activePage.url(), title: await activePage.title(), viewport: activePage.viewportSize() };
  },
  async click({ selector, timeout, force, button }) {
    await locate(selector).click({ timeout, force, button });
    return { url: activePage.url() };
  },
  async dblclick({ selector, timeout }) {
    await locate(selector).dblclick({ timeout });
    return { url: activePage.url() };
  },
  async hover({ selector, timeout }) {
    await locate(selector).hover({ timeout });
    return {};
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
    if (selector) {
      await locate(selector).scrollIntoViewIfNeeded({ timeout });
    } else {
      await activePage.mouse.wheel(0, Number(by ?? 600));
    }
    return {};
  },
  async wait({ selector, state, timeout }) {
    await locate(selector).first().waitFor({ state: state ?? "visible", timeout });
    return { state: state ?? "visible" };
  },
  async "wait-url"({ pattern, timeout }) {
    await activePage.waitForURL(new RegExp(pattern), { timeout });
    return { url: activePage.url() };
  },
  async "wait-text"({ text, timeout }) {
    await activePage.getByText(text).first().waitFor({ state: "visible", timeout });
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
    const loc = selector ? locate(selector).first() : activePage.locator("body");
    const tree = await loc.ariaSnapshot();
    let saved;
    if (feature || name) {
      saved = evidencePath(feature, name, ".aria.txt");
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
      if (!match) throw new Error("viewport needs desktop|phone|narrow|tablet|WxH");
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
    const value = await activePage.evaluate(
      (source) => {
        // eslint-disable-next-line no-new-func
        const result = new Function(`return (${source});`)();
        return result instanceof Promise ? result : Promise.resolve(result);
      },
      expression,
    );
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
    return { closed: Number(index), active: context.pages().indexOf(activePage) };
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
  async downloads() {
    return { downloads: events.filter((e) => e.kind === "download").slice(-10) };
  },
  async storage({ keysOnly }) {
    // Lists localStorage keys (values only when explicitly requested) so that
    // persisted UI state can be checked without printing secrets by default.
    const data = await activePage.evaluate(() =>
      Object.fromEntries(
        Object.keys(window.localStorage).map((k) => [k, window.localStorage.getItem(k)]),
      ),
    );
    if (keysOnly !== false) return { keys: Object.keys(data).sort() };
    return { storage: data };
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
      await probe.waitForLoadState("networkidle", { timeout: 15_000 }).catch(() => {});
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
    res.end(JSON.stringify({ ok: false, error: `unknown command ${payload.cmd}` }));
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
        "Several elements match. Scope it (`testid=dialog >> role=button[name=\"Save\"]`) or add `>> nth=0` after checking `browser testids`.";
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
    JSON.stringify({ pid: process.pid, port, token, startedAt: new Date().toISOString() }),
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
