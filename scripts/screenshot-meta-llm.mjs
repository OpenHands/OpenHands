import { chromium } from "playwright";
import fs from "node:fs";

const key = process.env.OPENHANDS_AUTOMATION_API_KEY ?? "";
const backend = {
  id: "default-local",
  name: "Local",
  host: "http://localhost:18000",
  apiKey: key,
  kind: "local",
  authMode: "api-key",
  connectionRevision: 0,
};

const browser = await chromium.launch({
  headless: true,
  channel: "chrome",
  args: ["--disable-web-security", "--disable-features=IsolateOrigins,site-per-process"],
});
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
await ctx.addInitScript((b, active) => {
  window.localStorage.setItem("openhands-backends", JSON.stringify([b]));
  window.localStorage.setItem("openhands-active-backend", JSON.stringify(active));
  window.localStorage.setItem("openhands-onboarded", "1");
}, backend, { backendId: "default-local" });

const page = await ctx.newPage();
const logs = [];
page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
page.on("requestfailed", (r) => logs.push(`[reqfail] ${r.url()} ${r.failure()?.errorText}`));
await page.goto("http://localhost:8080/settings/meta-llm", { waitUntil: "networkidle" });
// Wait for the toggle to render, then scroll it into view.
await page.getByTestId("meta-profile-run-at-conversation-start").waitFor({ state: "visible", timeout: 20000 }).catch(() => {});
await page.waitForTimeout(800);
await page.getByTestId("meta-profile-run-at-conversation-start").scrollIntoViewIfNeeded().catch(() => {});
await page.waitForTimeout(700);
// Crop to the settings panel only, excluding the conversation sidebar,
// so no private conversation data is published in the PR screenshot.
const toggleEl = await page.getByTestId("meta-profile-run-at-conversation-start").elementHandle().catch(() => null);
let clip = null;
if (toggleEl) {
  const tBox = await toggleEl.boundingBox();
  const settingsBox = await page.getByTestId("settings-screen").boundingBox().catch(() => null);
  if (tBox && settingsBox) {
    const y = Math.max(settingsBox.y, tBox.y - 180);
    clip = {
      x: settingsBox.x,
      y,
      width: Math.min(settingsBox.width, 720),
      height: Math.min(360, settingsBox.y + settingsBox.height - y),
    };
  }
}
await page.screenshot({ path: "pr-screenshot-meta-llm.png", clip: clip ?? undefined });
console.log("url:", page.url());
const bodyText = await page.evaluate(() => document.body.innerText);
fs.writeFileSync("/tmp/page-text.txt", bodyText);
fs.writeFileSync("/tmp/page-logs.txt", logs.join("\n"));
const testids = await page.evaluate(() =>
  Array.from(document.querySelectorAll("[data-testid]")).map((e) => e.getAttribute("data-testid")),
);
fs.writeFileSync("/tmp/testids.txt", testids.join("\n"));
const ls = await page.evaluate(() => ({
  backends: window.localStorage.getItem("openhands-backends"),
  active: window.localStorage.getItem("openhands-active-backend"),
  onboarded: window.localStorage.getItem("openhands-onboarded"),
  health: window.localStorage.getItem("openhands-backend-health"),
}));
fs.writeFileSync("/tmp/ls.txt", JSON.stringify(ls, null, 2));
// Probe cross-origin /server_info directly to see CORS / status.
const probe = await page.evaluate(async (host) => {
  try {
    const res = await fetch(`${host}/server_info`, { headers: { "X-Session-API-Key": "x" } });
    const text = await res.text();
    return { ok: res.ok, status: res.status, len: text.length, head: text.slice(0, 120) };
  } catch (e) {
    return { error: String(e) };
  }
}, "http://localhost:18000");
fs.writeFileSync("/tmp/probe.txt", JSON.stringify(probe, null, 2));
console.log("saved pr-screenshot-meta-llm.png");
await browser.close();
fs.writeFileSync("/tmp/screenshot-done", "1");
