// Screenshots of the mock-API Canvas after the dead-code cleanup.
// Start the frontend first:
//   VITE_MOCK_API=true npx react-router dev --port 5199 --strictPort
// then run: SHOT_PREFIX=after- node .pr/dead-code-cleanup/evidence.mjs
// (SHOT_PREFIX=before- against a main checkout for the baseline).
// Fails if any page shows a raw translation key (e.g. SETTINGS$FOO).
import { chromium } from "playwright";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const BASE = process.env.BASE_URL ?? "http://127.0.0.1:5199";
const PREFIX = process.env.SHOT_PREFIX ?? "";
const OUT = dirname(fileURLToPath(import.meta.url));
const RAW_KEY = /\b[A-Z][A-Z0-9_]*\$[A-Z0-9_$]+\b/g;

const pages = [
  { name: "chat", path: "/conversations/1", ready: '[data-testid="chat-interface"]' },
  { name: "settings-app", path: "/settings/app", ready: "main" },
  { name: "mcp", path: "/mcp", ready: "main" },
];

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: 1440, height: 900 },
});
await context.addInitScript(() => {
  const backend = {
    id: "mock",
    name: "Mock",
    host: window.location.origin,
    apiKey: "mock",
    kind: "local",
  };
  const active = JSON.stringify({ backendId: "mock", orgId: null });
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem("openhands-backends", JSON.stringify([backend]));
  localStorage.setItem("openhands-active-backend", active);
  sessionStorage.setItem("openhands-active-backend", active);
  localStorage.setItem(
    "openhands-sidebar-onboarding-checklist-dismissed",
    "true",
  );
});

const page = await context.newPage();
let failures = 0;
for (const { name, path, ready } of pages) {
  await page.goto(`${BASE}${path}`);
  await page.locator(ready).first().waitFor({ timeout: 60_000 });
  const consent = page.getByRole("button", { name: "Confirm preferences" });
  try {
    await consent.waitFor({ state: "visible", timeout: 8_000 });
    await consent.click();
  } catch {
    // The analytics dialog only appears once per browser context.
  }
  await page.waitForTimeout(1_500);
  const text = await page.evaluate(() => document.body.innerText);
  const rawKeys = [...new Set(text.match(RAW_KEY) ?? [])];
  console.log(`${name}: raw translation keys on page: ${rawKeys.length ? rawKeys.join(", ") : "none"}`);
  if (rawKeys.length) failures += 1;
  await page.screenshot({ path: join(OUT, `${PREFIX}${name}.png`) });
}
await browser.close();
process.exit(failures ? 1 : 0);
