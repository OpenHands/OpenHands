// End-to-end evidence through the running dev server: the msw mock catalog
// now serves the renamed id, and the real formatModelPillLabel renders the
// new label in a browser.
// Run with `npm run dev:mock -- --port 3001` already serving.
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const BASE = process.env.BASE_URL || "http://127.0.0.1:3001";
mkdirSync(".pr", { recursive: true });

const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || undefined,
});
const page = await browser.newPage({ viewport: { width: 1100, height: 620 } });

await page.goto(`${BASE}/`, { waitUntil: "domcontentloaded" });
await page.waitForTimeout(4000); // let the service worker register

// Hit the mocked endpoints from inside the page so msw intercepts them.
const api = await page.evaluate(async () => {
  const get = async (u) => {
    const r = await fetch(u);
    return { status: r.status, body: await r.json() };
  };
  return {
    verified: await get("/api/llm/models/verified"),
    models: await get("/api/llm/models"),
  };
});

const openhands = api.verified.body?.models?.openhands ?? [];
const allModels = api.models.body?.models ?? [];

// Render the real label function's output in the page, next to the raw API.
const label = await page.evaluate(async () => {
  const m = await import("/src/utils/format-model-name.ts");
  const id = "openhands/deepseek-v4.1-flash";
  return {
    plain: m.formatModelPillLabel(id),
    free: m.formatModelPillLabel(id, new Set([id])),
    table: m.FREE_OPENHANDS_MODEL_IDS,
  };
});

const report = {
  "GET /api/llm/models/verified → models.openhands": openhands,
  "GET /api/llm/models  (openhands entries)": allModels.filter((m) =>
    m.startsWith("openhands/"),
  ),
  "FREE_OPENHANDS_MODEL_IDS": label.table,
  "formatModelPillLabel(id)": label.plain,
  "formatModelPillLabel(id, freeSet)": label.free,
};
console.log(JSON.stringify(report, null, 2));

const stale = JSON.stringify(report).match(/deepseek-v4-flash/);
console.log("\nstale `deepseek-v4-flash` anywhere above:", Boolean(stale));

await page.setContent(`<body style="margin:0;background:#0d0d0d;color:#e6e6e6;
  font:13px/1.65 ui-monospace,Consolas,monospace;padding:22px">
  <div style="font-size:15px;color:#d8c26e;margin-bottom:4px">
    OpenHands #17262 — deepseek-v4.1-flash, served by the running dev server (npm run dev:mock)</div>
  <div style="color:#888;margin-bottom:14px">msw-intercepted API responses + real formatModelPillLabel output</div>
  <pre style="white-space:pre-wrap">${JSON.stringify(report, null, 2).replace(/</g, "&lt;")}</pre>
  <div style="margin-top:14px;color:#7fd47f">stale "deepseek-v4-flash" present: ${Boolean(stale)}</div>
</body>`);
await page.screenshot({ path: ".pr/02-mock-api-evidence.png", fullPage: true });

await browser.close();
