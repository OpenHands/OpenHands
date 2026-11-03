// Capture the OpenHands provider model list from the mock-API dev server,
// as evidence that the renamed model id/label renders end-to-end.
// Run: node .pr/capture.mjs   (with `npm run dev:mock` already serving :3001)
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const BASE = process.env.BASE_URL || "http://127.0.0.1:3001";
const OUT = ".pr";
mkdirSync(OUT, { recursive: true });

// ponytail: reuse the chromium already on this box; `playwright install`
// could not fetch the exactly-matching build and a screenshot does not care.
const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || undefined,
});
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

const seen = [];
page.on("console", (m) => seen.push(`[${m.type()}] ${m.text()}`));

await page.goto(`${BASE}/settings/llm`, { waitUntil: "domcontentloaded" });
await page.waitForLoadState("networkidle").catch(() => {});
await page.waitForTimeout(3000);

// Fresh profile lands on the "Add a backend" onboarding gate.
const skip = page.getByText("Skip for now", { exact: false });
if (await skip.isVisible().catch(() => false)) {
  await skip.click();
  await page.waitForTimeout(2000);
  await page.goto(`${BASE}/settings/llm`, { waitUntil: "domcontentloaded" });
  await page.waitForLoadState("networkidle").catch(() => {});
  await page.waitForTimeout(4000);
}
await page.screenshot({ path: `${OUT}/01-llm-settings.png`, fullPage: true });

// The free-models note lists the OpenHands ids straight from the mock catalog.
const note = await page
  .getByTestId("openhands-free-models-note")
  .textContent()
  .catch(() => null);

// Any visible mention of the new or the old id, wherever it renders.
const bodyText = await page.locator("body").innerText();
const hasNew = bodyText.includes("deepseek-v4.1-flash");
const hasOld = /deepseek-v4-flash/.test(bodyText);

console.log("free-models note:", note);
console.log("renders deepseek-v4.1-flash:", hasNew);
console.log("still renders old deepseek-v4-flash:", hasOld);

for (const line of bodyText.split("\n").filter((l) => /deepseek/i.test(l)))
  console.log("  match:", line.trim());

await browser.close();
