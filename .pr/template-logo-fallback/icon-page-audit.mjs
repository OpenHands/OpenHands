// For a Canvas origin: screenshot the templates page and report, for every
// template card badge (templates page) and rail badge (automations page),
// whether it draws a logo image or a glyph, or nothing.
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(
  "/Users/VascoSchiavo/git/OpenHands/.claude/worktrees/template-logo-fallback/package.json",
);
const { chromium } = require("playwright");
const [origin, outPrefix] = process.argv.slice(2);
const apiKey = readFileSync("/Users/VascoSchiavo/.openhands/agent-canvas/api-key.txt", "utf8").trim();

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1280, height: 1000 } });
await context.addInitScript(({ apiKey, origin }) => {
  const active = JSON.stringify({ backendId: "local", orgId: null });
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem("openhands-backends", JSON.stringify([{ id: "local", name: "Local", host: origin, apiKey, kind: "local" }]));
  localStorage.setItem("openhands-active-backend", active);
  sessionStorage.setItem("openhands-active-backend", active);
  localStorage.setItem("openhands-sidebar-onboarding-checklist-dismissed", "true");
}, { apiKey, origin });
const page = await context.newPage();

async function audit(path, prefix) {
  await page.goto(`${origin}${path}`);
  await page.getByRole("button", { name: "Confirm preferences" }).click({ timeout: 5000 }).catch(() => {});
  await page.locator(`[data-testid^="${prefix}"]`).first().waitFor({ timeout: 60000 });
  await page.waitForTimeout(1500); // let logo images load
  return page.$$eval(`[data-testid^="${prefix}"]`, (badges, prefix) =>
    badges.map((b) => {
      const imgs = [...b.querySelectorAll("img")].map((i) => i.alt.replace(/ logo$/, ""));
      const glyph = b.querySelector("svg") ? "glyph" : "";
      return { id: b.dataset.testid.slice(prefix.length), shows: imgs.join("+") || glyph || "NOTHING" };
    }), prefix);
}

for (const [path, prefix] of [
  ["/automations/templates", "recommended-automation-icon-"],
  ["/automations", "recommended-automation-rail-icon-"],
]) {
  const rows = await audit(path, prefix).catch((e) => [{ id: `(${path}: ${String(e).slice(0, 80)})`, shows: "" }]);
  console.log(`== ${path}`);
  for (const r of rows) console.log(" ", r.id.padEnd(46), r.shows);
  if (path === "/automations/templates") {
    await page.screenshot({ path: `${outPrefix}-templates.png` });
    const card = page.getByTestId("recommended-automation-icon-github-pr-reviewer").locator("xpath=ancestor::button[1]");
    await card.screenshot({ path: `${outPrefix}-card.png` }).catch(() => {});
  }
}
await browser.close();
