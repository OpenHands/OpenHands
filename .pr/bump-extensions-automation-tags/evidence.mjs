// Captures this PR's evidence from the real local stack (`npm run dev`: automation
// service from PyPI, @openhands/extensions from node_modules), on one scratch
// state dir that both runs share:
//   before: main's pins (automation 1.25.1, extensions 0.35.0); software-factory
//           creates two stations and news-digest is added, none tagged
//   after:  this branch (automation 1.26.0 migrates the same DB, extensions
//           0.36.0); `factory.py tag` and `create` go through the ingress
//   node .pr/bump-extensions-automation-tags/evidence.mjs [before|after]
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(path.resolve("package.json"));
const { chromium } = require("playwright");

const BASE = "http://localhost:8000";
const OUT = path.resolve(".pr/bump-extensions-automation-tags");
const mode = process.argv[2] ?? "after";
const API_KEY = process.env.LOCAL_BACKEND_API_KEY ?? "evidence-session-key";

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: 1280, height: 720 },
  deviceScaleFactor: 2,
});
await context.addInitScript((apiKey) => {
  const backend = {
    id: "evidence-local",
    name: "Local",
    host: window.location.origin,
    apiKey,
    kind: "local",
  };
  const selection = JSON.stringify({ backendId: backend.id, orgId: null });
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem("openhands-backends", JSON.stringify([backend]));
  localStorage.setItem("openhands-active-backend", selection);
  sessionStorage.setItem("openhands-active-backend", selection);
  // List rows show tags beside the name; grid cards fold them behind +N.
  localStorage.setItem("openhands-automations-view", "list");
  localStorage.setItem(
    "openhands-sidebar-onboarding-checklist-dismissed",
    "true",
  );
}, API_KEY);
const page = await context.newPage();

async function dismissAnalytics() {
  const confirm = page.getByRole("button", { name: /confirm preferences/i });
  try {
    await confirm.waitFor({ state: "visible", timeout: 8000 });
    await confirm.click();
    await confirm.waitFor({ state: "detached", timeout: 8000 });
  } catch {
    // The dialog did not open in this session.
  }
}

await page.goto(`${BASE}/automations`);
await page.getByText("News digest").first().waitFor({ timeout: 120000 });
await dismissAnalytics();

// Open the Filters menu: on main it holds no tag filter.
await page
  .getByTestId("automations-filters")
  .getByTestId("dropdown-trigger")
  .first()
  .click();

if (mode === "before") {
  await page.screenshot({ path: `${OUT}/before-list.png` });
} else {
  await page
    .getByTestId("automations-filter-tag")
    .getByTestId("dropdown-trigger")
    .click();
  await page.getByTestId("automations-filter-tag-software-factory").click();
  // The Filters menu stays open, as in the before shot; the list behind it
  // drops the news digest, which carries no tag.
  await page.getByText("News digest").waitFor({ state: "detached" });
  await page.screenshot({ path: `${OUT}/after-list-filtered.png` });

  // The tags editor's buttons, labelled with BUTTON$SAVE now.
  await page.keyboard.press("Escape");
  await page.mouse.click(5, 5);
  await page.getByText("AGENTS.md Maintainer").first().click();
  await page.getByTestId("automation-tags").waitFor({ timeout: 30000 });
  await page.getByTestId("automation-tags-edit").click();
  await page.getByTestId("automation-tags-save").waitFor();
  await page.screenshot({
    path: `${OUT}/after-detail-editing.png`,
    clip: { x: 0, y: 0, width: 1280, height: 420 },
  });
}

await browser.close();
