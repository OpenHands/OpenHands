// Captures the PR evidence against the mock-API frontend:
//   VITE_MOCK_API=true npx react-router dev --port 5199 --strictPort
//   node .pr/oss-14431-automation-tags/evidence.mjs [after|before]
// The bundled @openhands/extensions manifest predates the tag filter and the
// tags endpoint, so the "after" run patches its interface.json locally first
// (endpoints.tags = "/v1/tags" and a { id: "tag" } filter), as the extensions
// release will.
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(path.resolve("package.json"));
const { chromium } = require("playwright");

const BASE = "http://localhost:5199";
const OUT = path.resolve(".pr/oss-14431-automation-tags");
const mode = process.argv[2] ?? "after";
const REVIEWER_ID = "a1000000-0000-0000-0000-000000000006";

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: 1280, height: 900 },
  deviceScaleFactor: 2,
});
await context.addInitScript(() => {
  const backend = {
    id: "mock-local",
    name: "Local",
    host: window.location.origin,
    apiKey: "mock",
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
});
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

// The detail header, with the tags under it, at the top of the frame.
async function detailShot(name) {
  await page.locator("h1").first().scrollIntoViewIfNeeded();
  await page.locator("h1").first().evaluate((h1) => h1.scrollIntoView());
  await page.screenshot({
    path: `${OUT}/${name}.png`,
    clip: { x: 0, y: 0, width: 1280, height: 480 },
  });
}

await page.goto(`${BASE}/automations`);
await page.getByText("PR Review on Open").first().waitFor({ timeout: 60000 });
await dismissAnalytics();

if (mode === "before") {
  await page.screenshot({ path: `${OUT}/before-list.png` });
} else {
  await page
    .getByTestId("automations-filters")
    .getByTestId("dropdown-trigger")
    .first()
    .click();
  await page
    .getByTestId("automations-filter-tag")
    .getByTestId("dropdown-trigger")
    .click();
  await page.screenshot({ path: `${OUT}/after-tag-filter-menu.png` });
  await page.getByTestId("automations-filter-tag-software-factory").click();
  await page.keyboard.press("Escape");
  await page.mouse.click(5, 5);
  await page.getByText("Release Notes Generator").waitFor({ state: "detached" });
  await page.screenshot({ path: `${OUT}/after-list-filtered.png` });

  await page.goto(`${BASE}/automations/${REVIEWER_ID}`);
  await page.getByTestId("automation-tags").waitFor({ timeout: 30000 });
  await dismissAnalytics();
  await page.getByTestId("automation-tags-edit").click();
  await page.getByTestId("automation-tags-input").fill("Nightly Checks");
  await page.keyboard.press("Enter");
  await page.getByRole("button", { name: /code-review/ }).click();
  await detailShot("after-detail-editing");
  await page.getByTestId("automation-tags-save").click();
  await page
    .getByTestId("automation-tags")
    .getByText("nightly-checks")
    .waitFor({ timeout: 15000 });
  await detailShot("after-detail-saved");
}

await browser.close();
