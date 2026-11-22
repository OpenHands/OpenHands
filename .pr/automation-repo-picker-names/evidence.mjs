// Screenshot the automation setup page's repository picker with long,
// similar organization/project/repository names on the mock-API Canvas, and
// check that the filtered options read differently without hovering.
//
//   VITE_MOCK_API=true npx react-router dev --port 5199 --strictPort
//   node .pr/automation-repo-picker-names/evidence.mjs OUT_DIR [PORT] [WIDTH]
//
// Writes list.png, option-hover.png, selected-hover.png and result.json to
// OUT_DIR and exits 1 when two filtered options show the same text.
// The local backend lists repositories from the agent server, which the mock
// API does not serve, so the request is answered here.
import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const [outDir, port = "5199", width = "1400"] = process.argv.slice(2);
const require = createRequire(path.join(process.cwd(), "package.json"));
const { chromium } = require("playwright");
const base = `http://127.0.0.1:${port}`;
fs.mkdirSync(outDir, { recursive: true });

const NAMES = [
  "fabrikam-digital-innovation/Customer Experience Platform/booking-service",
  "fabrikam-digital-innovation/Customer Experience Platform Legacy/booking-service",
  "fabrikam-digital-innovation/Customer Experience Platform/booking-web",
  "fabrikam-digital-innovation/Customer Experience Portal Migration/booking-service",
  "fabrikam-digital-innovation/Partner Integrations/booking-service-partner-gateway",
  "fabrikam-digital-innovation/Web Apps/frontend-service",
  "fabrikam-digital-innovation/Web Apps Legacy/frontend-service",
];
const SEARCH = "booking";

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: Number(width), height: 900 },
  deviceScaleFactor: 2,
});
await context.addInitScript(() => {
  const backend = {
    id: "mock",
    name: "Mock",
    host: location.origin,
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
await context.route("**/api/git/repositories/search**", async (route) => {
  const query = (
    new URL(route.request().url()).searchParams.get("query") || ""
  ).toLowerCase();
  const items = NAMES.filter((name) => name.toLowerCase().includes(query)).map(
    (full_name, i) => ({
      id: String(i + 1),
      full_name,
      git_provider: "github",
      is_public: false,
    }),
  );
  await route.fulfill({ json: { items, next_page_id: null } });
});

const page = await context.newPage();
await page.goto(`${base}/automations/setup`);
await page.waitForSelector('[data-testid="automation-setup-repository-add"]', {
  timeout: 60000,
});
const confirm = page.getByRole("button", { name: "Confirm preferences" });
try {
  await confirm.waitFor({ state: "visible", timeout: 8000 });
  await confirm.click();
} catch {
  // The analytics dialog does not always open.
}

const menu = page.locator('[data-testid="automation-setup-repository-menu"]');
const option = (name) =>
  page.locator(`[data-testid="automation-setup-repository-option-${name}"]`);
const openFilteredMenu = async () => {
  await page.click('[data-testid="automation-setup-repository-add"]');
  await page.fill('[data-testid="automation-setup-repository-search"]', SEARCH);
  await page
    .locator('[data-testid^="automation-setup-repository-option-"]')
    .first()
    .waitFor({ timeout: 20000 });
};
const shoot = async (file, locator, extra = {}) => {
  const box = await locator.boundingBox();
  const x = Math.max(box.x - 40, 0);
  await page.screenshot({
    path: path.join(outDir, file),
    clip: {
      x,
      y: Math.max(box.y - (extra.above ?? 140), 0),
      width: Math.min(box.width + (extra.width ?? 80), Number(width) - x),
      height: box.height + (extra.below ?? 180),
    },
  });
};

await openFilteredMenu();
await page.mouse.move(0, 0);
// The text each option shows, with a truncated part ending in "…".
const shown = await menu.evaluate((root) =>
  [
    ...root.querySelectorAll(
      '[data-testid^="automation-setup-repository-option-"]',
    ),
  ].map((item) => {
    const parts = [];
    const walker = document.createTreeWalker(item, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const box = node.parentElement;
      const right = box.getBoundingClientRect().right;
      const truncated = box.scrollWidth > box.clientWidth;
      const range = document.createRange();
      let visible = "";
      for (let i = 0; i < node.length; i += 1) {
        range.setStart(node, i);
        range.setEnd(node, i + 1);
        const rect = range.getBoundingClientRect();
        if (rect.width > 0 && rect.right > right - (truncated ? 8 : 0) + 0.5)
          break;
        visible += node.data[i];
      }
      parts.push(visible.length < node.length ? `${visible}…` : visible);
    }
    return {
      name: item.dataset.testid.replace(
        "automation-setup-repository-option-",
        "",
      ),
      shown: parts.join(""),
    };
  }),
);
const menuWidth = (await menu.boundingBox()).width;
await shoot("list.png", menu);

await option(NAMES[1]).hover();
await page.waitForTimeout(1500);
const optionTitle = await option(NAMES[1])
  .locator("[title]")
  .first()
  .getAttribute("title")
  .catch(() => null);
const optionTooltip = await page
  .locator('[role="tooltip"]')
  .first()
  .textContent({ timeout: 1000 })
  .catch(() => null);
await shoot("option-hover.png", menu, { above: 160 });

// Select two names that differ only in the project.
await option(NAMES[0]).click();
await openFilteredMenu();
await option(NAMES[1]).click();
const values = page.locator(
  '[data-testid="automation-setup-repository-value"]',
);
await values.nth(1).waitFor();
await values.nth(1).hover();
await page.waitForTimeout(1500);
const valueTooltip = await page
  .locator('[role="tooltip"]')
  .first()
  .textContent({ timeout: 1000 })
  .catch(() => null);
const valueTitle = await values
  .nth(1)
  .locator("[title]")
  .first()
  .getAttribute("title")
  .catch(() => null);
await shoot(
  "selected-hover.png",
  page.locator('[data-testid="automation-setup-repository"]'),
  { above: 80, below: 120, width: 80 },
);

const texts = shown.map((entry) => entry.shown);
const duplicates = texts.filter((text, i) => texts.indexOf(text) !== i);
const result = {
  menuWidth,
  shown,
  duplicates,
  option: { title: optionTitle, tooltip: optionTooltip },
  selected: { title: valueTitle, tooltip: valueTooltip },
};
fs.writeFileSync(
  path.join(outDir, "result.json"),
  `${JSON.stringify(result, null, 2)}\n`,
);
console.log(JSON.stringify(result, null, 2));
await browser.close();
process.exit(duplicates.length > 0 ? 1 : 0);
