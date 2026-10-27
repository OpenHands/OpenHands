// Screenshot the automation setup page's repository picker with long
// organization/project/repository names, on the mock-API Canvas.
//
//   VITE_MOCK_API=true npx react-router dev --port 5199 --strictPort
//   node .pr/repo-picker-one-line/evidence.mjs .pr/repo-picker-one-line/after.png
//
// The local backend lists repositories from the agent server, which the mock
// API does not serve, so the request is answered here.
import { createRequire } from "node:module";
import path from "node:path";

const [outFile, port = "5199"] = process.argv.slice(2);
const require = createRequire(path.join(process.cwd(), "package.json"));
const { chromium } = require("playwright");
const base = `http://127.0.0.1:${port}`;

const NAMES = [
  "fabrikam-engineering/Intellectual Property Management/ip-management-notifications-service",
  "fabrikam-engineering/Customer Portal/customer-portal-frontend",
  "fabrikam-engineering/Legal Applications/legal-docs-generator",
  "fabrikam-engineering/Web Applications/fabrikam-web",
  "fabrikam-engineering/Web Applications/fabrikam-api",
  "fabrikam-engineering/Web Applications/fabrikam-notifications-worker",
];

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: 1400, height: 900 },
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

await page.click('[data-testid="automation-setup-repository-add"]');
const menu = page.locator('[data-testid="automation-setup-repository-menu"]');
await page.waitForSelector(
  '[data-testid^="automation-setup-repository-option-"]',
  { timeout: 20000 },
);
const box = await menu.boundingBox();
await page.screenshot({
  path: outFile,
  clip: {
    x: box.x - 40,
    y: box.y - 140,
    width: box.width + 80,
    height: box.height + 180,
  },
});
await browser.close();
