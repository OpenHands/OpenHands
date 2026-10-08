import { chromium } from "playwright";
import { expect } from "@playwright/test";
import { writeFile } from "node:fs/promises";
const variant = process.argv[2] || "after";
const origin = "http://127.0.0.1:19440";
const id = "11111111111111111111111111111111";
const conversation = {
  id,
  title: "Cloud editor availability · synthetic fixture",
  created_by_user_id: null,
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  trigger: null,
  pr_number: [],
  llm_model: "synthetic/model",
  metrics: null,
  created_at: "2026-10-08T12:00:00Z",
  updated_at: "2026-10-08T12:00:00Z",
  execution_status: "idle",
  conversation_url: `${origin}/api/conversations/${id}`,
  session_api_key: null,
  sandbox_id: "synthetic-sandbox",
  sandbox_status: "RUNNING",
  workspace: { working_dir: "/workspace/project" },
  public: false,
  sub_conversation_ids: [],
};
const sandbox = {
  id: "synthetic-sandbox",
  created_by_user_id: null,
  sandbox_spec_id: "synthetic",
  status: "RUNNING",
  session_api_key: null,
  created_at: conversation.created_at,
  exposed_urls: [{ name: "APP", url: `${origin}/synthetic-app` }],
};
if (variant === "positive-control")
  sandbox.exposed_urls.push({
    name: "VSCODE",
    url: `${origin}/synthetic-editor`,
  });
const browser = await chromium.launch({
  executablePath: "/snap/bin/chromium",
  headless: true,
  args: ["--no-sandbox"],
});
const context = await browser.newContext({
  viewport: { width: 1440, height: 1000 },
  deviceScaleFactor: 1,
  locale: "en-US",
  timezoneId: "UTC",
});
const page = await context.newPage();
const requests = [],
  errors = [];
await page.clock.setFixedTime(new Date("2026-10-08T13:00:00Z"));
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") console.log(m.text());
});
await context.addInitScript(() => {
  for (const [key, value] of Object.entries({
    "analytics-consent": "false",
    "openhands-telemetry-consent": "denied",
    "openhands-telemetry-first-use": "true",
    "openhands-onboarded": "1",
    "openhands-backends": JSON.stringify([
      {
        id: "synthetic-cloud",
        name: "Cloud (synthetic fixture)",
        host: location.origin,
        apiKey: "synthetic-not-a-secret",
        kind: "cloud",
      },
    ]),
    "openhands-active-backend": JSON.stringify({
      backendId: "synthetic-cloud",
      orgId: "synthetic-org",
    }),
  }))
    localStorage.setItem(key, value);
});
await context.route("**/*", async (route) => {
  const url = new URL(route.request().url());
  if (url.origin !== origin) return route.abort();
  const p = url.pathname;
  if (!p.startsWith("/api/") && p !== "/server_info") return route.continue();
  requests.push(p + url.search);
  let body = { items: [], profiles: [], next_page_id: null };
  if (p === "/api/v1/app-conversations") body = [conversation];
  else if (p === "/api/v1/app-conversations/search")
    body = { items: [conversation], next_page_id: null };
  else if (p === "/api/v1/sandboxes") body = [sandbox];
  else if (p === "/api/organizations")
    body = {
      items: [
        { id: "synthetic-org", name: "Synthetic workspace", is_personal: true },
      ],
      current_org_id: "synthetic-org",
    };
  else if (p === "/api/keys/current") body = { org_id: "synthetic-org" };
  else if (p.endsWith("/me"))
    body = {
      id: "synthetic-user",
      role: "owner",
      email: "fixture@example.invalid",
    };
  else if (p === "/api/v1/settings")
    body = { llm_api_key_set: true, llm_model: "synthetic/model" };
  else if (p === "/api/v1/web-client/config") body = {};
  else if (p.endsWith("-schema")) body = { sections: [] };
  else if (p === "/server_info") body = { version: "1.53.0", uptime: 1 };
  else if (p === `/api/conversations/${id}`)
    body = {
      ...conversation,
      agent: { kind: "Agent", llm: { model: "synthetic/model" }, tools: [] },
    };
  else if (p.endsWith("/events/count")) body = 0;
  else if (p.endsWith("/files")) body = [];
  else if (p.includes("/git/"))
    body = { branch: null, files: [], staged: [], unstaged: [] };
  await route.fulfill({ json: body });
});
const sandboxResponse = page.waitForResponse(
  (r) =>
    new URL(r.url()).pathname === "/api/v1/sandboxes" && r.status() === 200,
);
await page.goto(`${origin}/conversations/${id}`);
await page.getByTestId("right-panel-toggle").click();
await page.getByTestId("files-tab").waitFor({ state: "visible" });
await sandboxResponse;
await page.waitForTimeout(3000);
const control = page.getByTestId("drawer-vscode-link");
if (variant === "after") await expect(control).toHaveCount(0);
else await expect(control).toBeVisible();
await expect(page.getByTestId("onboarding-modal")).toHaveCount(0);
await page.mouse.move(700, 700);
console.log(await page.locator("body").innerText());
console.log(
  "TESTIDS",
  await page
    .locator("[data-testid]")
    .evaluateAll((els) => els.map((e) => e.dataset.testid)),
);
await page.screenshot({ path: `.pr/18149/${variant}.png` });
await writeFile(
  `.pr/18149/${variant}.json`,
  JSON.stringify(
    {
      variant,
      viewport: { width: 1440, height: 1000 },
      route: page.url(),
      vscodeControlCount: await control.count(),
      sandbox,
      requests,
      errors,
    },
    null,
    2,
  ),
);
await browser.close();
