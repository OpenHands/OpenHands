// Browser smoke test for the provider setup notice against `npm run dev:mock`
// (port 3001). Seeds a cloud backend that points at the mock, stubs the cloud
// secrets and settings endpoints the mock does not serve, and walks the three
// scenarios: nothing set up, token secret saved, GitHub MCP server installed.
// The mock's web-client config must report `deployment_mode: "self_hosted"`
// (temporary edit in src/mocks/settings-handlers.ts) for the self-hosted path.
const { createRequire } = require("module");
const path = require("path");
// Run from the repo root: node .pr/ohe-3903-provider-setup-notice/smoke.cjs <screenshot dir>
const req = createRequire(path.resolve(__dirname, "../../package.json"));
const { chromium } = req("@playwright/test");
const BASE = "http://127.0.0.1:3001";
const SHOTS = process.argv[2];
const card = '[data-testid="recommended-automation-card-github-pr-reviewer"]';
const notice = '[data-testid="provider-setup-notice"]';

async function newPage(browser, { secrets, mcpInstalled }) {
  const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
  await context.addInitScript(({ secrets, mcpInstalled }) => {
    localStorage.setItem("analytics-consent", "false");
    localStorage.setItem("openhands-telemetry-consent", "denied");
    localStorage.setItem("openhands-telemetry-first-use", "true");
    localStorage.setItem("openhands-onboarded", "1");
    localStorage.setItem("openhands-backends", JSON.stringify([
      { id: "ohe-mock", name: "OHE", host: window.location.origin, apiKey: "cloud-test-api-key", kind: "cloud" },
    ]));
    localStorage.setItem("openhands-active-backend", JSON.stringify({ backendId: "ohe-mock", orgId: "test-org" }));
    const realFetch = window.fetch.bind(window);
    window.__stubHits = [];
    window.fetch = async (input, init) => {
      const url = typeof input === "string" ? input : input.url;
      if (url.includes("/api/v1/secrets/search")) {
        window.__stubHits.push("secrets");
        return new Response(JSON.stringify({ items: secrets, next_page_id: null }), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      if (mcpInstalled && url.includes("/api/v1/settings") && !url.includes("schema") && (!init || !init.method || init.method === "GET")) {
        const res = await realFetch(input, init);
        const body = await res.json();
        body.agent_settings = { ...(body.agent_settings || {}), mcp_config: { github: { url: "https://api.githubcopilot.com/mcp/" } } };
        window.__stubHits.push("settings+mcp");
        return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      return realFetch(input, init);
    };
  }, { secrets, mcpInstalled });
  const page = await context.newPage();
  page.on("pageerror", (e) => console.log("PAGEERROR:", e.message));
  page.on("response", (r) => { if (r.status() >= 400 && r.url().includes("/api/")) console.log("HTTP", r.status(), r.request().method(), r.url().replace(BASE, "")); });
  await page.goto(`${BASE}/automations/templates`);
  const consent = page.getByTestId("confirm-telemetry-preferences");
  if (await consent.isVisible({ timeout: 3000 }).catch(() => false)) await consent.click();
  await page.waitForSelector(card, { timeout: 30000 });
  await page.waitForTimeout(1500);
  return { context, page };
}
const state = async (page) => ({ notice: (await page.locator(notice).count()) > 0, setup: (await page.locator('[data-testid="setup-dialog"]').count()) > 0, url: page.url().replace(BASE, ""), stubs: await page.evaluate(() => window.__stubHits) });

(async () => {
  const browser = await chromium.launch({ channel: "chrome" });
  {
    const { context, page } = await newPage(browser, { secrets: [], mcpInstalled: false });
    await page.click(card);
    await page.waitForSelector(notice, { timeout: 10000 });
    console.log("A1 nothing set up ->", JSON.stringify(await state(page)));
    console.log("A1 text:", (await page.locator(notice).innerText()).replace(/\s+/g, " "));
    await page.screenshot({ path: `${SHOTS}/a1-notice.png` });
    await page.click('[data-testid="provider-setup-notice-cancel"]');
    await page.waitForSelector(notice, { state: "detached", timeout: 5000 });
    console.log("A2 cancel ->", JSON.stringify(await state(page)));
    await page.click(card);
    await page.waitForSelector(notice, { timeout: 10000 });
    await page.click('[data-testid="provider-setup-notice-continue"]');
    await page.waitForSelector('[data-testid="mcp-install-modal"]', { timeout: 10000 });
    console.log("A3 continue -> install modal, PAT field:", (await page.locator('[data-testid="mcp-install-field-api_key"]').count()) > 0, "| save-as-secret toggle present:", (await page.locator('[data-testid="mcp-install-modal"]').innerText()).toLowerCase().includes("secret"));
    await page.screenshot({ path: `${SHOTS}/a3-install-modal.png` });
    await page.fill('[data-testid="mcp-install-field-api_key"]', "github_pat_smoke_test_value");
    await page.click('[data-testid="mcp-install-submit"]');
    await page.waitForTimeout(4000);
    console.log("A4 submit ->", JSON.stringify(await state(page)), "| error:", await page.locator('[data-testid="mcp-install-modal-error"]').innerText().catch(() => "none"));
    await page.screenshot({ path: `${SHOTS}/a4-after-submit.png` });
    await context.close();
  }
  {
    const { context, page } = await newPage(browser, { secrets: [{ name: "GITHUB_PERSONAL_ACCESS_TOKEN", description: "pat" }], mcpInstalled: false });
    await page.click(card);
    await page.waitForTimeout(2000);
    console.log("B1 token saved ->", JSON.stringify(await state(page)));
    await page.screenshot({ path: `${SHOTS}/b1-token.png` });
    await context.close();
  }
  {
    const { context, page } = await newPage(browser, { secrets: [], mcpInstalled: true });
    await page.click(card);
    await page.waitForTimeout(2000);
    console.log("C1 github mcp installed ->", JSON.stringify(await state(page)));
    await page.screenshot({ path: `${SHOTS}/c1-mcp.png` });
    await context.close();
  }
  await browser.close();
})().catch((e) => { console.error("SMOKE FAILED:", e); process.exit(1); });
