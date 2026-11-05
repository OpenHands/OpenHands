/* Screenshots of Agent Canvas's Super Admin setup guide on the canvas mock
 * server (port 3001) with a cloud backend on the same origin. The two cloud
 * calls the guide makes are answered from the page itself. */
const { chromium } = require("/Users/hieple/Documents/openhands/agent-server-gui/node_modules/@playwright/test");
const OUT = process.env.OUT || "/private/tmp/claude-501/-Users-hieple-Documents-openhands-OpenHands/42354342-2c22-438d-ac9a-3cb1f36a89f0/scratchpad";
const BASE = "http://localhost:3001";

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  await context.addInitScript(() => {
    localStorage.setItem("analytics-consent", "false");
    localStorage.setItem("openhands-telemetry-consent", "denied");
    localStorage.setItem("openhands-telemetry-first-use", "true");
    localStorage.setItem("openhands-onboarded", "1");
    localStorage.setItem(
      "openhands-backends",
      JSON.stringify([{ id: "cloud-1", name: "OHE", host: location.origin, apiKey: "", kind: "cloud" }]),
    );
    localStorage.setItem("openhands-active-backend", JSON.stringify({ backendId: "cloud-1", orgId: "2" }));
    const STATE = {
      wizard_pending: false,
      guide_org_id: "2",
      guide_dismissed: false,
      guide_steps: { org_llm: true, git_provider: false, mcp_server: false, automation: false, invite: false },
    };
    const ME = { org_id: "2", user_id: "user-1", role: "owner", permissions: ["manage_super_admins", "view_automations"] };
    const realFetch = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      const url = typeof input === "string" ? input : input.url;
      const method = ((init && init.method) || (input && input.method) || "GET").toUpperCase();
      const json = (body) => new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
      if (method === "GET" && url.includes("/api/admin/setup-state")) return json(STATE);
      if (method === "GET" && /\/api\/organizations\/[^/]+\/me(\?|$)/.test(url)) return json(ME);
      return realFetch(input, init);
    };
  });
  const page = await context.newPage();

  for (let i = 0; i < 40; i += 1) {
    try {
      await page.goto(`${BASE}/mcp`, { waitUntil: "domcontentloaded", timeout: 15000 });
      await page.waitForTimeout(4000);
      await page.goto(`${BASE}/mcp`, { waitUntil: "domcontentloaded", timeout: 15000 });
      const consent = page.getByTestId("confirm-telemetry-preferences");
      if (await consent.isVisible({ timeout: 3000 }).catch(() => false)) await consent.click();
      await page.getByTestId("super-admin-setup-guide-panel").waitFor({ timeout: 20000 });
      break;
    } catch (e) {
      if (i === 39) throw e;
      await page.waitForTimeout(3000);
    }
  }

  const panel = page.getByTestId("super-admin-setup-guide-panel");
  const stepIds = await panel.locator('[data-testid^="super-admin-setup-guide-step-"]').evaluateAll((els) =>
    els.map((el) => el.getAttribute("data-testid").replace("super-admin-setup-guide-step-", "")),
  );
  console.log("[canvas] steps:", JSON.stringify(stepIds));
  console.log("[canvas] progress:", await panel.locator("span.tabular-nums").innerText(), "| next:", await page.getByTestId("super-admin-setup-guide-next").innerText());
  console.log("[canvas] start href:", await page.getByTestId("super-admin-setup-guide-start").getAttribute("href"));
  await panel.screenshot({ path: `${OUT}/canvas-after-panel.png` });

  // Start runs the Connect Git tour on the MCP page: the GitHub card first.
  await page.getByTestId("super-admin-setup-guide-start").click();
  await page.locator(".driver-popover").waitFor({ timeout: 20000 });
  console.log("[canvas] tour stop:", (await page.locator(".driver-popover").innerText()).replace(/\s+/g, " "));
  await page.screenshot({ path: `${OUT}/canvas-after-tour-github-card.png` });
  await page.getByTestId("mcp-marketplace-card-github").click();
  await page.getByTestId("mcp-install-modal").waitFor({ timeout: 20000 });
  await page.waitForTimeout(800);
  console.log("[canvas] tour stop 2:", (await page.locator(".driver-popover").innerText()).replace(/\s+/g, " "));
  await page.screenshot({ path: `${OUT}/canvas-after-tour-token.png` });
  await browser.close();
})().catch((e) => {
  console.error("CANVAS SMOKE FAILED:", e);
  process.exit(1);
});
