// Smoke test for OHE-3934 against the Canvas mock dev server.
//
// Seeds a cloud backend, answers the two cloud endpoints the composer gate
// reads (`/api/v1/web-client/config` and `/api/agent-profiles`) from inside
// the page, and screenshots the home composer with the instance reporting
// Agent Profiles on and off.
//
// Usage (from the repo root, with `npm run dev:mock` running):
//   node .pr/ohe-3934/smoke.cjs http://127.0.0.1:3001 .pr/ohe-3934
const path = require("node:path");
const fs = require("node:fs");
const { chromium } = require("@playwright/test");

const [, , baseUrl = "http://localhost:3001", outDir = "."] = process.argv;
fs.mkdirSync(outDir, { recursive: true });

const PROFILES = {
  profiles: [
    { id: "p1", name: "Reviewer", agent_kind: "openhands", scope: "org" },
  ],
  active_agent_profile_id: "p1",
};

async function openHome(browser, { agentProfilesEnabled }) {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    colorScheme: "dark",
  });
  await context.addInitScript(
    ({ origin, flag, profiles }) => {
      localStorage.setItem("openhands-onboarded", "1");
      localStorage.setItem(
        "openhands-backends",
        JSON.stringify([
          {
            id: "ohe",
            name: "OHE",
            host: origin,
            apiKey: "cloud-test-api-key",
            kind: "cloud",
          },
        ]),
      );
      localStorage.setItem(
        "openhands-active-backend",
        JSON.stringify({ backendId: "ohe", orgId: "test-org" }),
      );
      localStorage.setItem("analytics-consent", "false");
      localStorage.setItem("openhands-telemetry-consent", "denied");
      localStorage.setItem("openhands-telemetry-first-use", "true");

      const json = (body) =>
        new Response(JSON.stringify(body), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      const originalFetch = window.fetch.bind(window);
      window.fetch = async (input, init) => {
        const url = typeof input === "string" ? input : input.url;
        const method = (init && init.method) || "GET";
        if (url.includes("/api/v1/web-client/config")) {
          return json({
            feature_flags: {
              deployment_mode: "self_hosted",
              enable_agent_profiles_page: flag,
              enable_org_agent_profiles: flag,
              enable_agent_context_page: flag,
            },
          });
        }
        if (method === "GET" && /\/api\/agent-profiles(\?|$)/.test(url)) {
          return json(profiles);
        }
        return originalFetch(input, init);
      };
    },
    { origin: baseUrl, flag: agentProfilesEnabled, profiles: PROFILES },
  );
  const page = await context.newPage();
  await page.goto(baseUrl + "/", { waitUntil: "domcontentloaded" });
  // Vite optimises deps on the first request and reloads once.
  await page.waitForTimeout(8000);
  await page.reload({ waitUntil: "domcontentloaded" });
  const consent = page.getByTestId("confirm-telemetry-preferences");
  if (await consent.isVisible({ timeout: 5000 }).catch(() => false)) {
    await consent.click();
  }
  await page.getByTestId("chat-plus-button").waitFor({ timeout: 30000 });
  return { context, page };
}

async function capture(browser, agentProfilesEnabled, name) {
  const { context, page } = await openHome(browser, { agentProfilesEnabled });
  // Let the profile list and the instance config settle.
  await page.waitForTimeout(3000);
  const pill = page.getByTestId("chat-input-agent-profile");
  const pillShown = await pill.isVisible().catch(() => false);
  await page.getByTestId("chat-plus-button").click();
  await page.getByTestId("tools-context-menu").waitFor({ timeout: 10000 });
  const itemShown = await page
    .getByTestId("switch-agent-profile-button")
    .isVisible()
    .catch(() => false);
  await page.screenshot({
    path: path.join(outDir, `${name}.png`),
    fullPage: false,
  });
  console.log(
    `${name}: enable_agent_profiles_page=${agentProfilesEnabled} ` +
      `pill=${pillShown} switch-agent-profile-item=${itemShown}`,
  );
  await context.close();
  return { pillShown, itemShown };
}

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const on = await capture(browser, true, "agent-profiles-on");
    const off = await capture(browser, false, "agent-profiles-off");
    const ok = on.pillShown && on.itemShown && !off.pillShown && !off.itemShown;
    console.log(ok ? "RESULT: gate behaves as expected" : "RESULT: UNEXPECTED");
    process.exitCode = ok ? 0 : 1;
  } finally {
    await browser.close();
  }
})();
