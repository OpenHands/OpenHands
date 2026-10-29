// Evidence for the Azure DevOps link fix: open a conversation whose repository
// is an Azure DevOps repo while the connection's `host` holds the organization
// (as the server stores it), and capture the git control bar with the link
// target the repository and branch buttons point to.
//
// Usage: node .pr/azure-devops-link-host/evidence.mjs <out.png>
// Needs a running Canvas stack on http://localhost:8000 (npm run dev) and
// QA_API_KEY / QA_AGENT_PROFILE_ID in the environment.
import { createRequire } from "node:module";

const require = createRequire(new URL("../../package.json", import.meta.url));
const { chromium } = require("playwright");

const ORIGIN = "http://localhost:8000";
const KEY = process.env.QA_API_KEY;
const PROFILE = process.env.QA_AGENT_PROFILE_ID;
const OUT = process.argv[2];
const HOST = "https://dev.azure.com/my-org";
const SVC = "/src/api/conversation-service/agent-server-conversation-service.api.ts";

const browser = await chromium.launch();
const ctx = await browser.newContext({
  viewport: { width: 1200, height: 760 },
  deviceScaleFactor: 2,
});
await ctx.addInitScript(
  ({ origin, key }) => {
    const backend = { id: "qa", name: "QA", host: origin, apiKey: key, kind: "local" };
    localStorage.setItem("openhands-onboarded", "1");
    localStorage.setItem("openhands-backends", JSON.stringify([backend]));
    const active = JSON.stringify({ backendId: "qa", orgId: null });
    localStorage.setItem("openhands-active-backend", active);
    sessionStorage.setItem("openhands-active-backend", active);
    localStorage.setItem("openhands-sidebar-onboarding-checklist-dismissed", "true");
  },
  { origin: ORIGIN, key: KEY },
);
const page = await ctx.newPage();
await page.goto(`${ORIGIN}/`);
await page.waitForLoadState("networkidle").catch(() => {});
const id = await page.evaluate(
  async ({ SVC, PROFILE }) => {
    const { default: S } = await import(SVC);
    const c = await S.createConversation({
      agentProfileId: PROFILE,
      metadata: {
        selected_repository: "my-org/Web Apps/epv",
        selected_branch: "main",
        git_provider: "azure_devops",
      },
    });
    return c.id;
  },
  { SVC, PROFILE },
);

await page.goto(`${ORIGIN}/conversations/${id}`);
await page.waitForSelector('[data-testid="chat-interface"]', { timeout: 60000 });
const confirm = page.getByRole("button", { name: "Confirm preferences" });
await confirm.waitFor({ state: "visible", timeout: 15000 }).catch(() => {});
if (await confirm.isVisible().catch(() => false)) await confirm.click();
// Give the cached settings the Azure DevOps connection host the way a Cloud
// backend reports it (the organization). The git control bar reads
// `settings.provider_tokens_set[provider]` from this cache.
await page.evaluate((host) => {
  const client = window.__OH_QUERY_CLIENT__;
  const settingsQueries = client
    .getQueryCache()
    .findAll()
    .filter((q) => typeof q.state.data === "object" && q.state.data !== null && "provider_tokens_set" in q.state.data);
  if (settingsQueries.length === 0) throw new Error("no cached settings query");
  for (const q of settingsQueries) {
    client.setQueryData(q.queryKey, {
      ...q.state.data,
      provider_tokens_set: { ...q.state.data.provider_tokens_set, azure_devops: host },
    });
  }
}, HOST);
await page.waitForTimeout(3000);
const repoLink = page.locator('a[href*="dev.azure.com"], a[href*="my-org"]').first();
await repoLink.waitFor({ timeout: 30000 });
await repoLink.hover();
const hrefs = await page.$$eval('a[href*="my-org"]', (as) => as.map((a) => a.getAttribute("href")));

// Show the link targets on the page, like a browser status bar would.
await page.evaluate(
  ({ host, hrefs }) => {
    const note = document.createElement("div");
    note.setAttribute("data-evidence-note", "");
    note.style.cssText =
      "position:fixed;right:16px;top:72px;z-index:99999;padding:8px 12px;border-radius:6px;" +
      "background:#111;color:#eee;font:13px/1.5 ui-monospace,monospace;border:1px solid #555;";
    note.innerHTML =
      `<div>connection host: ${host}</div>` +
      hrefs.map((h, i) => `<div>${i === 0 ? "repo" : "branch"} link: ${h}</div>`).join("");
    document.body.appendChild(note);
  },
  { host: HOST, hrefs },
);
await page.screenshot({ path: OUT });
console.log(JSON.stringify({ id, hrefs }));
await browser.close();
