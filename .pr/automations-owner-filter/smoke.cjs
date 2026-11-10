// Smoke test for the owner filter against the Canvas mock dev server.
//
// Seeds a cloud team workspace (a member, no manage permission), answers the
// cloud endpoints the list page reads from inside the page, and serves an
// automations list that honours `run_as`: one automation that runs as the
// member, one that runs as a teammate, one that runs as a service account.
// Screenshots the owner filter with the instance reporting service accounts
// on and off, then selects Service accounts and records the list request.
//
// Usage (from the repo root, with the mock dev server running and the
// extensions manifest that declares the `service` option installed):
//   node .pr/automations-owner-filter/smoke.cjs http://localhost:3012 .pr/automations-owner-filter
const path = require("node:path");
const fs = require("node:fs");
const { chromium } = require("@playwright/test");

const [, , baseUrl = "http://localhost:3001", outDir = "."] = process.argv;
fs.mkdirSync(outDir, { recursive: true });

const AUTOMATIONS = [
  {
    id: "a-mine",
    name: "Nightly dependency audit",
    trigger: { type: "cron", schedule: "0 9 * * *" },
    enabled: true,
    prompt: "Audit the dependencies",
    user_id: "user-me",
    run_as_kind: "user",
    created_by_user_id: "user-me",
    created_at: "2026-11-01T00:00:00Z",
    updated_at: "2026-11-01T00:00:00Z",
  },
  {
    id: "a-teammate",
    name: "Weekly changelog",
    trigger: { type: "cron", schedule: "0 9 * * 1" },
    enabled: true,
    prompt: "Write the changelog",
    user_id: "user-teammate",
    run_as_kind: "user",
    created_by_user_id: "user-teammate",
    created_at: "2026-11-02T00:00:00Z",
    updated_at: "2026-11-02T00:00:00Z",
  },
  {
    id: "a-service",
    name: "CVE remediation",
    trigger: { type: "cron", schedule: "0 */6 * * *" },
    enabled: true,
    prompt: "Fix the open CVEs",
    user_id: "svc-release-bot",
    run_as_kind: "service",
    created_by_user_id: "user-teammate",
    created_at: "2026-11-03T00:00:00Z",
    updated_at: "2026-11-03T00:00:00Z",
  },
];

async function waitForServer() {
  for (let attempt = 0; attempt < 90; attempt += 1) {
    try {
      const response = await fetch(baseUrl + "/");
      if (response.ok) return;
    } catch {
      // not up yet
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error(`No dev server at ${baseUrl}`);
}

async function openList(browser, { serviceAccountsEnabled }) {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    colorScheme: "dark",
  });
  await context.addInitScript(
    ({ origin, flag, automations }) => {
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
        JSON.stringify({ backendId: "ohe", orgId: "org-team" }),
      );
      localStorage.setItem("analytics-consent", "false");
      localStorage.setItem("openhands-telemetry-consent", "denied");
      localStorage.setItem("openhands-telemetry-first-use", "true");

      const json = (body) =>
        new Response(JSON.stringify(body), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      const listRequests = JSON.parse(
        sessionStorage.getItem("list-requests") || "[]",
      );
      const originalFetch = window.fetch.bind(window);
      window.fetch = async (input, init) => {
        const url = typeof input === "string" ? input : input.url;
        const method = (init && init.method) || "GET";
        const { pathname, searchParams } = new URL(url, origin);
        if (pathname === "/api/v1/web-client/config") {
          return json({
            feature_flags: {
              deployment_mode: "self_hosted",
              enable_service_accounts: flag,
            },
          });
        }
        if (pathname === "/api/organizations") {
          return json({
            items: [
              { id: "org-team", name: "Widget team", is_personal: false },
              { id: "user-me", name: "Personal", is_personal: true },
            ],
            current_org_id: "org-team",
          });
        }
        if (pathname === "/api/keys/current") {
          return json({ org_id: "org-team" });
        }
        if (pathname === "/api/organizations/org-team/me") {
          return json({
            org_id: "org-team",
            user_id: "user-me",
            role: "member",
            permissions: ["view_automations"],
          });
        }
        if (pathname.endsWith("/profiles")) {
          return json({ profiles: [] });
        }
        if (method === "GET" && pathname === "/api/automation/v1") {
          const runAs = searchParams.get("run_as");
          listRequests.push(url);
          sessionStorage.setItem("list-requests", JSON.stringify(listRequests));
          const matches = automations.filter((automation) => {
            if (runAs === "me") {
              return (
                automation.run_as_kind !== "service" &&
                automation.user_id === "user-me"
              );
            }
            if (runAs === "others") {
              return (
                automation.run_as_kind !== "service" &&
                automation.user_id !== "user-me"
              );
            }
            if (runAs === "service") return automation.run_as_kind === "service";
            return true;
          });
          return json({ automations: matches, total: matches.length });
        }
        if (/^\/api\/automation\/v1\/[^/]+\/runs$/.test(pathname)) {
          return json({ runs: [], total: 0, status_counts: {} });
        }
        if (pathname === "/api/automation/v1/tags") {
          return json({ tags: [] });
        }
        if (pathname.startsWith("/api/automation/v1/drafts")) {
          return json({ drafts: [], total: 0 });
        }
        return originalFetch(input, init);
      };
    },
    { origin: baseUrl, flag: serviceAccountsEnabled, automations: AUTOMATIONS },
  );
  const page = await context.newPage();
  await page.goto(baseUrl + "/automations", { waitUntil: "domcontentloaded" });
  // Vite optimises deps on the first request and reloads once.
  await page.waitForTimeout(8000);
  await page.reload({ waitUntil: "domcontentloaded" });
  const consent = page.getByTestId("confirm-telemetry-preferences");
  if (await consent.isVisible({ timeout: 5000 }).catch(() => false)) {
    await consent.click();
  }
  await page.getByTestId("automation-card-a-service").waitFor({ timeout: 60000 });
  return { context, page };
}

async function openOwnerDropdown(page) {
  await page
    .getByTestId("automations-filters")
    .getByTestId("dropdown-trigger")
    .click();
  const field = page.getByTestId("automations-filter-created-by");
  await field.waitFor({ timeout: 10000 });
  await field.getByTestId("dropdown-trigger").click();
  await page.getByTestId("automations-filter-created-by-others").waitFor();
}

async function visibleCardIds(page) {
  return page.$$eval("[data-testid^='automation-card-']", (cards) =>
    cards.map((card) => card.getAttribute("data-testid")),
  );
}

async function capture(browser, serviceAccountsEnabled) {
  const name = serviceAccountsEnabled
    ? "service-accounts-on"
    : "service-accounts-off";
  const { context, page } = await openList(browser, { serviceAccountsEnabled });
  await page.waitForTimeout(2000);
  await openOwnerDropdown(page);
  const label = await page.getByText("Filter by owner").count();
  const serviceOption = await page
    .getByTestId("automations-filter-created-by-service")
    .isVisible()
    .catch(() => false);
  await page.screenshot({ path: path.join(outDir, `${name}.png`) });
  console.log(
    `${name}: enable_service_accounts=${serviceAccountsEnabled} ` +
      `label "Filter by owner"=${label > 0} service-option=${serviceOption}`,
  );

  let selected = null;
  if (serviceAccountsEnabled) {
    await page.getByTestId("automations-filter-created-by-service").click();
    await page
      .getByTestId("automation-card-a-mine")
      .waitFor({ state: "detached", timeout: 15000 });
    await page.waitForTimeout(1000);
    const cards = await visibleCardIds(page);
    const requests = await page.evaluate(() =>
      JSON.parse(sessionStorage.getItem("list-requests") || "[]"),
    );
    const last = requests.at(-1) || "";
    await page.screenshot({
      path: path.join(outDir, "service-accounts-selected.png"),
    });
    console.log(
      `service-accounts-selected: cards=${JSON.stringify(cards)} ` +
        `last list request=${last.replace(baseUrl, "")}`,
    );
    selected = { cards, last };
  }
  await context.close();
  return { label: label > 0, serviceOption, selected };
}

(async () => {
  await waitForServer();
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const on = await capture(browser, true);
    const off = await capture(browser, false);
    const ok =
      on.label &&
      off.label &&
      on.serviceOption &&
      !off.serviceOption &&
      on.selected.cards.join() === "automation-card-a-service" &&
      on.selected.last.includes("run_as=service");
    console.log(ok ? "RESULT: owner filter behaves as expected" : "RESULT: UNEXPECTED");
    process.exitCode = ok ? 0 : 1;
  } finally {
    await browser.close();
  }
})();
