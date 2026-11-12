/*
 * Drives the two repository launch paths of a running Canvas (mock API mode)
 * against a stubbed cloud backend that has a Bitbucket Data Center repository
 * whose default branch is `master` and that has no `main` branch.
 *
 *   npm run make-i18n
 *   VITE_MOCK_API=true npx react-router dev --port 3021 --strictPort
 *   BASE=http://127.0.0.1:3021 OUT=after node .pr/ohe-3919/default-branch-smoke.cjs
 *
 * It seeds a cloud backend at the dev server's own origin, answers the git
 * endpoints the mock API does not serve, records the body of the conversation
 * start request, and writes `<OUT>-home-dialog.png`.
 *
 * DETAILS=404 makes the repository details endpoint answer 404, like a backend
 * that does not have it yet.
 */
const path = require("node:path");
const { chromium } = require("@playwright/test");

const BASE = process.env.BASE ?? "http://127.0.0.1:3021";
const OUT = process.env.OUT ?? "after";
const DETAILS = process.env.DETAILS ?? "200";

const REPOSITORY = "BANKOPS/management";

function stubCloudBackend({ detailsStatus }) {
  const origin = window.location.origin;
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem(
    "openhands-backends",
    JSON.stringify([
      {
        id: "ohe",
        name: "OpenHands Enterprise",
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

  const repositories = [
    {
      id: "42",
      full_name: "BANKOPS/management",
      git_provider: "bitbucket_data_center",
      is_public: false,
      owner_type: "organization",
      main_branch: null,
    },
    {
      id: "43",
      full_name: "BANKOPS/payments-api",
      git_provider: "bitbucket_data_center",
      is_public: false,
      owner_type: "organization",
      main_branch: null,
    },
  ];
  const branches = ["develop", "master", "release/2.4"].map((name, index) => ({
    name,
    commit_sha: `c0ffee${index}`,
    protected: false,
  }));

  const json = (body, status = 200) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  const record = (key, value) => {
    const seen = JSON.parse(sessionStorage.getItem(key) ?? "[]");
    seen.push(value);
    sessionStorage.setItem(key, JSON.stringify(seen));
  };

  const realFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const request = input instanceof Request ? input : null;
    const url = new URL(request ? request.url : String(input), origin);
    const method = (init?.method ?? request?.method ?? "GET").toUpperCase();
    const { pathname, searchParams } = url;

    if (pathname === "/api/v1/settings" && method === "GET") {
      const response = await realFetch(input, init);
      const settings = await response.json();
      return json({
        ...settings,
        provider_tokens_set: {
          bitbucket_data_center: "bitbucket.example.com",
        },
      });
    }
    if (pathname === "/api/v1/git/installations/search") {
      return json({ items: ["BANKOPS"], next_page_id: null });
    }
    if (pathname === "/api/v1/git/repositories/search") {
      return json({ items: repositories, next_page_id: null });
    }
    if (pathname === "/api/v1/git/repositories/details") {
      record("smoke-details-requests", searchParams.get("repository"));
      if (detailsStatus !== 200) {
        return json({ detail: "Not Found" }, detailsStatus);
      }
      const repository = repositories.find(
        (repo) => repo.full_name === searchParams.get("repository"),
      );
      return json({ ...repository, main_branch: "master" });
    }
    if (pathname === "/api/v1/git/branches/search") {
      const query = (searchParams.get("query") ?? "").toLowerCase();
      return json({
        items: branches.filter((branch) => branch.name.includes(query)),
        next_page_id: null,
      });
    }
    if (pathname === "/api/v1/app-conversations" && method === "POST") {
      const body = JSON.parse(
        init?.body ?? (request ? await request.clone().text() : "{}"),
      );
      record("smoke-start-requests", {
        selected_repository: body.selected_repository,
        selected_branch: body.selected_branch,
        git_provider: body.git_provider,
      });
      return json({
        id: "00000000000000000000000000000001",
        created_by_user_id: "user-1",
        status: "WORKING",
        detail: null,
        app_conversation_id: null,
        agent_server_url: null,
        request: body,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });
    }
    return realFetch(input, init);
  };
}

const readRecorded = (page, key) =>
  page.evaluate((k) => JSON.parse(sessionStorage.getItem(k) ?? "[]"), key);

async function openHome(page) {
  await page.goto(`${BASE}/`);
  // Vite optimizes dependencies on the first request and reloads the page.
  await page.waitForTimeout(8000);
  await page.goto(`${BASE}/`);
  const confirmTelemetry = page.getByTestId("confirm-telemetry-preferences");
  if (await confirmTelemetry.isVisible().catch(() => false)) {
    await confirmTelemetry.click();
  }
}

(async () => {
  const browser = await chromium.launch({ channel: "chrome" });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
  });
  await context.addInitScript(stubCloudBackend, {
    detailsStatus: Number(DETAILS),
  });
  const page = await context.newPage();

  // 1. Home "Open repository" dialog: pick the repository, read the branch.
  await openHome(page);
  await page.getByTestId("open-repository-button").click();
  const dialog = page.getByTestId("open-repository-dialog-body");
  await dialog.getByTestId("git-repo-dropdown").click();
  await page.getByText(REPOSITORY, { exact: true }).first().click();
  const branchInput = dialog.getByTestId("git-branch-dropdown-input");
  await page.waitForTimeout(2500);
  console.log(
    `home dialog: branch field after picking ${REPOSITORY} = "${await branchInput.inputValue()}"`,
  );
  await page.screenshot({
    path: path.join(__dirname, `${OUT}-home-dialog.png`),
  });

  // 2. Sidebar "new conversation" menu: click the repository to launch it.
  await page.evaluate(() => sessionStorage.clear());
  await openHome(page);
  await page.getByTestId("conversation-panel-new-thread-picker").click();
  await page
    .getByTestId("conversation-panel-new-thread-popover")
    .locator(`[data-repo-name="${REPOSITORY}"]`)
    .click();
  // The launch first waits for profile lookups the mock API answers slowly.
  await page
    .waitForFunction(() => sessionStorage.getItem("smoke-start-requests"), {
      timeout: 60000,
    })
    .catch(() => undefined);
  const started = await readRecorded(page, "smoke-start-requests");
  const details = await readRecorded(page, "smoke-details-requests");
  console.log(`sidebar launch: details lookups = ${JSON.stringify(details)}`);
  console.log(
    `sidebar launch: POST /api/v1/app-conversations = ${JSON.stringify(started[0] ?? null)}`,
  );

  await browser.close();
})();
