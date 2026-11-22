// Evidence capture for OHE-4205: the Private | Shared toggle of the sidebar's
// conversation list.
//
// The toggle is offered only in a team organization on a cloud backend, and
// the Shared list needs a backend that serves
// `GET /api/shared-conversations/search`. The mock API has neither, so this
// script registers the mock server as a cloud backend and answers three
// requests the way such a deployment would:
//   - GET /api/organizations/org-1/me            a member of a team organization
//   - GET /api/v1/app-conversations/search       the member's own conversations
//   - GET /api/shared-conversations/search       what teammates shared with the org
// Everything else is the unmodified mock API.
//
// Requires the mock dev server:
//   VITE_DO_NOT_TRACK=1 VITE_MOCK_API=true npx react-router dev --port 3198
// Usage: node .pr/shared-conversations/capture.mjs [baseUrl] [outputDir]
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const baseUrl = process.argv[2] ?? "http://127.0.0.1:3198";
const output = process.argv[3] ?? ".pr/shared-conversations";
await mkdir(output, { recursive: true });

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
page.setDefaultTimeout(20000);

await page.addInitScript(() => {
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem(
    "openhands-backends",
    JSON.stringify([
      {
        id: "mock-team",
        name: "Mock team organization",
        host: location.origin,
        apiKey: "mock-api-key",
        kind: "cloud",
      },
    ]),
  );
  localStorage.setItem(
    "openhands-active-backend",
    JSON.stringify({ backendId: "mock-team", orgId: "org-1" }),
  );
  localStorage.setItem("analytics-consent", "false");
  localStorage.setItem("openhands-telemetry-consent", "denied");
  localStorage.setItem("openhands-telemetry-first-use", "true");

  const json = (body) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  const minutesAgo = (minutes) =>
    new Date(Date.now() - minutes * 60_000).toISOString();
  const conversation = (id, title, createdBy, updatedMinutesAgo) => ({
    id,
    title,
    created_by_user_id: createdBy,
    sandbox_id: `sandbox-${id}`,
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    pr_number: [],
    llm_model: null,
    metrics: null,
    parent_conversation_id: null,
    sub_conversation_ids: [],
    created_at: minutesAgo(updatedMinutesAgo),
    updated_at: minutesAgo(updatedMinutesAgo),
  });
  const answers = {
    "/api/organizations/org-1/me": () => ({
      org_id: "org-1",
      user_id: "user-1",
      role: "member",
      permissions: ["view_automations"],
    }),
    "/api/v1/app-conversations/search": () => ({
      items: [
        {
          ...conversation(
            "11111111-1111-4111-8111-111111111111",
            "Fix flaky checkout test",
            "user-1",
            12,
          ),
          execution_status: "finished",
          sandbox_status: "PAUSED",
          trigger: "gui",
        },
        {
          ...conversation(
            "22222222-2222-4222-8222-222222222222",
            "Draft the release notes",
            "user-1",
            190,
          ),
          execution_status: "finished",
          sandbox_status: "PAUSED",
          trigger: "gui",
        },
      ],
      next_page_id: null,
    }),
    "/api/shared-conversations/search": () => ({
      items: [
        conversation(
          "33333333-3333-4333-8333-333333333333",
          "API design review",
          "user-2",
          75,
        ),
        conversation(
          "44444444-4444-4444-8444-444444444444",
          "Q4 planning doc",
          "user-3",
          4320,
        ),
      ],
      next_page_id: null,
    }),
  };

  window.answeredRequests = [];
  const mockFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const request = new Request(input, init);
    const { pathname, search } = new URL(request.url);
    const answer = answers[pathname];
    if (!answer) return mockFetch(request);
    window.answeredRequests.push(`${request.method} ${pathname}${search}`);
    return json(answer());
  };
});

try {
  await page.goto(`${baseUrl}/conversations`, {
    waitUntil: "domcontentloaded",
  });
  await page.getByTestId("conversation-list-scope").first().waitFor();
  await page.getByText("Fix flaky checkout test").first().waitFor();

  // Dismiss the analytics consent dialog when it covers the page.
  const consent = page.getByRole("button", { name: /Confirm preferences/ });
  if (await consent.isVisible().catch(() => false)) {
    await consent.click();
    await page.waitForTimeout(500);
  }

  await page.screenshot({ path: `${output}/01-private-list.png` });

  await page
    .getByTestId("conversation-list-scope-option-shared")
    .first()
    .click();
  const sharedList = page.getByTestId("shared-conversation-list").first();
  await sharedList.getByText("API design review").waitFor();
  await page.screenshot({ path: `${output}/02-shared-list.png` });

  const rows = await sharedList
    .locator("a")
    .evaluateAll((links) =>
      links.map(
        (link) =>
          `${link.querySelector("p")?.textContent ?? link.textContent} -> ${link.getAttribute("href")}`,
      ),
    );
  console.log("shared rows:", rows);
  console.log(
    "own conversation visible while Shared is selected:",
    await page.getByText("Fix flaky checkout test").first().isVisible(),
  );
  console.log(
    "layouts menu visible while Shared is selected:",
    await page.getByTestId("conversation-layouts-toggle").first().isVisible(),
  );
  console.log(
    "answered requests:",
    await page.evaluate(() => window.answeredRequests),
  );
} finally {
  await browser.close();
}
