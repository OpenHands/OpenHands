// Evidence capture for OHE-4205: the Private | Shared toggle of the sidebar's
// conversation list, unsharing from the Shared list, and the organization
// setting that hides the list.
//
// The toggle is offered only in a team organization on a cloud backend, and
// the Shared list needs a backend that serves
// `GET /api/shared-conversations/search`. The mock API has neither, so this
// script registers the mock server as a cloud backend and answers these
// requests the way such a deployment would:
//   - GET   /api/organizations                   the team organization, with its
//                                                "Discover shared conversations" setting
//   - GET   /api/organizations/org-1/me          a member of the team organization
//   - GET   /api/v1/app-conversations/search     the member's own conversations
//   - GET   /api/shared-conversations/search     what members shared with the org
//   - PATCH /api/v1/app-conversations/{id}       the member unsharing their own conversation
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

async function openConversations({ discovery }) {
  const page = await browser.newPage({
    viewport: { width: 1280, height: 800 },
  });
  page.setDefaultTimeout(20000);

  await page.addInitScript(
    ({ discovery }) => {
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
      const ownShared = conversation(
        "55555555-5555-4555-8555-555555555555",
        "Release checklist",
        "user-1",
        30,
      );
      const shared = [
        conversation(
          "33333333-3333-4333-8333-333333333333",
          "API design review",
          "user-2",
          75,
        ),
        ownShared,
        conversation(
          "44444444-4444-4444-8444-444444444444",
          "Q4 planning doc",
          "user-3",
          4320,
        ),
      ].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
      const unshared = new Set();

      const answers = {
        "/api/organizations": () => ({
          items: [
            {
              id: "org-1",
              name: "Mock team organization",
              is_personal: false,
              enable_shared_conversation_discovery: discovery,
            },
          ],
          current_org_id: "org-1",
        }),
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
          items: shared.filter(({ id }) => !unshared.has(id)),
          next_page_id: null,
        }),
      };

      window.answeredRequests = [];
      const mockFetch = window.fetch.bind(window);
      window.fetch = async (input, init) => {
        const request = new Request(input, init);
        const { pathname, search } = new URL(request.url);
        const update = pathname.match(
          /^\/api\/v1\/app-conversations\/([^/]+)$/,
        );
        if (request.method === "PATCH" && update) {
          const body = await request.clone().json();
          if (body.public === false) unshared.add(update[1]);
          window.answeredRequests.push(
            `PATCH ${pathname} ${JSON.stringify(body)}`,
          );
          return json({ ...ownShared, public: body.public });
        }
        const answer = answers[pathname];
        if (!answer || request.method !== "GET") return mockFetch(request);
        window.answeredRequests.push(`GET ${pathname}${search}`);
        return json(answer());
      };
    },
    { discovery },
  );

  await page.goto(`${baseUrl}/conversations`, {
    waitUntil: "domcontentloaded",
  });
  await page.getByText("Fix flaky checkout test").first().waitFor();
  // The organization list decides whether the toggle is offered.
  await page.waitForFunction(() =>
    window.answeredRequests.includes("GET /api/organizations"),
  );
  await page.waitForTimeout(1000);

  // Dismiss the analytics consent dialog when it covers the page.
  const consent = page.getByRole("button", { name: /Confirm preferences/ });
  if (await consent.isVisible().catch(() => false)) {
    await consent.click();
    await page.waitForTimeout(500);
  }
  return page;
}

try {
  const page = await openConversations({ discovery: true });
  await page.getByTestId("conversation-list-scope").first().waitFor();
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

  // Only the member's own shared conversation has a menu.
  const teammateRow = sharedList.locator("a", { hasText: "API design review" });
  const ownRow = sharedList.locator("a", { hasText: "Release checklist" });
  console.log(
    "menu on a conversation a teammate shared:",
    await teammateRow.getByTestId("ellipsis-button").count(),
  );
  await ownRow.hover();
  await ownRow.getByTestId("ellipsis-button").click();
  const shareSwitch = page.getByTestId("toggle-shared-button");
  await shareSwitch.waitFor();
  console.log(
    "menu on own shared conversation:",
    await shareSwitch.innerText(),
  );
  await page.screenshot({ path: `${output}/03-unshare-menu.png` });

  await shareSwitch.click();
  await sharedList
    .getByText("Release checklist")
    .waitFor({ state: "detached" });
  await page.screenshot({ path: `${output}/04-after-unshare.png` });
  console.log(
    "shared rows after unsharing:",
    await sharedList
      .locator("a")
      .evaluateAll((links) =>
        links.map((link) => link.querySelector("p")?.textContent),
      ),
  );
  console.log(
    "answered requests:",
    await page.evaluate(() =>
      window.answeredRequests.filter(
        (request) =>
          request.startsWith("PATCH") ||
          request.includes("/api/shared-conversations/"),
      ),
    ),
  );
  await page.close();

  // An Org Admin turned "Discover shared conversations" off for the org.
  const hidden = await openConversations({ discovery: false });
  console.log(
    "toggle offered while discovery is off:",
    await hidden.getByTestId("conversation-list-scope").count(),
  );
  await hidden.screenshot({ path: `${output}/05-discovery-off.png` });
} finally {
  await browser.close();
}
