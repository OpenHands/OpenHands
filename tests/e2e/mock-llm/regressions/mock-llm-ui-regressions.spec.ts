/**
 * Mock-LLM E2E: UI regression tests.
 *
 * Ported from the former `tests/e2e/regressions/` directory so they run
 * against the real agent-server stack (via `bin/agent-canvas.mjs`) instead
 * of the MSW-only dev-mock server that was never wired into CI.
 *
 * Tests use `page.route()` where specific API responses need to be
 * controlled, following the same pattern as
 * `mock-llm-onboarding-regressions.spec.ts`.
 */

import test, { expect, type Page, type Request } from "@playwright/test";
import type { ActionEvent, MessageEvent } from "#/types/agent-server/core";
import { SecurityRisk } from "#/types/agent-server/core";
import type { FinishAction } from "#/types/agent-server/core/base/action";
import type { CriticResult } from "#/types/agent-server/core/base/critic";
import { seedLocalStorage, routeSessionApiKey } from "../utils/mock-llm-helpers";

test.describe.configure({ mode: "serial" });

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

// ─── pagination helpers ──────────────────────────────────────────────

const PAGINATION_CONVERSATION_ID = "pagination-e2e";
const CRITIC_CONVERSATION_ID = "critic-rendering-e2e";
const PAGE_SIZE = 50;
const PAGINATION_EVENT_COUNT = 100;
const PAGINATION_BASE_TIME = Date.UTC(2026, 4, 13, 0, 0, 0);

function timestampForEvent(index: number): string {
  return new Date(PAGINATION_BASE_TIME + index * 60_000).toISOString();
}

interface PaginationEvent {
  id: string;
  timestamp: string;
  source: string;
  kind: string;
  llm_message: {
    role: string;
    content: Array<{ type: string; text: string }>;
  };
}

type CriticEvent = MessageEvent | ActionEvent<FinishAction>;

function createPaginationEvent(index: number, prefix: string): PaginationEvent {
  return {
    id: `${prefix.toLowerCase().replaceAll(" ", "-")}-${index}`,
    timestamp: timestampForEvent(index),
    source: "agent",
    kind: "MessageEvent",
    llm_message: {
      role: "assistant",
      content: [{ type: "text", text: `${prefix} ${index}` }],
    },
  };
}

function createAllPaginationEvents(prefix: string): PaginationEvent[] {
  return Array.from({ length: PAGINATION_EVENT_COUNT }, (_, i) =>
    createPaginationEvent(i + 1, prefix),
  );
}

function searchPaginationEvents(
  events: PaginationEvent[],
  searchParams: URLSearchParams,
) {
  const limit = Number(searchParams.get("limit") ?? "100");
  const timestampLt = searchParams.get("timestamp__lt");
  const sortOrder = searchParams.get("sort_order");
  const filtered = timestampLt
    ? events.filter((e) => e.timestamp < timestampLt)
    : events;
  const sorted = [...filtered].sort((a, b) =>
    sortOrder === "TIMESTAMP_DESC"
      ? b.timestamp.localeCompare(a.timestamp)
      : a.timestamp.localeCompare(b.timestamp),
  );
  return {
    items: sorted.slice(0, limit),
    next_page_id: sorted.length > limit ? "next-page" : null,
  };
}

/** Build the mock conversation with the same fields the real agent-server
 *  returns so `requireDirectConversationInfo` can parse it and the
 *  conversation route doesn't redirect to `/conversations`. */
function buildMockConversation() {
  return {
    id: PAGINATION_CONVERSATION_ID,
    conversation_id: PAGINATION_CONVERSATION_ID,
    status: "STOPPED",
    execution_status: "stopped",
    created_at: timestampForEvent(1),
    updated_at: timestampForEvent(PAGINATION_EVENT_COUNT),
    title: "Pagination test",
  };
}

function buildCriticConversation() {
  return {
    id: CRITIC_CONVERSATION_ID,
    conversation_id: CRITIC_CONVERSATION_ID,
    status: "STOPPED",
    execution_status: "stopped",
    created_at: timestampForEvent(1),
    updated_at: timestampForEvent(4),
    title: "Critic rendering test",
  };
}

function buildCriticResult(score: number, eventId: string): CriticResult {
  return {
    score,
    message: null,
    metadata: {
      categorized_features: {
        agent_behavioral_issues: [
          {
            name: "incomplete_changes",
            display_name: "Incomplete Changes",
            probability: 0.74,
          },
        ],
        infrastructure_issues: [
          {
            name: "test_environment",
            display_name: "Test Environment",
            probability: 0.52,
          },
        ],
        user_followup_patterns: [
          {
            name: "asks_for_tests",
            display_name: "Asks for tests",
            probability: 0.33,
          },
        ],
      },
      event_ids: [eventId],
    },
  };
}

function createCriticEvents(): CriticEvent[] {
  return [
    {
      id: "critic-user-message",
      timestamp: timestampForEvent(1),
      source: "user",
      llm_message: {
        role: "user",
        content: [{ type: "text", text: "Please do the task." }],
      },
      activated_microagents: [],
      extended_content: [],
    },
    {
      id: "critic-agent-message",
      timestamp: timestampForEvent(2),
      source: "agent",
      llm_message: {
        role: "assistant",
        content: [
          {
            type: "text",
            text: "The requested change has been implemented.",
          },
        ],
      },
      activated_microagents: [],
      extended_content: [],
      critic_result: buildCriticResult(0.82, "critic-agent-message"),
    },
    {
      id: "critic-finish-action",
      timestamp: timestampForEvent(3),
      source: "agent",
      thought: [],
      thinking_blocks: [],
      action: {
        kind: "FinishAction",
        message: "Finished with critic evaluation.",
      },
      tool_name: "finish",
      tool_call_id: "call_critic_finish",
      tool_call: {
        id: "call_critic_finish",
        type: "function",
        function: {
          name: "finish",
          arguments: JSON.stringify({
            message: "Finished with critic evaluation.",
          }),
        },
      },
      llm_response_id: "response_critic_finish",
      security_risk: SecurityRisk.UNKNOWN,
      critic_result: buildCriticResult(0.64, "critic-finish-action"),
    },
  ];
}

function searchCriticEvents(
  events: CriticEvent[],
  searchParams: URLSearchParams,
) {
  const limit = Number(searchParams.get("limit") ?? "100");
  const sortOrder = searchParams.get("sort_order");
  const sorted = [...events].sort((a, b) =>
    sortOrder === "TIMESTAMP_DESC"
      ? b.timestamp.localeCompare(a.timestamp)
      : a.timestamp.localeCompare(b.timestamp),
  );
  return {
    items: sorted.slice(0, limit),
    next_page_id: null,
  };
}

/** Intercept conversation lookup + event search for pagination tests. */
async function routePaginationConversation(page: Page) {
  const allEvents = createAllPaginationEvents("Pagination message");

  // The app fetches conversations via the batch endpoint:
  //   GET /api/conversations?ids=<id>
  // Use a regex to match the query-param form of the URL. The glob `?`
  // character is a single-char wildcard in Playwright, so a regex is
  // more reliable for matching literal query strings.
  await page.route(/\/api\/conversations\?/, async (route, req) => {
    if (req.method() !== "GET") {
      await route.fallback();
      return;
    }
    const url = new URL(req.url());
    // Axios serializes array params as `ids[]=a&ids[]=b` (bracket notation).
    const ids = [
      ...url.searchParams.getAll("ids"),
      ...url.searchParams.getAll("ids[]"),
    ];
    if (ids.includes(PAGINATION_CONVERSATION_ID)) {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([buildMockConversation()]),
      });
    } else {
      await route.fallback();
    }
  });

  // Stub the event search endpoint with the synthetic paginated events.
  // Older-events requests (those with timestamp__lt) are delayed slightly so
  // the loading indicator has time to render before the response arrives.
  // Without this, React can batch the isLoading true→false transition into a
  // single commit and the DOM element never materialises — making the
  // "loading-older-events" assertion flaky.
  await page.route(
    `**/api/conversations/${PAGINATION_CONVERSATION_ID}/events/search**`,
    async (route, req) => {
      if (req.method() !== "GET") {
        await route.fallback();
        return;
      }
      const url = new URL(req.url());
      const isOlderPage = url.searchParams.has("timestamp__lt");
      const result = searchPaginationEvents(allEvents, url.searchParams);
      if (isOlderPage) {
        await new Promise((r) => setTimeout(r, 300));
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(result),
      });
    },
  );
}

/** Intercept conversation lookup + event search for critic rendering tests. */
async function routeCriticConversation(page: Page) {
  const criticEvents = createCriticEvents();

  await page.route(/\/api\/conversations\?/, async (route, req) => {
    if (req.method() !== "GET") {
      await route.fallback();
      return;
    }
    const url = new URL(req.url());
    const ids = [
      ...url.searchParams.getAll("ids"),
      ...url.searchParams.getAll("ids[]"),
    ];
    if (ids.includes(CRITIC_CONVERSATION_ID)) {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([buildCriticConversation()]),
      });
    } else {
      await route.fallback();
    }
  });

  await page.route(
    `**/api/conversations/${CRITIC_CONVERSATION_ID}/events/search**`,
    async (route, req) => {
      if (req.method() !== "GET") {
        await route.fallback();
        return;
      }
      const url = new URL(req.url());
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(
          searchCriticEvents(criticEvents, url.searchParams),
        ),
      });
    },
  );
}

async function getChatScroller(page: Page) {
  const chatInterface = page.getByTestId("chat-interface");
  await expect(chatInterface).toBeVisible({ timeout: 15_000 });
  const scroller = chatInterface.locator(".custom-scrollbar-always").first();
  await expect(scroller).toBeVisible();
  return scroller;
}

async function waitForScrollableConversation(page: Page) {
  const scroller = await getChatScroller(page);
  await expect
    .poll(() => scroller.evaluate((el) => el.scrollHeight > el.clientHeight), {
      timeout: 15_000,
    })
    .toBe(true);
  return scroller;
}

async function triggerOlderEventLoad(page: Page) {
  const scroller = await waitForScrollableConversation(page);
  await scroller.evaluate((el) => {
    el.scrollTop = 0;
    el.dispatchEvent(new Event("scroll", { bubbles: true }));
  });
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test.describe("UI regressions", () => {
  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  // ── CSS isolation ────────────────────────────────────────────────

  test("scopes standalone styles to the agent-server-ui shell", async ({
    page,
  }) => {
    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });

    const shell = page.locator("[data-agent-server-ui]").first();
    await expect(shell).toBeVisible({ timeout: 15_000 });
    const layout = page.getByTestId("root-layout");
    await expect(layout).toBeVisible();

    // The scoped background comes from the built stylesheet. In this SPA
    // build (`ssr: false`) `root-layout` can be visible — it only needs
    // geometry — a frame before the global CSS <link> lands in the CSSOM,
    // so a single read can catch the unstyled first paint as a transient
    // "rgba(0, 0, 0, 0)". The window is widest under the resource-starved
    // Docker E2E runner, where this test flaked. Poll until the scoped
    // styles resolve before asserting.
    await expect
      .poll(
        () => shell.evaluate((el) => getComputedStyle(el).backgroundColor),
        { timeout: 15_000 },
      )
      .not.toBe("rgba(0, 0, 0, 0)");

    const insideBackground = await shell.evaluate(
      (el) => getComputedStyle(el).backgroundColor,
    );

    const outsideStyles = await page.evaluate(() => {
      const probe = document.createElement("div");
      probe.className = "bg-base text-content-2";
      probe.textContent = "host";
      document.documentElement.appendChild(probe);
      const styles = getComputedStyle(probe);
      return {
        backgroundColor: styles.backgroundColor,
        color: styles.color,
      };
    });

    expect(insideBackground).not.toBe("rgba(0, 0, 0, 0)");
    expect(outsideStyles.backgroundColor).not.toBe(insideBackground);
  });

  // ── critic result rendering ──────────────────────────────────────

  test("renders critic results on agent messages and finish actions", async ({
    page,
  }) => {
    await routeSessionApiKey(page);
    await routeCriticConversation(page);

    await page.goto(`/conversations/${CRITIC_CONVERSATION_ID}`, {
      waitUntil: "domcontentloaded",
    });

    const criticLabels = page.getByText("Critic: agent success likelihood");
    await expect(criticLabels).toHaveCount(2, { timeout: 15_000 });
    await expect(page.locator('[aria-label="Score: 82.0%"]')).toBeVisible();
    await expect(page.locator('[aria-label="Score: 64.0%"]')).toBeVisible();

    await page.getByLabel("Expand details").first().click();
    await expect(page.getByText("Potential Issues:")).toBeVisible();
    await expect(page.getByText("Incomplete Changes")).toBeVisible();
    await expect(page.getByText("Infrastructure:")).toBeVisible();
    await expect(page.getByText("Test Environment")).toBeVisible();
    await expect(page.getByText("Likely Follow-up:")).toBeVisible();
    await expect(page.getByText("Asks for tests")).toBeVisible();
  });

  // ── event pagination on scroll-up ────────────────────────────────

  test("loads older events when scrolling up", async ({ page }) => {
    test.setTimeout(60_000);
    await routeSessionApiKey(page);
    await routePaginationConversation(page);

    const initialRequestPromise = page.waitForRequest((req: Request) => {
      if (req.method() !== "GET") return false;
      if (
        !req
          .url()
          .includes(
            `/api/conversations/${PAGINATION_CONVERSATION_ID}/events/search`,
          )
      )
        return false;
      return !new URL(req.url()).searchParams.has("timestamp__lt");
    });

    await page.goto(`/conversations/${PAGINATION_CONVERSATION_ID}`, {
      waitUntil: "domcontentloaded",
    });

    // Verify initial request sends correct params.
    const initialRequest = await initialRequestPromise;
    const initialUrl = new URL(initialRequest.url());
    expect(initialUrl.searchParams.get("limit")).toBe(String(PAGE_SIZE));
    expect(initialUrl.searchParams.get("sort_order")).toBe("TIMESTAMP_DESC");

    // Most recent message should be visible; older page should not.
    await expect(
      page.getByText("Pagination message 100", { exact: true }),
    ).toBeVisible({ timeout: 15_000 });
    await expect(
      page.getByText("Pagination message 50", { exact: true }),
    ).toHaveCount(0);

    // Scroll to top to trigger older-event loading.
    const olderRequestPromise = page.waitForRequest((req: Request) => {
      if (req.method() !== "GET") return false;
      if (
        !req
          .url()
          .includes(
            `/api/conversations/${PAGINATION_CONVERSATION_ID}/events/search`,
          )
      )
        return false;
      return new URL(req.url()).searchParams.has("timestamp__lt");
    });

    await triggerOlderEventLoad(page);

    await expect(page.getByTestId("loading-older-events")).toContainText(
      "Fetching older messages",
    );

    const olderRequest = await olderRequestPromise;
    const olderUrl = new URL(olderRequest.url());
    expect(olderUrl.searchParams.get("limit")).toBe(String(PAGE_SIZE));
    expect(olderUrl.searchParams.get("sort_order")).toBe("TIMESTAMP_DESC");
    expect(olderUrl.searchParams.get("timestamp__lt")).toBe(
      timestampForEvent(51),
    );

    await expect(
      page.getByText("Pagination message 50", { exact: true }),
    ).toBeAttached({ timeout: 15_000 });
    await expect(page.getByTestId("loading-older-events")).toHaveCount(0);
  });

  // ── #1076: workspace selection persistence ───────────────────────

  test("selected workspace persists after navigating away and returning", async ({
    page,
  }) => {
    test.setTimeout(60_000);
    await routeSessionApiKey(page);

    // Stub the workspace listing endpoint (GET /api/workspaces) to return
    // deterministic entries in the server-persisted shape:
    //   { workspaces: [...], workspaceParents: [...] }
    const workspacesMock = {
      workspaces: [
        {
          id: "/workspace/project/demo-app",
          name: "demo-app",
          path: "/workspace/project/demo-app",
        },
        {
          id: "/workspace/project/sample-tools",
          name: "sample-tools",
          path: "/workspace/project/sample-tools",
        },
        {
          id: "/workspace/project/notes-service",
          name: "notes-service",
          path: "/workspace/project/notes-service",
        },
      ],
      workspaceParents: [],
    };
    await page.route("**/api/workspaces**", async (route, req) => {
      if (req.method() !== "GET") {
        await route.fallback();
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(workspacesMock),
      });
    });

    await page.goto("/conversations", { waitUntil: "domcontentloaded" });

    // Open workspace dialog and pick a workspace
    const openButton = page.getByTestId("open-workspace-button");
    await expect(openButton).toBeEnabled({ timeout: 15_000 });
    await openButton.click();
    await expect(page.getByTestId("open-workspace-dialog-body")).toBeVisible();

    const dropdown = page.getByTestId("workspace-dropdown");
    await expect(dropdown).toBeEnabled({ timeout: 15_000 });
    await dropdown.click();

    const menu = page.getByTestId("workspace-dropdown-menu");
    await expect(menu).toBeVisible();
    await menu.getByText("demo-app", { exact: true }).click();

    await expect(dropdown).toHaveValue("demo-app");

    // Close the dialog
    await page.getByTestId("close-open-workspace-dialog").click();
    await expect(
      page.getByTestId("open-workspace-dialog-body"),
    ).not.toBeVisible();

    // Navigate away and back
    await page.goto("/settings", { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("home-screen")).not.toBeVisible({
      timeout: 5_000,
    });

    await page.goto("/conversations", { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("home-screen")).toBeVisible({
      timeout: 10_000,
    });

    // Reopen workspace dialog — selection should be restored.
    const reopenButton = page.getByTestId("open-workspace-button");
    await expect(reopenButton).toBeEnabled({ timeout: 15_000 });
    await reopenButton.click();
    await expect(page.getByTestId("open-workspace-dialog-body")).toBeVisible();

    const restored = page.getByTestId("workspace-dropdown");
    await expect(restored).toBeEnabled({ timeout: 15_000 });
    await expect(restored).toHaveValue("demo-app");
  });

  test("cleared sessionStorage yields empty workspace selection", async ({
    page,
  }) => {
    test.setTimeout(60_000);
    await routeSessionApiKey(page);

    await page.route("**/api/workspaces**", async (route, req) => {
      if (req.method() !== "GET") {
        await route.fallback();
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          workspaces: [
            {
              id: "/workspace/project/demo-app",
              name: "demo-app",
              path: "/workspace/project/demo-app",
            },
            {
              id: "/workspace/project/sample-tools",
              name: "sample-tools",
              path: "/workspace/project/sample-tools",
            },
          ],
          workspaceParents: [],
        }),
      });
    });

    // Explicitly clear persisted workspace path
    await page.addInitScript(() => {
      window.sessionStorage.removeItem("oh:home-selected-workspace-path");
    });

    await page.goto("/conversations", { waitUntil: "domcontentloaded" });

    const openButton = page.getByTestId("open-workspace-button");
    await expect(openButton).toBeEnabled({ timeout: 15_000 });
    await openButton.click();
    await expect(page.getByTestId("open-workspace-dialog-body")).toBeVisible();

    const dropdown = page.getByTestId("workspace-dropdown");
    await expect(dropdown).toBeEnabled({ timeout: 15_000 });
    await expect(dropdown).toHaveValue("");
  });
});
