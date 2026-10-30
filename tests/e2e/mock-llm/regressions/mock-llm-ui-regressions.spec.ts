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
import {
  seedLocalStorage,
  routeSessionApiKey,
} from "../utils/mock-llm-helpers";

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
      activated_skills: [],
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
      activated_skills: [],
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

    const probeStyles = await shell.evaluate((el) => {
      const shellBackground = getComputedStyle(el).backgroundColor;
      const probe = document.createElement("div");
      probe.className = "bg-canvas-base";
      // Keep the token available outside the shell: an undefined variable
      // would make even a leaked utility transparent and hide a scoping bug.
      probe.style.setProperty("--oh-color-base", shellBackground);
      el.appendChild(probe);
      const insideBackground = getComputedStyle(probe).backgroundColor;
      document.documentElement.appendChild(probe);
      const outsideBackground = getComputedStyle(probe).backgroundColor;
      probe.remove();
      return {
        shellBackground,
        insideBackground,
        outsideBackground,
      };
    });

    // First prove the utility exists, then prove it cannot style the host.
    expect(probeStyles.insideBackground).toBe(probeStyles.shellBackground);
    expect(probeStyles.outsideBackground).toBe("rgba(0, 0, 0, 0)");
  });

  // @spec BM-004 — Backend feedback consumes the embedding host's theme roles
  test("backend connection feedback follows host color overrides", async ({
    page,
  }) => {
    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    const shell = page.locator("[data-agent-server-ui]").first();
    await expect(shell).toBeVisible();
    await shell.evaluate((el) => {
      const scope = el as HTMLElement;
      scope.style.setProperty("--oh-feedback-error", "#8b1029");
      scope.style.setProperty("--oh-feedback-success", "#17642f");
    });

    await page.getByTestId("backend-selector").click();
    await page.getByTestId("manage-backends-menu-item").click();
    await expect(page.getByTestId("manage-backends-status-Local")).toHaveCSS(
      "color",
      "rgb(23, 100, 47)",
    );
    await page.getByTestId("manage-backends-done").click();

    await page.getByTestId("backend-selector").click();
    await page.getByTestId("add-backend-menu-item").click();
    await page.getByTestId("add-backend-option-agent-server").click();
    await page.getByTestId("add-backend-name").fill("Unavailable server");
    await page.getByTestId("add-backend-host").fill("http://127.0.0.1:9");
    await page.getByTestId("add-backend-submit").click();
    await expect(page.getByTestId("add-backend-error")).toHaveCSS(
      "color",
      "rgb(139, 16, 41)",
    );
  });

  // @spec TN-001 — Toasts follow the embedding host's contrast ink
  test("error and success toasts follow the host contrast color", async ({
    page,
  }) => {
    await routeSessionApiKey(page);
    // A host stylesheet override that survives full-page navigation.
    await page.addInitScript(() => {
      document.addEventListener("DOMContentLoaded", () => {
        const style = document.createElement("style");
        style.textContent =
          "[data-agent-server-ui] { --oh-contrast: #8b1029 !important; }";
        document.head.append(style);
      });
    });

    await page.goto("/conversations/0123456789abcdef0123456789abcdef", {
      waitUntil: "domcontentloaded",
    });
    await expect(
      page.getByRole("status").filter({ hasText: "does not exist" }),
    ).toHaveCSS("color", "rgb(139, 16, 41)");

    await page.goto("/settings/app", { waitUntil: "domcontentloaded" });
    const soundSwitch = page.getByText("Sound Notifications", { exact: true });
    const saveButton = page.getByRole("button", { name: "Save Changes" });
    await soundSwitch.click();
    await saveButton.click();
    await expect(
      page.getByRole("status").filter({ hasText: "Settings saved" }),
    ).toHaveCSS("color", "rgb(139, 16, 41)");

    // Restore the original setting for later serial scenarios.
    await soundSwitch.click();
    await saveButton.click();
    await expect(saveButton).toBeDisabled();
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

  // ── Drag-selection stays inside the message feed ─────────────────

  /**
   * Drags from a message in the feed past the top/bottom edge. Without the
   * clamp the selection escapes to the pane header (up) or the composer
   * (down); with it the focus stays inside the feed.
   */
  async function dragSelectionPastEdge(page: Page, direction: "up" | "down") {
    const scroller = await waitForScrollableConversation(page);
    // Park mid-feed, then let any auto-scroll settle before measuring so the
    // pointer coordinates still point at a message when the drag starts.
    await scroller.evaluate((el) => {
      const feed = el;
      feed.scrollTop = Math.round((feed.scrollHeight - feed.clientHeight) / 2);
    });
    await page.waitForTimeout(300);

    const start = await pickVisibleMessage(
      page,
      direction === "down" ? "first" : "last",
    );
    // Stay inside the viewport but outside the feed so the selection tries to
    // escape into the pane header (up) or the composer (down).
    const targetY = direction === "down" ? page.viewportSize()!.height - 1 : 1;

    await page.mouse.move(start.x, start.y);
    await page.mouse.down();
    await page.mouse.move(start.x, targetY, { steps: 30 });
    // Wait for the auto-scroll to run and the clamp to engage rather than a
    // fixed delay, so the assertion does not race the feed's scroll.
    await expect
      .poll(
        () =>
          scroller.evaluate((el) =>
            el.contains(window.getSelection()?.focusNode ?? null),
          ),
        { timeout: 5_000 },
      )
      .toBe(true);

    const result = await scroller.evaluate((el) => {
      const feed = el;
      const selection = window.getSelection();
      return {
        focusInFeed: feed.contains(selection?.focusNode ?? null),
        text: selection?.toString() ?? "",
      };
    });

    await page.mouse.up();
    return result;
  }

  /** A point inside the first or last fully visible message text node, with the
   *  message number that text belongs to. */
  async function pickVisibleMessage(page: Page, position: "first" | "last") {
    const scroller = await waitForScrollableConversation(page);
    return scroller.evaluate((el, pos) => {
      const feed = el;
      const bounds = feed.getBoundingClientRect();
      const walker = document.createTreeWalker(feed, NodeFilter.SHOW_TEXT);
      const candidates: { x: number; y: number; number: number }[] = [];
      let node: Node | null;
      while ((node = walker.nextNode())) {
        const match = node.textContent?.match(/Pagination message (\d+)/);
        if (!match) continue;
        const range = document.createRange();
        range.selectNodeContents(node);
        const rect = range.getBoundingClientRect();
        if (rect.top > bounds.top + 30 && rect.bottom < bounds.bottom - 30) {
          candidates.push({
            x: Math.round(rect.left + 8),
            y: Math.round(rect.top + rect.height / 2),
            number: Number(match[1]),
          });
        }
      }
      const point = pos === "first" ? candidates[0] : candidates.at(-1);
      if (!point) throw new Error(`no draggable message found (${pos})`);
      return point;
    }, position);
  }

  for (const direction of ["up", "down"] as const) {
    test(`keeps drag-selection in the feed when dragging ${direction} past the edge`, async ({
      page,
    }) => {
      test.setTimeout(60_000);
      await routeSessionApiKey(page);
      await routePaginationConversation(page);

      await page.goto(`/conversations/${PAGINATION_CONVERSATION_ID}`, {
        waitUntil: "domcontentloaded",
      });
      await expect(
        page.getByText("Pagination message 100", { exact: true }),
      ).toBeVisible({ timeout: 15_000 });

      const selection = await dragSelectionPastEdge(page, direction);

      expect(selection.focusInFeed).toBe(true);
      expect(selection.text.length).toBeGreaterThan(0);
      expect(selection.text).not.toContain("Disconnected");
    });
  }

  /**
   * Drags from a message, past the top edge, and back into the feed without
   * releasing. With the clamp rebuilding the range via `collapse()` +
   * `extend()`, the browser anchor survives, so the selection resumes from the
   * original message instead of restarting at the first feed child.
   */
  test("keeps the original anchor when an upward drag reenters the feed", async ({
    page,
  }) => {
    test.setTimeout(60_000);
    await routeSessionApiKey(page);
    await routePaginationConversation(page);
    await page.goto(`/conversations/${PAGINATION_CONVERSATION_ID}`, {
      waitUntil: "domcontentloaded",
    });
    await expect(
      page.getByText("Pagination message 100", { exact: true }),
    ).toBeVisible({ timeout: 15_000 });

    const scroller = await waitForScrollableConversation(page);
    await scroller.evaluate((el) => {
      const feed = el;
      feed.scrollTop = Math.round((feed.scrollHeight - feed.clientHeight) / 2);
    });
    await page.waitForTimeout(300);

    const start = await pickVisibleMessage(page, "last");

    // Start on the bottom-most visible message, escape past the top edge, then
    // come back down into the feed without releasing the button.
    await page.mouse.move(start.x, start.y);
    await page.mouse.down();
    await page.mouse.move(start.x, 1, { steps: 20 });
    await page.waitForTimeout(400);

    const reenter = await pickVisibleMessage(page, "last");
    await page.mouse.move(reenter.x, reenter.y, { steps: 20 });
    await page.waitForTimeout(400);

    const selection = await scroller.evaluate((el) => {
      const s = window.getSelection();
      const text = s?.toString() ?? "";
      const numbers = [...text.matchAll(/Pagination message (\d+)/g)].map((m) =>
        Number(m[1]),
      );
      // The message the browser anchor sits in. `anchorNode` is a text node, so
      // resolve the enclosing message element and read its number.
      const anchorText = s?.anchorNode?.textContent ?? "";
      const anchorNumber = Number(
        anchorText.match(/Pagination message (\d+)/)?.[1],
      );
      return {
        anchorInFeed: el.contains(s?.anchorNode ?? null),
        anchorIsFeedElement: s?.anchorNode === el,
        anchorNumber: Number.isNaN(anchorNumber) ? null : anchorNumber,
        focusInFeed: el.contains(s?.focusNode ?? null),
        minNumber: numbers.length ? Math.min(...numbers) : null,
      };
    });
    await page.mouse.up();

    expect(selection.anchorInFeed).toBe(true);
    // A regression that resets the anchor to the feed element must fail.
    expect(selection.anchorIsFeedElement).toBe(false);
    expect(selection.focusInFeed).toBe(true);
    // The browser anchor must still be the message the drag started on. With the
    // anchor reset to the feed element this is null; with it reset to the feed
    // start the anchor message would be 1.
    expect(selection.anchorNumber).toBe(start.number);
    // Re-entering near the bottom must not restart the range at message 1.
    expect(selection.minNumber).toBeGreaterThan(1);
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

  // @spec FF-001 — Validation feedback follows the embedding host's error role
  test("profile validation feedback follows the host error role", async ({
    page,
  }) => {
    await routeSessionApiKey(page);
    // Isolate the missing-model state from profiles saved by earlier scenarios.
    // Mock the underlying read contract, while the real editor and built CSS run.
    await page.route("**/api/profiles", async (route, request) => {
      if (request.method() !== "GET") return route.fallback();
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ profiles: [], active_profile: null }),
      });
    });
    await page.goto("/settings/agents", { waitUntil: "domcontentloaded" });
    await page.getByTestId("add-agent-profile").click();
    await expect(page.getByTestId("agent-profile-no-llm")).toBeVisible();
    await page.getByTestId("agent-profile-no-llm").evaluate((el) => {
      const scope = el.closest("[data-agent-server-ui]");
      if (!scope) throw new Error("Agent Server UI theme scope not found");
      (scope as HTMLElement).style.setProperty(
        "--oh-feedback-error",
        "#8b1029",
      );
    });
    await expect(page.getByTestId("agent-profile-no-llm")).toHaveCSS(
      "color",
      "rgb(139, 16, 41)",
    );
  });

  // @spec WN-001 — Home warning icon follows the embedding host warning role
  test("home warning icon follows the host warning color", async ({ page }) => {
    await routeSessionApiKey(page);
    await page.route("**/api/profiles", async (route, request) => {
      if (request.method() !== "GET") return route.fallback();
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ profiles: [], active_profile: null }),
      });
    });
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await expect(
      page.getByTestId("home-llm-not-configured-banner"),
    ).toBeVisible();
    const icon = page
      .getByTestId("home-llm-not-configured-banner")
      .locator("svg")
      .first();
    await icon.evaluate((element) => {
      const scope = element.closest("[data-agent-server-ui]");
      if (!scope) throw new Error("Agent Server UI theme scope not found");
      (scope as HTMLElement).style.setProperty(
        "--oh-feedback-warning",
        "#8b1029",
      );
    });
    await expect(icon).toHaveCSS("color", "rgb(139, 16, 41)");
  });

  // @spec WN-001 — Canvas Apps trust notice follows the host warning role
  test("Apps trust notice follows the host warning color", async ({ page }) => {
    await routeSessionApiKey(page);
    await page.goto("/apps", { waitUntil: "domcontentloaded" });
    const notice = page.getByText("This app runs trusted JavaScript", {
      exact: false,
    });
    await expect(notice).toBeVisible();
    await notice.evaluate((element) => {
      const scope = element.closest("[data-agent-server-ui]");
      if (!scope) throw new Error("Agent Server UI theme scope not found");
      (scope as HTMLElement).style.setProperty(
        "--oh-feedback-warning",
        "#8b1029",
      );
    });
    await expect(notice).toHaveCSS("color", "rgb(139, 16, 41)");
  });

  // @spec HL-001 — Home mode labels follow the embedding host's contrast ink
  test("home mode toggle labels follow the host contrast color", async ({
    page,
  }) => {
    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    const toggle = page.getByTestId("home-launcher-mode-toggle");
    const selected = toggle.getByRole("radio", { checked: true });
    const unselected = toggle.getByRole("radio", { checked: false });
    await expect(selected).toBeVisible();
    await toggle.evaluate((element) => {
      const scope = element.closest("[data-agent-server-ui]");
      if (!scope) throw new Error("Agent Server UI theme scope not found");
      (scope as HTMLElement).style.setProperty("--oh-contrast", "#8b1029");
    });

    await expect(selected).toHaveCSS("color", "rgb(139, 16, 41)");
    await unselected.hover();
    await expect(unselected).toHaveCSS("color", "rgb(139, 16, 41)");
  });

  test("backend keyboard focus follows the host focus role", async ({
    page,
  }) => {
    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    const shell = page.locator("[data-agent-server-ui]").first();
    await expect(shell).toBeVisible();
    await shell.evaluate((el) => {
      (el as HTMLElement).style.setProperty("--oh-focus", "#7314b9");
    });
    await page.getByTestId("backend-selector").click();
    await page.getByTestId("add-backend-menu-item").click();
    await page.getByTestId("add-backend-option-agent-server").click();

    for (const id of [
      "add-backend-option-agent-server",
      "add-backend-local-guidance-toggle",
    ]) {
      const control = page.getByTestId(id);
      await control.focus();
      await page.keyboard.press("Space");
      await page.keyboard.press("Tab");
      await page.keyboard.press("Shift+Tab");
      await expect(control).toBeFocused();
      await expect(control).toHaveCSS("outline-color", "rgb(115, 20, 185)");
    }
  });
});
