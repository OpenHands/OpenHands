import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { RuntimeTokenUsage } from "#/api/conversation-service/agent-server-conversation-service.types";
import EventService from "#/api/event-service/event-service.api";
import { useEventStore } from "#/stores/use-event-store";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import {
  ExecutionStatus,
  SecurityRisk,
} from "#/types/agent-server/core/base/common";
import { TokenUsageBreakdown } from "./token-usage-breakdown";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, string>) =>
      options ? `${key} ${JSON.stringify(options)}` : key,
  }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      id: "conv-1",
      conversation_url: null,
      session_api_key: null,
    },
  }),
}));

const call = (
  responseId: string,
  promptTokens: number,
  completionTokens: number,
): RuntimeTokenUsage => ({
  response_id: responseId,
  prompt_tokens: promptTokens,
  completion_tokens: completionTokens,
  cache_read_tokens: 0,
  cache_write_tokens: 0,
  context_window: 200_000,
  per_turn_token: promptTokens + completionTokens,
});

const toolCall = (
  id: string,
  toolName: string,
  responseId: string,
  timestamp: string,
): OpenHandsEvent => ({
  id,
  timestamp,
  source: "agent",
  thought: [],
  thinking_blocks: [],
  action: {
    kind: "ExecuteBashAction",
    command: "true",
    is_input: false,
    timeout: null,
    reset: false,
  },
  tool_name: toolName,
  tool_call_id: `call-${id}`,
  tool_call: {
    id: `call-${id}`,
    type: "function",
    function: { name: toolName, arguments: "{}" },
  },
  llm_response_id: responseId,
  security_risk: SecurityRisk.UNKNOWN,
});

function mockRuntimeStats(tokenUsages: RuntimeTokenUsage[]) {
  vi.spyOn(
    AgentServerConversationService,
    "getRuntimeConversation",
  ).mockResolvedValue({
    id: "conv-1",
    title: "Test",
    metrics: null,
    created_at: "2026-10-06T00:00:00Z",
    updated_at: "2026-10-06T00:00:00Z",
    status: ExecutionStatus.IDLE,
    stats: {
      usage_to_metrics: {
        default: {
          model_name: "test-model",
          accumulated_cost: 0,
          max_budget_per_task: null,
          accumulated_token_usage: null,
          costs: [],
          response_latencies: [],
          token_usages: tokenUsages,
        },
      },
    },
  });
}

function renderBreakdown() {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <TokenUsageBreakdown />
    </QueryClientProvider>,
  );
}

const renderedActivityKeys = () =>
  screen
    .getAllByTestId("token-usage-activity")
    .map((row) => row.getAttribute("data-activity-key"));

describe("TokenUsageBreakdown", () => {
  beforeEach(() => {
    useEventStore.getState().clearEventsForConversation("conv-1");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("ranks the activities of the live events by their exact token totals", async () => {
    // Arrange
    mockRuntimeStats([call("r1", 1_000, 20), call("r2", 5_000, 80)]);
    const searchSpy = vi.spyOn(EventService, "searchEvents");
    useEventStore
      .getState()
      .addEvents([
        toolCall("a1", "terminal", "r1", "2026-10-06T00:00:01Z"),
        toolCall("a2", "file_editor", "r2", "2026-10-06T00:00:02Z"),
      ]);

    // Act
    renderBreakdown();

    // Assert
    await waitFor(() =>
      expect(renderedActivityKeys()).toEqual([
        "tools:file_editor",
        "tools:terminal",
      ]),
    );
    const [fileEditorRow] = screen.getAllByTestId("token-usage-activity");
    expect(within(fileEditorRow).getByText("file_editor")).toBeInTheDocument();
    expect(
      within(fileEditorRow).getByText((5_080).toLocaleString(), {
        exact: false,
      }),
    ).toBeInTheDocument();
    // Every call matched a live event, so no history request is needed.
    expect(searchSpy).not.toHaveBeenCalled();
  });

  it("loads the persisted history when a call has no live event", async () => {
    // Arrange: the chat only paged in the newest event (r2).
    mockRuntimeStats([call("r1", 1_000, 20), call("r2", 500, 10)]);
    useEventStore
      .getState()
      .addEvents([toolCall("a2", "terminal", "r2", "2026-10-06T00:00:02Z")]);
    vi.spyOn(EventService, "getEventCount").mockResolvedValue(2);
    const searchSpy = vi.spyOn(EventService, "searchEvents").mockResolvedValue({
      items: [
        toolCall("a2", "terminal", "r2", "2026-10-06T00:00:02Z"),
        toolCall("a1", "browser_navigate", "r1", "2026-10-06T00:00:01Z"),
      ],
      next_page_id: null,
    });

    // Act
    renderBreakdown();

    // Assert: the older call is attributed instead of left unmatched.
    await waitFor(() =>
      expect(renderedActivityKeys()).toEqual([
        "tools:browser_navigate",
        "tools:terminal",
      ]),
    );
    expect(searchSpy).toHaveBeenCalledTimes(1);
  });
});
