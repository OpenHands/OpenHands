import { screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { createRoutesStub } from "react-router";
import { renderWithProviders } from "test-utils";
import { ActivityView } from "#/components/features/activity/activity-view";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import EventService from "#/api/event-service/event-service.api";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import { SEEDED_DEFAULT_BACKEND_ID } from "#/api/backend-registry/default-backend";
import { ExecutionStatus } from "#/types/agent-server/core";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { OpenHandsEvent } from "#/types/agent-server/core";

const conversation = (
  overrides: Partial<AppConversation> = {},
): AppConversation =>
  ({
    id: "conv-1",
    title: "Test Conversation",
    selected_repository: null,
    git_provider: null,
    selected_branch: null,
    updated_at: new Date().toISOString(),
    created_at: new Date().toISOString(),
    execution_status: ExecutionStatus.RUNNING,
    conversation_url: "http://runtime/conv-1",
    created_by_user_id: "user1",
    metrics: null,
    llm_model: null,
    trigger: null,
    pr_number: [],
    session_api_key: null,
    sandbox_id: null,
    sub_conversation_ids: [],
    ...overrides,
  }) as AppConversation;

const bashAction = (): OpenHandsEvent =>
  ({
    id: "action-1",
    timestamp: "2026-10-06T00:00:00Z",
    source: "agent",
    action: { kind: "ExecuteBashAction", command: "npm test" },
    tool_name: "execute_bash",
    tool_call_id: "call-1",
  }) as unknown as OpenHandsEvent;

const RouterStub = createRoutesStub([
  { Component: () => <ActivityView />, path: "/" },
  { Component: () => null, path: "/conversations/:conversationId" },
]);

const renderActivity = () =>
  renderWithProviders(
    <ActiveBackendProvider>
      <RouterStub />
    </ActiveBackendProvider>,
    { navigation: { currentPath: "/" } },
  );

describe("ActivityView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({ items: [], next_page_id: null });
    vi.spyOn(EventService, "searchEvents").mockResolvedValue({ items: [] });
  });

  afterEach(() => {
    __resetActiveStoreForTests();
  });

  // @spec LAV-001 — Only actively executing agents are listed
  it("lists only actively executing conversations and links into them", async () => {
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        conversation({ id: "running", title: "Running agent" }),
        conversation({
          id: "waiting",
          title: "Waiting agent",
          execution_status: ExecutionStatus.WAITING_FOR_CONFIRMATION,
        }),
        conversation({
          id: "finished",
          title: "Finished agent",
          execution_status: ExecutionStatus.FINISHED,
        }),
      ],
      next_page_id: null,
    });
    vi.spyOn(EventService, "searchEvents").mockResolvedValue({
      items: [bashAction()],
    });

    renderActivity();

    expect(await screen.findByText("Running agent")).toBeInTheDocument();
    expect(await screen.findByText("Waiting agent")).toBeInTheDocument();
    expect(screen.queryByText("Finished agent")).not.toBeInTheDocument();

    const runningRow = (await screen.findByText("Running agent")).closest("a");
    expect(runningRow).toHaveAttribute(
      "href",
      `/conversations/running?backend=${SEEDED_DEFAULT_BACKEND_ID}`,
    );
  });

  // @spec LAV-002 — A row conveys status, current step, and spend
  it("shows the current step, a subagent count, and needs-attention state", async () => {
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        conversation({
          id: "waiting",
          title: "Waiting agent",
          execution_status: ExecutionStatus.WAITING_FOR_CONFIRMATION,
          metrics: {
            accumulated_cost: 0.5,
            max_budget_per_task: null,
            accumulated_token_usage: {
              prompt_tokens: 100,
              completion_tokens: 50,
              cache_read_tokens: 0,
              cache_write_tokens: 0,
              context_window: 0,
              per_turn_token: 0,
            },
          },
        }),
      ],
      next_page_id: null,
    });
    vi.spyOn(EventService, "searchEvents").mockResolvedValue({
      items: [
        {
          id: "action-1",
          timestamp: "2026-10-06T00:00:00Z",
          source: "agent",
          action: { kind: "TaskAction", subagent_type: "explorer" },
          tool_name: "task",
          tool_call_id: "call-1",
        } as unknown as OpenHandsEvent,
      ],
    });

    renderActivity();

    const row = await screen.findByTestId("activity-row");
    // The shared action-title descriptor resolves through the i18n singleton,
    // which has no loaded resources in this environment, so the key itself is
    // rendered — proof the row used the shared "Running subagent" descriptor.
    expect(row).toHaveTextContent("ACTION_MESSAGE$TASK");
    expect(row).toHaveTextContent("ACTIVITY$NEEDS_ATTENTION");
    expect(row).toHaveTextContent("ACTIVITY$SUBAGENT_COUNT");
    expect(row).toHaveTextContent("$0.5000");
    expect(row).toHaveTextContent("ACTIVITY$TOKENS");
  });

  // @spec LAV-005 — The view is reachable and has defined states
  it("renders a defined empty state when nothing is running", async () => {
    renderActivity();

    expect(await screen.findByTestId("activity-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("activity-row")).not.toBeInTheDocument();
  });

  // @spec LAV-001 — Only actively executing agents are listed
  it("keeps Load more available when the first page has no running agents", async () => {
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        conversation({
          id: "finished",
          title: "Finished agent",
          execution_status: ExecutionStatus.FINISHED,
        }),
      ],
      next_page_id: "page-2",
    });

    renderActivity();

    // The fetched page is empty of active rows, but later pages may hold
    // running agents, so the control must remain reachable.
    expect(await screen.findByTestId("activity-empty")).toBeInTheDocument();
    expect(await screen.findByText("ACTIVITY$LOAD_MORE")).toBeInTheDocument();
  });

  // @spec LAV-005 — The view is reachable and has defined states
  it("renders an error state with a retry affordance on a failed load", async () => {
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockRejectedValue(new Error("boom"));

    renderActivity();

    const error = await screen.findByTestId("activity-error");
    expect(error).toHaveTextContent("ACTIVITY$RETRY");
  });
});
