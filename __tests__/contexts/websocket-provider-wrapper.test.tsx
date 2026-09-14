/**
 * Tests that WebSocketProviderWrapper correctly gates the conversationUrl
 * it passes down to ConversationWebSocketProvider based on sandbox_status.
 *
 * Regression: when a cloud sandbox is PAUSED the API does NOT clear
 * conversation_url — the stale URL persists. We must suppress it until the
 * sandbox has fully resumed, otherwise the WS provider immediately tries to
 * connect to a dead host and the browser console fills with connection errors.
 */
import React from "react";
import { render } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { WebSocketProviderWrapper } from "#/contexts/websocket-provider-wrapper";
import { useConversationStore } from "#/stores/conversation-store";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { LOCAL_PLANNER_PARENT_TAG_KEY } from "#/utils/plan-file";

// ── Mocks ────────────────────────────────────────────────────────────────────

const capturedUrlPerRender: (string | null | undefined)[] = [];
const capturedSubConversationIdsPerRender: (readonly string[])[] = [];

vi.mock("#/contexts/conversation-websocket-context", () => ({
  ConversationWebSocketProvider: ({
    conversationUrl,
    subConversationIds,
    children,
  }: {
    conversationUrl?: string | null;
    subConversationIds?: readonly string[];
    children?: React.ReactNode;
  }) => {
    capturedUrlPerRender.push(conversationUrl);
    capturedSubConversationIdsPerRender.push(subConversationIds ?? []);
    return <>{children}</>;
  },
}));

const mockUseActiveConversation = vi.fn();
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => mockUseActiveConversation(),
}));

const mockUseSubConversations = vi.fn();
vi.mock("#/hooks/query/use-sub-conversations", () => ({
  useSubConversations: (...args: unknown[]) => mockUseSubConversations(...args),
}));

const mockUseActiveBackend = vi.fn();
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => mockUseActiveBackend(),
}));

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeConversation(
  overrides: Partial<AppConversation> = {},
): AppConversation {
  return {
    id: "conv-1",
    created_by_user_id: null,
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    title: "Test",
    trigger: null,
    pr_number: [],
    llm_model: null,
    metrics: null,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    execution_status: null,
    conversation_url: "https://sandbox.example.com/api/conversations/conv-1",
    session_api_key: "sess-key",
    sandbox_id: "sbx-1",
    sub_conversation_ids: [],
    ...overrides,
  };
}

function renderWrapper() {
  render(
    <WebSocketProviderWrapper conversationId="conv-1">
      <div />
    </WebSocketProviderWrapper>,
  );
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("WebSocketProviderWrapper — conversationUrl gating", () => {
  beforeEach(() => {
    capturedUrlPerRender.length = 0;
    capturedSubConversationIdsPerRender.length = 0;
    vi.clearAllMocks();
    mockUseActiveBackend.mockReturnValue({ backend: { kind: "local" } });
    mockUseSubConversations.mockReturnValue({ data: [] });
  });

  it("passes conversation_url through when sandbox_status is null (local backend)", () => {
    mockUseActiveConversation.mockReturnValue({
      data: makeConversation({ sandbox_status: null }),
    });

    renderWrapper();

    expect(capturedUrlPerRender.at(-1)).toBe(
      "https://sandbox.example.com/api/conversations/conv-1",
    );
  });

  it("passes conversation_url through when sandbox_status is RUNNING", () => {
    mockUseActiveConversation.mockReturnValue({
      data: makeConversation({ sandbox_status: "RUNNING" }),
    });

    renderWrapper();

    expect(capturedUrlPerRender.at(-1)).toBe(
      "https://sandbox.example.com/api/conversations/conv-1",
    );
  });

  it("suppresses conversation_url (returns null) when sandbox_status is PAUSED", () => {
    mockUseActiveConversation.mockReturnValue({
      data: makeConversation({
        sandbox_status: "PAUSED",
        // The API keeps the stale URL even while paused — this is the regression.
        conversation_url:
          "https://sandbox.example.com/api/conversations/conv-1",
      }),
    });

    renderWrapper();

    expect(capturedUrlPerRender.at(-1)).toBeNull();
  });

  it("passes null through when conversation data has no url (sandbox still starting)", () => {
    mockUseActiveConversation.mockReturnValue({
      data: makeConversation({ sandbox_status: null, conversation_url: null }),
    });

    renderWrapper();

    expect(capturedUrlPerRender.at(-1)).toBeNull();
  });

  it("passes undefined through when conversation data is not yet fetched", () => {
    mockUseActiveConversation.mockReturnValue({ data: undefined });

    renderWrapper();

    expect(capturedUrlPerRender.at(-1)).toBeUndefined();
  });
});

/**
 * Regression: ConversationWebSocketProvider resets its planning-history
 * tracking (and drops any pending PLAN.md update) whenever the
 * `subConversationIds` array it receives changes *by reference* — even when
 * the resolved planner id hasn't actually changed. `useSubConversations`
 * returns a brand-new array on every refetch (e.g. the default
 * refetchOnWindowFocus, or just `execution_status` ticking between polls),
 * so deriving `subConversationIds` straight from that array without an
 * intermediate primitive-keyed memo silently breaks live PLAN.md updates
 * after the first refetch — the Planner tab then only catches up on a full
 * reload. See websocket-provider-wrapper.tsx's plannerConversationId memo.
 */
describe("WebSocketProviderWrapper — subConversationIds reference stability", () => {
  beforeEach(() => {
    capturedUrlPerRender.length = 0;
    capturedSubConversationIdsPerRender.length = 0;
    vi.clearAllMocks();
    mockUseActiveBackend.mockReturnValue({ backend: { kind: "local" } });
  });

  it("keeps the same subConversationIds array reference across refetches that resolve to the same planner", () => {
    mockUseActiveConversation.mockReturnValue({
      data: makeConversation({ sub_conversation_ids: ["planner-1"] }),
    });
    // A fresh array/object literal every call — exactly what a refetch (even
    // one returning identical planner data) produces from useSubConversations.
    mockUseSubConversations.mockImplementation(() => ({
      data: [
        makeConversation({
          id: "planner-1",
          tags: { [LOCAL_PLANNER_PARENT_TAG_KEY]: "conv-1" },
        }),
      ],
    }));

    const { rerender } = render(
      <WebSocketProviderWrapper conversationId="conv-1">
        <div />
      </WebSocketProviderWrapper>,
    );
    rerender(
      <WebSocketProviderWrapper conversationId="conv-1">
        <div />
      </WebSocketProviderWrapper>,
    );

    expect(capturedSubConversationIdsPerRender.length).toBeGreaterThanOrEqual(
      2,
    );
    expect(capturedSubConversationIdsPerRender.at(-1)).toEqual(["planner-1"]);
    // The actual regression check: same reference, not just same value.
    expect(capturedSubConversationIdsPerRender.at(-1)).toBe(
      capturedSubConversationIdsPerRender.at(-2),
    );
  });
});

/**
 * Regression: `localPlanningConversationId` is a single unscoped Zustand
 * field, not keyed by conversation. Right after switching conversations,
 * this component re-renders with the new `conversationId` prop before
 * `useActiveConversation()` has refetched for it (and before the separate
 * reset effect in routes/conversation.tsx has cleared the *previous*
 * conversation's value out of the store) — trusting the stale store value
 * during that window would open the planner socket for the wrong
 * conversation's planner. See `trustedLocalPlanningConversationId` in
 * websocket-provider-wrapper.tsx.
 */
describe("WebSocketProviderWrapper — stale cross-conversation localPlanningConversationId", () => {
  beforeEach(() => {
    capturedUrlPerRender.length = 0;
    capturedSubConversationIdsPerRender.length = 0;
    vi.clearAllMocks();
    mockUseActiveBackend.mockReturnValue({ backend: { kind: "local" } });
    mockUseSubConversations.mockReturnValue({ data: [] });
  });

  afterEach(() => {
    useConversationStore.setState({ localPlanningConversationId: null });
  });

  it("does not fall back to a stale localPlanningConversationId while the active conversation query hasn't caught up to the rendered conversationId", () => {
    // The store still holds the *previous* conversation's planner id, and
    // useActiveConversation hasn't resolved data for the new conversationId
    // yet (data undefined — exactly the state on the first render after a
    // conversation switch).
    useConversationStore.setState({ localPlanningConversationId: "planner-A" });
    mockUseActiveConversation.mockReturnValue({ data: undefined });

    render(
      <WebSocketProviderWrapper conversationId="conv-B">
        <div />
      </WebSocketProviderWrapper>,
    );

    expect(capturedSubConversationIdsPerRender.at(-1)).toEqual([]);
  });

  it("trusts localPlanningConversationId once the active conversation has resolved to the conversation being rendered", () => {
    useConversationStore.setState({ localPlanningConversationId: "planner-B" });
    mockUseActiveConversation.mockReturnValue({
      data: makeConversation({ id: "conv-B", sub_conversation_ids: [] }),
    });

    render(
      <WebSocketProviderWrapper conversationId="conv-B">
        <div />
      </WebSocketProviderWrapper>,
    );

    expect(capturedSubConversationIdsPerRender.at(-1)).toEqual(["planner-B"]);
  });
});
