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
import { describe, it, expect, vi, beforeEach } from "vitest";
import { WebSocketProviderWrapper } from "#/contexts/websocket-provider-wrapper";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

// ── Mocks ────────────────────────────────────────────────────────────────────

const capturedUrlPerRender: (string | null | undefined)[] = [];

vi.mock("#/contexts/conversation-websocket-context", () => ({
  ConversationWebSocketProvider: ({
    conversationUrl,
    children,
  }: {
    conversationUrl?: string | null;
    children?: React.ReactNode;
  }) => {
    capturedUrlPerRender.push(conversationUrl);
    return <>{children}</>;
  },
}));

const mockUseActiveConversation = vi.fn();
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => mockUseActiveConversation(),
}));

vi.mock("#/hooks/query/use-sub-conversations", () => ({
  useSubConversations: () => ({ data: [] }),
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
    vi.clearAllMocks();
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
        conversation_url: "https://sandbox.example.com/api/conversations/conv-1",
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
