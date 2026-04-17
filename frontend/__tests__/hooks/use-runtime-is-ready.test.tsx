import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { useAgentState } from "#/hooks/use-agent-state";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { AgentState } from "#/types/agent-state";
import { V1AppConversation } from "#/api/conversation-service/v1-conversation-service.types";

vi.mock("#/hooks/use-agent-state");
vi.mock("#/hooks/query/use-active-conversation");

function asMockReturnValue<T>(value: Partial<T>): T {
  return value as T;
}

function makeConversation(): V1AppConversation {
  return {
    id: "conv-123",
    title: "Test Conversation",
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    updated_at: new Date().toISOString(),
    created_at: new Date().toISOString(),
    sandbox_status: "RUNNING",
    execution_status: null,
    conversation_url: null,
    session_api_key: null,
    sub_conversation_ids: [],
    created_by_user_id: null,
    sandbox_id: "sandbox-123",
    trigger: null,
    pr_number: [],
    llm_model: "llm-model",
    metrics: null,
  };
}

describe("useRuntimeIsReady", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(useActiveConversation).mockReturnValue(
      asMockReturnValue<ReturnType<typeof useActiveConversation>>({
        data: makeConversation(),
      }),
    );
  });

  it("treats agent errors as not ready by default", () => {
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.ERROR,
    });

    const { result } = renderHook(() => useRuntimeIsReady());

    expect(result.current).toBe(false);
  });

  it("allows runtime-backed tabs to stay ready when the agent errors", () => {
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.ERROR,
    });

    const { result } = renderHook(() =>
      useRuntimeIsReady({ allowAgentError: true }),
    );

    expect(result.current).toBe(true);
  });
});
