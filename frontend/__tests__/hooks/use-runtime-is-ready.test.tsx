import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Conversation } from "#/api/open-hands.types";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { useAgentState } from "#/hooks/use-agent-state";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { AgentState } from "#/types/agent-state";

vi.mock("#/hooks/use-agent-state");
vi.mock("#/hooks/query/use-active-conversation");

function asMockReturnValue<T>(value: Partial<T>): T {
  return value as T;
}

function makeConversation(): Conversation {
  return {
    conversation_id: "conv-123",
    title: "Test Conversation",
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    last_updated_at: new Date().toISOString(),
    created_at: new Date().toISOString(),
    status: "RUNNING",
    runtime_status: null,
    url: null,
    session_api_key: null,
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
