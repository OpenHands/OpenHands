import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useAgentState } from "./use-agent-state";
import { useActiveConversation } from "./query/use-active-conversation";
import { useOptionalConversationId } from "./use-conversation-id";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { AgentState } from "#/types/agent-state";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

const TEST_CONVERSATION_ID = "test-conversation-id";

vi.mock("./query/use-active-conversation", () => ({
  useActiveConversation: vi.fn(),
}));

vi.mock("./use-conversation-id", () => ({
  useOptionalConversationId: vi.fn(),
}));

describe("useAgentState", () => {
  const mockUseActiveConversation = vi.mocked(useActiveConversation);
  const mockUseOptionalConversationId = vi.mocked(useOptionalConversationId);

  beforeEach(() => {
    act(() => {
      useConversationStateStore.getState().reset();
    });

    mockUseActiveConversation.mockReturnValue({
      data: null,
    } as ReturnType<typeof useActiveConversation>);

    mockUseOptionalConversationId.mockReturnValue({
      conversationId: TEST_CONVERSATION_ID,
    });
  });

  it("prefers live websocket execution status over cached conversation status", () => {
    mockUseActiveConversation.mockReturnValue({
      data: {
        execution_status: ExecutionStatus.FINISHED,
      },
    } as ReturnType<typeof useActiveConversation>);

    act(() => {
      useConversationStateStore
        .getState()
        .setExecutionStatus(TEST_CONVERSATION_ID, ExecutionStatus.RUNNING);
    });

    const { result } = renderHook(() => useAgentState());

    expect(result.current.executionStatus).toBe(ExecutionStatus.RUNNING);
    expect(result.current.curAgentState).toBe(AgentState.RUNNING);
  });

  it("falls back to cached conversation execution status when live state is empty", () => {
    mockUseActiveConversation.mockReturnValue({
      data: {
        execution_status: ExecutionStatus.WAITING_FOR_CONFIRMATION,
      },
    } as ReturnType<typeof useActiveConversation>);

    const { result } = renderHook(() => useAgentState());

    expect(result.current.executionStatus).toBe(
      ExecutionStatus.WAITING_FOR_CONFIRMATION,
    );
    expect(result.current.curAgentState).toBe(
      AgentState.AWAITING_USER_CONFIRMATION,
    );
  });
});
