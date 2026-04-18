import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useAgentState } from "./use-agent-state";
import { useActiveConversation } from "./query/use-active-conversation";
import { useV1ConversationStateStore } from "#/stores/v1-conversation-state-store";
import { AgentState } from "#/types/agent-state";
import { V1ExecutionStatus } from "#/types/v1/core/base/common";

vi.mock("./query/use-active-conversation", () => ({
  useActiveConversation: vi.fn(),
}));

describe("useAgentState", () => {
  const mockUseActiveConversation = vi.mocked(useActiveConversation);

  beforeEach(() => {
    act(() => {
      useV1ConversationStateStore.getState().reset();
    });

    mockUseActiveConversation.mockReturnValue({
      data: null,
    } as ReturnType<typeof useActiveConversation>);
  });

  it("prefers live websocket execution status over cached conversation status", () => {
    mockUseActiveConversation.mockReturnValue({
      data: {
        execution_status: V1ExecutionStatus.FINISHED,
      },
    } as ReturnType<typeof useActiveConversation>);

    act(() => {
      useV1ConversationStateStore
        .getState()
        .setExecutionStatus(V1ExecutionStatus.RUNNING);
    });

    const { result } = renderHook(() => useAgentState());

    expect(result.current.executionStatus).toBe(V1ExecutionStatus.RUNNING);
    expect(result.current.curAgentState).toBe(AgentState.RUNNING);
  });

  it("falls back to cached conversation execution status when live state is empty", () => {
    mockUseActiveConversation.mockReturnValue({
      data: {
        execution_status: V1ExecutionStatus.WAITING_FOR_CONFIRMATION,
      },
    } as ReturnType<typeof useActiveConversation>);

    const { result } = renderHook(() => useAgentState());

    expect(result.current.executionStatus).toBe(
      V1ExecutionStatus.WAITING_FOR_CONFIRMATION,
    );
    expect(result.current.curAgentState).toBe(
      AgentState.AWAITING_USER_CONFIRMATION,
    );
  });
});
