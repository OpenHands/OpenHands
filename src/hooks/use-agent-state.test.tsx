import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useAgentState } from "./use-agent-state";
import { useActiveConversation } from "./query/use-active-conversation";
import { useOptionalConversationId } from "./use-conversation-id";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { useEventStore } from "#/stores/use-event-store";
import { AgentState } from "#/types/agent-state";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import type { OHEvent } from "#/stores/use-event-store";

const TEST_CONVERSATION_ID = "test-conversation-id";

const makeStateUpdate = (value: ExecutionStatus): OHEvent =>
  ({
    id: `state-update-${value}`,
    timestamp: "2026-10-13T00:00:00Z",
    source: "environment",
    kind: "ConversationStateUpdateEvent",
    key: "execution_status",
    value,
  }) as unknown as OHEvent;

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
      useEventStore.setState({
        events: [],
        eventIds: new Set(),
        uiEvents: [],
        loadedConversationId: null,
      });
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

  it("reads a preloaded running status from the event store before any live update", () => {
    // Simulate the first-prompt path: the REST history preload seeded the
    // event store with the conversation's early `execution_status: running`
    // update, but the WebSocket (which resumes past that tail) has not yet
    // delivered a state update, and the cached conversation still says idle.
    act(() => {
      useEventStore.setState({
        events: [makeStateUpdate(ExecutionStatus.RUNNING)],
        eventIds: new Set(["state-update-running"]),
        uiEvents: [],
        loadedConversationId: TEST_CONVERSATION_ID,
      });
    });
    mockUseActiveConversation.mockReturnValue({
      data: {
        execution_status: ExecutionStatus.IDLE,
      },
    } as ReturnType<typeof useActiveConversation>);

    const { result } = renderHook(() => useAgentState());

    expect(result.current.executionStatus).toBe(ExecutionStatus.RUNNING);
    expect(result.current.curAgentState).toBe(AgentState.RUNNING);
  });

  it("ignores preloaded events when they belong to another conversation", () => {
    act(() => {
      useEventStore.setState({
        events: [makeStateUpdate(ExecutionStatus.RUNNING)],
        eventIds: new Set(["state-update-running"]),
        uiEvents: [],
        loadedConversationId: "some-other-conversation",
      });
    });
    mockUseActiveConversation.mockReturnValue({
      data: {
        execution_status: ExecutionStatus.IDLE,
      },
    } as ReturnType<typeof useActiveConversation>);

    const { result } = renderHook(() => useAgentState());

    expect(result.current.executionStatus).toBe(ExecutionStatus.IDLE);
    expect(result.current.curAgentState).toBe(AgentState.AWAITING_USER_INPUT);
  });
});
