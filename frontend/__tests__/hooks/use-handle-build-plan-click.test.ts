import { renderHook, act } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { useHandleBuildPlanClick } from "#/hooks/use-handle-build-plan-click";
import { useConversationStore } from "#/stores/conversation-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { createChatMessage } from "#/services/chat-service";

// Mock the send message hook - we'll mock the underlying WebSocket services
vi.mock("#/hooks/use-send-message", () => ({
  useSendMessage: vi.fn(),
}));

// Mock the chat service
vi.mock("#/services/chat-service", () => ({
  createChatMessage: vi.fn(),
}));

// Import mocked modules
import { useSendMessage } from "#/hooks/use-send-message";

describe("useHandleBuildPlanClick", () => {
  const mockSend = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    // Reset store states
    useConversationStore.setState({
      conversationMode: "plan",
    });
    useOptimisticUserMessageStore.setState({
      optimisticUserMessage: null,
    });

    // Setup send message hook mock
    (useSendMessage as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      send: mockSend,
    });

    // Setup chat service mock
    (createChatMessage as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      action: "message",
      args: {
        content:
          "Execute the plan based on the workspace/project/PLAN.md file.",
        image_urls: [],
        file_urls: [],
        timestamp: expect.any(String),
      },
    });
  });

  afterEach(() => {
    // Clean up store states
    useConversationStore.setState({
      conversationMode: "code",
    });
    useOptimisticUserMessageStore.setState({
      optimisticUserMessage: null,
    });
  });

  it("should switch conversation mode to code when handleBuildPlanClick is called", () => {
    // Arrange
    useConversationStore.setState({ conversationMode: "plan" });
    const { result } = renderHook(() => useHandleBuildPlanClick());

    // Act
    act(() => {
      result.current.handleBuildPlanClick();
    });

    // Assert
    expect(useConversationStore.getState().conversationMode).toBe("code");
  });

  it("should send build prompt message when handleBuildPlanClick is called", () => {
    // Arrange
    const { result } = renderHook(() => useHandleBuildPlanClick());
    const expectedPrompt =
      "Execute the plan based on the workspace/project/PLAN.md file.";

    // Act
    act(() => {
      result.current.handleBuildPlanClick();
    });

    // Assert
    expect(createChatMessage).toHaveBeenCalledTimes(1);
    expect(createChatMessage).toHaveBeenCalledWith(
      expectedPrompt,
      [],
      [],
      expect.any(String),
    );
    expect(mockSend).toHaveBeenCalledTimes(1);
    expect(mockSend).toHaveBeenCalledWith(
      expect.objectContaining({
        action: "message",
        args: expect.objectContaining({
          content: expectedPrompt,
        }),
      }),
    );
  });

  it("should set optimistic user message when handleBuildPlanClick is called", () => {
    // Arrange
    useOptimisticUserMessageStore.setState({ optimisticUserMessage: null });
    const { result } = renderHook(() => useHandleBuildPlanClick());
    const expectedPrompt =
      "Execute the plan based on the workspace/project/PLAN.md file.";

    // Act
    act(() => {
      result.current.handleBuildPlanClick();
    });

    // Assert
    expect(useOptimisticUserMessageStore.getState().optimisticUserMessage).toBe(
      expectedPrompt,
    );
  });

  it("should prevent default and stop propagation when event is provided", () => {
    // Arrange
    const { result } = renderHook(() => useHandleBuildPlanClick());
    const mockEvent = {
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
    } as unknown as React.MouseEvent<HTMLButtonElement>;

    // Act
    act(() => {
      result.current.handleBuildPlanClick(mockEvent);
    });

    // Assert
    expect(mockEvent.preventDefault).toHaveBeenCalledTimes(1);
    expect(mockEvent.stopPropagation).toHaveBeenCalledTimes(1);
  });

  it("should handle call without event parameter", () => {
    // Arrange
    useConversationStore.setState({ conversationMode: "plan" });
    useOptimisticUserMessageStore.setState({ optimisticUserMessage: null });
    const { result } = renderHook(() => useHandleBuildPlanClick());

    // Act & Assert - should not throw
    act(() => {
      result.current.handleBuildPlanClick();
    });

    // Assert all expected behaviors still occur
    expect(useConversationStore.getState().conversationMode).toBe("code");
    expect(mockSend).toHaveBeenCalledTimes(1);
    expect(useOptimisticUserMessageStore.getState().optimisticUserMessage).toBe(
      "Execute the plan based on the workspace/project/PLAN.md file.",
    );
  });

  it("should handle keyboard event", () => {
    // Arrange
    useConversationStore.setState({ conversationMode: "plan" });
    const { result } = renderHook(() => useHandleBuildPlanClick());
    const mockKeyboardEvent = {
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
    } as unknown as KeyboardEvent;

    // Act
    act(() => {
      result.current.handleBuildPlanClick(mockKeyboardEvent);
    });

    // Assert
    expect(mockKeyboardEvent.preventDefault).toHaveBeenCalledTimes(1);
    expect(mockKeyboardEvent.stopPropagation).toHaveBeenCalledTimes(1);
    expect(useConversationStore.getState().conversationMode).toBe("code");
  });
});
