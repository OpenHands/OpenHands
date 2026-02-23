import { screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { QueryClient } from "@tanstack/react-query";
import { ChangeAgentButton } from "#/components/features/chat/change-agent-button";
import { renderWithProviders } from "../../../../test-utils";
import { useConversationStore } from "#/stores/conversation-store";

// Mock feature flag to enable planning agent
vi.mock("#/utils/feature-flags", () => ({
  USE_PLANNING_AGENT: () => true,
}));

// Mock WebSocket status
vi.mock("#/hooks/use-unified-websocket-status", () => ({
  useUnifiedWebSocketStatus: () => "CONNECTED",
}));

// Mock agent state
vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({ curAgentState: "IDLE" }),
}));

// Track invalidateQueries calls
const mockInvalidateQueries = vi.fn();

// Mock react-query to track invalidateQueries calls
vi.mock("@tanstack/react-query", async () => {
  const actual = await vi.importActual("@tanstack/react-query");
  return {
    ...actual,
    useQueryClient: () => ({
      invalidateQueries: mockInvalidateQueries,
    }),
  };
});

// Mock the active conversation hook
const mockConversationData = {
  conversation_id: "parent-conversation-123",
  sub_conversation_ids: [],
};

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: mockConversationData,
    isFetched: true,
    refetch: vi.fn(),
  }),
}));

// Mock the sub-conversation task polling hook to control task status
const mockTaskPollingResult = {
  task: null as any,
  taskStatus: undefined as string | undefined,
  taskDetail: null,
  taskError: null,
  isLoadingTask: false,
  subConversationId: undefined as string | undefined,
};

vi.mock("#/hooks/query/use-sub-conversation-task-polling", () => ({
  useSubConversationTaskPolling: () => mockTaskPollingResult,
}));

// Mock the handle plan click hook
vi.mock("#/hooks/use-handle-plan-click", () => ({
  useHandlePlanClick: () => ({
    handlePlanClick: vi.fn(),
    isCreatingConversation: false,
  }),
}));

describe("ChangeAgentButton - Cache Invalidation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset store state
    useConversationStore.setState({
      conversationMode: "code",
      subConversationTaskId: null,
    });
    // Reset mock task polling result
    mockTaskPollingResult.taskStatus = undefined;
    mockTaskPollingResult.subConversationId = undefined;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should invalidate parent conversation cache exactly once when task becomes READY", async () => {
    // Arrange - Set up a task ID in the store
    useConversationStore.setState({
      subConversationTaskId: "task-456",
    });

    // Simulate task becoming READY
    mockTaskPollingResult.taskStatus = "READY";
    mockTaskPollingResult.subConversationId = "sub-conversation-789";

    // Act - Render the component
    renderWithProviders(<ChangeAgentButton />);

    // Assert - Cache should be invalidated exactly once
    await waitFor(() => {
      expect(mockInvalidateQueries).toHaveBeenCalledTimes(1);
    });

    expect(mockInvalidateQueries).toHaveBeenCalledWith({
      queryKey: ["user", "conversation", "parent-conversation-123"],
    });
  });

  it("should not invalidate cache when task status is not READY", async () => {
    // Arrange - Set up a task ID with WORKING status
    useConversationStore.setState({
      subConversationTaskId: "task-456",
    });

    mockTaskPollingResult.taskStatus = "WORKING";
    mockTaskPollingResult.subConversationId = undefined;

    // Act
    renderWithProviders(<ChangeAgentButton />);

    // Assert - Wait a bit then verify no invalidation occurred
    await new Promise((resolve) => {
      setTimeout(resolve, 100);
    });
    expect(mockInvalidateQueries).not.toHaveBeenCalled();
  });

  it("should not invalidate cache when there is no subConversationTaskId", async () => {
    // Arrange - No task ID set
    useConversationStore.setState({
      subConversationTaskId: null,
    });

    mockTaskPollingResult.taskStatus = "READY";
    mockTaskPollingResult.subConversationId = "sub-conversation-789";

    // Act
    renderWithProviders(<ChangeAgentButton />);

    // Assert
    await new Promise((resolve) => {
      setTimeout(resolve, 100);
    });
    expect(mockInvalidateQueries).not.toHaveBeenCalled();
  });

  it("should render the button when planning agent feature is enabled", () => {
    // Arrange & Act
    renderWithProviders(<ChangeAgentButton />);

    // Assert
    const button = screen.getByRole("button");
    expect(button).toBeInTheDocument();
  });
});
