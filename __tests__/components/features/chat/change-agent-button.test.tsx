import type { ReactNode } from "react";
import { screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { QueryClient } from "@tanstack/react-query";
import { ChangeAgentButton } from "#/components/features/chat/change-agent-button";
import { renderWithProviders } from "../../../../test-utils";
import { useConversationStore } from "#/stores/conversation-store";

// Stub StyledTooltip so its content renders eagerly. HeroUI's Tooltip (the
// real engine) only mounts content on real-DOM hover, which jsdom doesn't
// fire reliably, so we surface the content as a marker element instead.
vi.mock("#/components/shared/buttons/styled-tooltip", () => ({
  StyledTooltip: ({
    content,
    children,
  }: {
    content: ReactNode;
    children: ReactNode;
  }) => (
    <>
      {children}
      <span data-testid="styled-tooltip-content">{content}</span>
    </>
  ),
}));

// Mock WebSocket status. Controllable so a test can simulate a ready
// connection and isolate the home-page disable from the no-connection one.
const wsState = vi.hoisted(() => ({ status: "CONNECTED" }));
vi.mock("#/hooks/use-unified-websocket-status", () => ({
  useUnifiedWebSocketStatus: () => wsState.status,
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
  id: "parent-conversation-123",
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
    wsState.status = "CONNECTED";
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

describe("ChangeAgentButton - Home page", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useConversationStore.setState({
      conversationMode: "code",
      subConversationTaskId: null,
    });
    mockTaskPollingResult.taskStatus = undefined;
    mockTaskPollingResult.subConversationId = undefined;
    // A ready websocket isolates the home-page disable: the only reason the
    // button is disabled here is the absence of a started conversation.
    wsState.status = "OPEN";
  });

  it("disables the button and explains why via a tooltip on the home page", () => {
    // Arrange & Act - render with no active conversation (the home page)
    renderWithProviders(<ChangeAgentButton />, {
      navigation: { conversationId: null },
    });

    // Assert - button stays disabled and the explanatory tooltip is wired up
    expect(screen.getByRole("button")).toBeDisabled();
    expect(screen.getByTestId("styled-tooltip-content")).toHaveTextContent(
      "CHANGE_AGENT$SWITCH_AFTER_CONVERSATION",
    );
  });

  it("drops the explanatory tooltip once a conversation has begun", () => {
    // Arrange & Act - render inside an active conversation
    renderWithProviders(<ChangeAgentButton />, {
      navigation: { conversationId: "conversation-1" },
    });

    // Assert - no home-page tooltip wrapper is rendered
    expect(
      screen.queryByTestId("styled-tooltip-content"),
    ).not.toBeInTheDocument();
  });
});
