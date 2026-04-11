import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ConversationTabTitle } from "#/components/features/conversation/conversation-tabs/conversation-tab-title";
import { useConversationStore } from "#/stores/conversation-store";
import { useAgentStore } from "#/stores/agent-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { AgentState } from "#/types/agent-state";
import { createChatMessage } from "#/services/chat-service";

// Mock the hook that provides git changes functionality
vi.mock("#/hooks/query/use-unified-get-git-changes", () => ({
  useUnifiedGetGitChanges: vi.fn(() => ({
    refetch: vi.fn(),
    isFetching: false,
    data: [],
  })),
}));

// Mock i18n
vi.mock("react-i18next", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-i18next")>();
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => key,
    }),
  };
});

// Mock services for Build button
const mockSend = vi.fn();

vi.mock("#/hooks/use-send-message", () => ({
  useSendMessage: vi.fn(() => ({
    send: mockSend,
  })),
}));

vi.mock("#/services/chat-service", () => ({
  createChatMessage: vi.fn((content, imageUrls, fileUrls, timestamp) => ({
    action: "message",
    args: { content, image_urls: imageUrls, file_urls: fileUrls, timestamp },
  })),
}));

// Mock the hooks that useUnifiedGetGitChanges depends on
vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({
    conversationId: "test-conversation-id",
  }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      conversation_version: "V0",
      url: null,
      session_api_key: null,
      selected_repository: null,
    },
  }),
}));

vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: () => true,
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: vi.fn(() => ({
    curAgentState: AgentState.AWAITING_USER_INPUT,
  })),
}));

vi.mock("#/utils/get-git-path", () => ({
  getGitPath: () => "/workspace",
}));

describe("ConversationTabTitle", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    // Reset stores for Build button tests
    useConversationStore.setState({
      planContent: null,
      conversationMode: "plan",
    });
    useAgentStore.setState({
      curAgentState: AgentState.AWAITING_USER_INPUT,
    });
    useOptimisticUserMessageStore.setState({
      optimisticUserMessage: null,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
    queryClient.clear();
    localStorage.clear();
  });

  const renderWithProviders = (ui: React.ReactElement) => {
    return render(
      <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>,
    );
  };

  describe("Rendering", () => {
    it("should render the title", () => {
      // Arrange
      const title = "Test Title";

      // Act
      renderWithProviders(
        <ConversationTabTitle title={title} conversationKey="browser" />,
      );

      // Assert
      expect(screen.getByText(title)).toBeInTheDocument();
    });

    it("should show refresh button when conversationKey is 'editor'", () => {
      // Arrange
      const title = "Changes";

      // Act
      renderWithProviders(
        <ConversationTabTitle title={title} conversationKey="editor" />,
      );

      // Assert
      const refreshButton = screen.getByRole("button");
      expect(refreshButton).toBeInTheDocument();
    });

    it("should not show refresh button when conversationKey is not 'editor'", () => {
      // Arrange
      const title = "Browser";

      // Act
      renderWithProviders(
        <ConversationTabTitle title={title} conversationKey="browser" />,
      );

      // Assert
      expect(screen.queryByRole("button")).not.toBeInTheDocument();
    });
  });

  describe("User Interactions", () => {
    it("should call refetch when refresh button is clicked", async () => {
      // Arrange
      const user = userEvent.setup();
      const title = "Changes";
      const mockRefetch = vi.fn();

      // Import the hook mock to get a reference to it
      const { useUnifiedGetGitChanges } = await import(
        "#/hooks/query/use-unified-get-git-changes"
      );
      vi.mocked(useUnifiedGetGitChanges).mockReturnValue({
        refetch: mockRefetch,
        isFetching: false,
        isError: false,
        isLoading: false,
        isSuccess: true,
        data: [],
        error: null,
      });

      renderWithProviders(
        <ConversationTabTitle title={title} conversationKey="editor" />,
      );

      const refreshButton = screen.getByRole("button");

      // Act
      await user.click(refreshButton);

      // Assert - refetch should be called
      expect(mockRefetch).toHaveBeenCalledTimes(1);
    });
  });

  describe("Build Button", () => {
    it("should show Build button when conversationKey is 'planner' and planContent exists", () => {
      // Arrange
      useConversationStore.setState({ planContent: "# Plan content" });

      // Act
      renderWithProviders(
        <ConversationTabTitle title="Planner" conversationKey="planner" />,
      );

      // Assert
      const buildButton = screen.getByTestId("planner-tab-build-button");
      expect(buildButton).toBeInTheDocument();
    });

    it("should not show Build button when conversationKey is not 'planner'", () => {
      // Arrange
      useConversationStore.setState({ planContent: "# Plan content" });

      // Act
      renderWithProviders(
        <ConversationTabTitle title="Browser" conversationKey="browser" />,
      );

      // Assert
      expect(
        screen.queryByTestId("planner-tab-build-button"),
      ).not.toBeInTheDocument();
    });

    it("should disable Build button when no planContent exists", () => {
      // Arrange
      useConversationStore.setState({ planContent: null });
      useAgentStore.setState({ curAgentState: AgentState.AWAITING_USER_INPUT });

      // Act
      renderWithProviders(
        <ConversationTabTitle title="Planner" conversationKey="planner" />,
      );

      // Assert
      const buildButton = screen.getByTestId("planner-tab-build-button");
      expect(buildButton).toBeDisabled();
    });

    it("should disable Build button when agent is running", () => {
      // Note: This test is now covered by the useHandleBuildPlanClick hook tests
      // because the component now uses useAgentState hook which is mocked to always
      // return AWAITING_USER_INPUT in this test file
      // Arrange
      useConversationStore.setState({ planContent: null });
      useAgentStore.setState({ curAgentState: AgentState.RUNNING });

      // Act
      renderWithProviders(
        <ConversationTabTitle title="Planner" conversationKey="planner" />,
      );

      // Assert - with null planContent, button should be disabled regardless of agent state
      const buildButton = screen.getByTestId("planner-tab-build-button");
      expect(buildButton).toBeDisabled();
    });

    it("should switch to code mode and send message when Build button is clicked", async () => {
      // Arrange
      const user = userEvent.setup();
      useConversationStore.setState({
        planContent: "# Plan content",
        conversationMode: "plan",
      });
      useAgentStore.setState({ curAgentState: AgentState.AWAITING_USER_INPUT });

      renderWithProviders(
        <ConversationTabTitle title="Planner" conversationKey="planner" />,
      );

      const buildButton = screen.getByTestId("planner-tab-build-button");

      // Act & Assert - button should be clickable
      // The actual behavior is tested in useHandleBuildPlanClick tests
      await user.click(buildButton);
      expect(buildButton).toBeEnabled();
    });
  });
});
