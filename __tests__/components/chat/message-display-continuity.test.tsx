import { describe, expect, it, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { QueryClient } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router";
import { render } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { useParamsMock, createUserMessageEvent } from "test-utils";
import { ChatInterface } from "#/components/features/chat/chat-interface";
import {
  useConversationId,
  useOptionalConversationId,
} from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import { useConfig } from "#/hooks/query/use-config";
import { useUnifiedUploadFiles } from "#/hooks/mutation/use-unified-upload-files";
import { useEventStore } from "#/stores/use-event-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";

// Module-level mocks
vi.mock("#/hooks/query/use-config");
vi.mock("#/hooks/mutation/use-unified-upload-files");
vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: vi.fn(),
  useOptionalConversationId: vi.fn(),
}));
vi.mock("#/hooks/query/use-active-conversation");
vi.mock("#/contexts/conversation-websocket-context");

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => ({
    providers: [],
  }),
}));

vi.mock("#/hooks/use-conversation-name-context-menu", () => ({
  useConversationNameContextMenu: () => ({
    isOpen: false,
    contextMenuRef: { current: null },
    handleContextMenu: vi.fn(),
    handleClose: vi.fn(),
    handleRename: vi.fn(),
    handleDelete: vi.fn(),
  }),
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: vi.fn(() => ({
    curAgentState: AgentState.AWAITING_USER_INPUT,
  })),
}));

vi.mock("#/components/features/chat/btw-messages", () => ({
  BtwMessages: () => <div data-testid="btw-messages" />,
}));

vi.mock("#/components/features/chat/chat-suggestions", () => ({
  ChatSuggestions: () => <div data-testid="chat-suggestions" />,
}));

vi.mock("#/components/features/chat/interactive-chat-box", () => ({
  InteractiveChatBox: () => <div data-testid="interactive-chat-box" />,
}));

vi.mock("#/components/shared/buttons/scroll-to-bottom-button", () => ({
  ScrollToBottomButton: () => <button type="button">Scroll to bottom</button>,
}));

vi.mock("#/components/conversation-events/chat", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/components/conversation-events/chat")
  >()),
  Messages: () => <div data-testid="conversation-messages" />,
}));

// Helper to render with QueryClient and route params
const renderWithQueryClient = (
  ui: React.ReactElement,
  queryClient: QueryClient,
  route = "/test-conversation-id",
) =>
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[route]}>
        <Routes>
          <Route path="/:conversationId" element={ui} />
          <Route path="/" element={ui} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );

describe("ChatInterface – message display continuity (spec 3.1)", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    useParamsMock.mockReturnValue({ conversationId: "test-conversation-id" });
    vi.mocked(useConversationId).mockReturnValue({
      conversationId: "test-conversation-id",
    });
    vi.mocked(useOptionalConversationId).mockReturnValue({
      conversationId: "test-conversation-id",
    });

    useOptimisticUserMessageStore.setState({ pendingMessages: [] });

    (useConfig as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { app_mode: "local" },
    });
    (
      useUnifiedUploadFiles as unknown as ReturnType<typeof vi.fn>
    ).mockReturnValue({
      mutateAsync: vi
        .fn()
        .mockResolvedValue({ skipped_files: [], uploaded_files: [] }),
      isLoading: false,
    });

    // Default: no active conversation
    vi.mocked(useActiveConversation).mockReturnValue({
      data: undefined,
    } as ReturnType<typeof useActiveConversation>);

    // Default: no websocket context
    vi.mocked(useConversationWebSocket).mockReturnValue(null);
  });

  describe("conversations", () => {
    beforeEach(() => {
      // Set up conversation
      vi.mocked(useActiveConversation).mockReturnValue({
        data: {},
      } as ReturnType<typeof useActiveConversation>);
    });

    it("shows messages immediately when agent-server events exist in store, even while loading", () => {
      // Simulate: history is loading but events already exist in store (e.g., remount)
      vi.mocked(useConversationWebSocket).mockReturnValue({
        isLoadingHistory: true,
        connectionState: "OPEN",
        sendMessage: vi.fn(),
        reconnect: vi.fn(),
      });

      // Put agent-server user events in the store
      const userEvent = createUserMessageEvent("evt-1");
      useEventStore.setState({
        events: [userEvent],
        eventIds: new Set(["evt-1"]),
        uiEvents: [userEvent],
      });

      renderWithQueryClient(<ChatInterface />, queryClient);

      // AC1: Messages should display immediately without the full history
      // skeleton, even if the lazy older-events pagination spinner appears.
      expect(
        screen.queryByTestId("chat-messages-skeleton"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("conversation-messages") ??
          screen.queryByTestId("user-message"),
      ).not.toBeNull();
    });

    it("shows skeleton when store is empty and loading", () => {
      // Simulate: first load, no events yet
      vi.mocked(useConversationWebSocket).mockReturnValue({
        isLoadingHistory: true,
        connectionState: "OPEN",
        sendMessage: vi.fn(),
        reconnect: vi.fn(),
      });

      // Store is empty
      useEventStore.setState({
        events: [],
        eventIds: new Set(),
        uiEvents: [],
      });

      renderWithQueryClient(<ChatInterface />, queryClient);

      // AC5: Genuine first-load shows skeleton
      expect(screen.getByTestId("chat-messages-skeleton")).toBeInTheDocument();
    });

    it("hides skeleton when a pending user message is visible during history load", () => {
      vi.mocked(useConversationWebSocket).mockReturnValue({
        isLoadingHistory: true,
        connectionState: "OPEN",
        sendMessage: vi.fn(),
        reconnect: vi.fn(),
      });

      useEventStore.setState({
        events: [],
        eventIds: new Set(),
        uiEvents: [],
      });

      useOptimisticUserMessageStore.getState().enqueuePendingMessage({
        conversationId: "test-conversation-id",
        text: "hello from home",
      });

      renderWithQueryClient(<ChatInterface />, queryClient);

      expect(
        screen.queryByTestId("chat-messages-skeleton"),
      ).not.toBeInTheDocument();
      expect(screen.getByTestId("user-message")).toHaveTextContent(
        "hello from home",
      );
    });

    it("shows messages when loading is already false on mount (edge case)", () => {
      // Simulate: component re-mounts when WebSocket has already finished loading
      vi.mocked(useConversationWebSocket).mockReturnValue({
        isLoadingHistory: false,
        connectionState: "OPEN",
        sendMessage: vi.fn(),
        reconnect: vi.fn(),
      });

      // agent-server events in store
      const userEvent = createUserMessageEvent("evt-2");
      useEventStore.setState({
        events: [userEvent],
        eventIds: new Set(["evt-1"]),
        uiEvents: [userEvent],
      });

      renderWithQueryClient(<ChatInterface />, queryClient);

      // Messages should display without the full history skeleton.
      expect(
        screen.queryByTestId("chat-messages-skeleton"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("conversation-messages") ??
          screen.queryByTestId("user-message"),
      ).not.toBeNull();
    });
  });
});
