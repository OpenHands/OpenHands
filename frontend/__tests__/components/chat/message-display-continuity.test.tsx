import { describe, expect, it, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { QueryClient } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router";
import { render } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { useParamsMock, createUserMessageEvent } from "test-utils";
import { ChatInterface } from "#/components/features/chat/chat-interface";
import { useWsClient } from "#/context/ws-client-provider";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import { useConfig } from "#/hooks/query/use-config";
import { useGetTrajectory } from "#/hooks/mutation/use-get-trajectory";
import { useUnifiedUploadFiles } from "#/hooks/mutation/use-unified-upload-files";
import { useEventStore } from "#/stores/use-event-store";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import { OpenHandsAction } from "#/types/core/actions";

// Module-level mocks
vi.mock("#/context/ws-client-provider");
vi.mock("#/hooks/query/use-config");
vi.mock("#/hooks/mutation/use-get-trajectory");
vi.mock("#/hooks/mutation/use-unified-upload-files");
vi.mock("#/hooks/use-conversation-id");
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

// V0 user event (numeric id, action property)
const createV0UserEvent = (): OpenHandsAction => ({
  id: 1,
  source: "user",
  action: "message",
  args: {
    content: "Hello from V0",
    image_urls: [],
    file_urls: [],
  },
  message: "Hello from V0",
  timestamp: "2025-07-01T00:00:00Z",
});

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

    // Default: V0, no loading, no events
    (useWsClient as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      send: vi.fn(),
      isLoadingMessages: false,
      parsedEvents: [],
    });

    (useConfig as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { app_mode: "local" },
    });
    (useGetTrajectory as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (
      useUnifiedUploadFiles as unknown as ReturnType<typeof vi.fn>
    ).mockReturnValue({
      mutateAsync: vi
        .fn()
        .mockResolvedValue({ skipped_files: [], uploaded_files: [] }),
      isLoading: false,
    });

    // Default: no conversation (V0 behavior)
    vi.mocked(useActiveConversation).mockReturnValue({
      data: undefined,
    } as ReturnType<typeof useActiveConversation>);

    // Default: no websocket context
    vi.mocked(useConversationWebSocket).mockReturnValue(null);
  });

  describe("V1 conversations", () => {
    beforeEach(() => {
      // Set up V1 conversation
      vi.mocked(useActiveConversation).mockReturnValue({
        data: { conversation_version: "V1" },
      } as ReturnType<typeof useActiveConversation>);
    });

    it("shows messages immediately when V1 events exist in store, even while loading", () => {
      // Simulate: history is loading but events already exist in store (e.g., remount)
      vi.mocked(useConversationWebSocket).mockReturnValue({
        isLoadingHistory: true,
        connectionState: "OPEN",
        sendMessage: vi.fn(),
      });

      // Put V1 user events in the store
      const v1UserEvent = createUserMessageEvent("evt-1");
      useEventStore.setState({
        events: [v1UserEvent],
        uiEvents: [v1UserEvent],
      });

      renderWithQueryClient(<ChatInterface />, queryClient);

      // AC1: Messages should display immediately without skeleton
      expect(
        screen.queryByTestId("chat-messages-skeleton"),
      ).not.toBeInTheDocument();
      expect(screen.queryByTestId("loading-spinner")).not.toBeInTheDocument();
    });

    it("shows skeleton when store is empty and loading", () => {
      // Simulate: first load, no events yet
      vi.mocked(useConversationWebSocket).mockReturnValue({
        isLoadingHistory: true,
        connectionState: "OPEN",
        sendMessage: vi.fn(),
      });

      // Store is empty
      useEventStore.setState({
        events: [],
        uiEvents: [],
      });

      renderWithQueryClient(<ChatInterface />, queryClient);

      // AC5: Genuine first-load shows skeleton
      expect(screen.getByTestId("chat-messages-skeleton")).toBeInTheDocument();
    });

    it("shows messages when loading is already false on mount (edge case)", () => {
      // Simulate: component re-mounts when WebSocket has already finished loading
      vi.mocked(useConversationWebSocket).mockReturnValue({
        isLoadingHistory: false,
        connectionState: "OPEN",
        sendMessage: vi.fn(),
      });

      // V1 events in store
      const v1UserEvent = createUserMessageEvent("evt-2");
      useEventStore.setState({
        events: [v1UserEvent],
        uiEvents: [v1UserEvent],
      });

      renderWithQueryClient(<ChatInterface />, queryClient);

      // Messages should display, no skeleton
      expect(
        screen.queryByTestId("chat-messages-skeleton"),
      ).not.toBeInTheDocument();
      expect(screen.queryByTestId("loading-spinner")).not.toBeInTheDocument();
    });
  });

  describe("V0 conversations", () => {
    it("shows messages when V0 events exist in store even if isLoadingMessages is true", () => {
      // Simulate: loading flag is still true but events already exist in store (e.g., remount)
      (useWsClient as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
        send: vi.fn(),
        isLoadingMessages: true,
        parsedEvents: [],
      });

      // Put V0 user events in the store
      useEventStore.setState({
        events: [createV0UserEvent()],
        uiEvents: [],
      });

      renderWithQueryClient(<ChatInterface />, queryClient);

      // AC1/AC4: Messages display immediately, no skeleton
      expect(
        screen.queryByTestId("chat-messages-skeleton"),
      ).not.toBeInTheDocument();
    });

    it("shows skeleton when store is empty and isLoadingMessages is true", () => {
      // Simulate: genuine first load, no events yet
      (useWsClient as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
        send: vi.fn(),
        isLoadingMessages: true,
        parsedEvents: [],
      });

      // Store is empty
      useEventStore.setState({
        events: [],
        uiEvents: [],
      });

      renderWithQueryClient(<ChatInterface />, queryClient);

      // AC5: Genuine first-load shows skeleton
      expect(screen.getByTestId("chat-messages-skeleton")).toBeInTheDocument();
    });
  });
});
