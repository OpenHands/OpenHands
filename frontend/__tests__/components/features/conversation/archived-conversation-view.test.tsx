import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router";
import { I18nextProvider, initReactI18next } from "react-i18next";
import i18n from "i18next";
import { createUserMessageEvent } from "test-utils";
import { useEventStore } from "#/stores/use-event-store";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import { ArchivedConversationView } from "#/components/features/conversation/archived-conversation-view";

// Initialize i18n for tests
i18n.use(initReactI18next).init({
  lng: "en",
  fallbackLng: "en",
  ns: ["translation"],
  defaultNS: "translation",
  resources: {
    en: {
      translation: {
        CONVERSATION$ARCHIVED_READ_ONLY:
          "This conversation is archived and read-only",
        CONVERSATION$NO_HISTORY_AVAILABLE: "No history available",
      },
    },
  },
  interpolation: {
    escapeValue: false,
  },
});

// Module-level mocks
vi.mock("#/contexts/conversation-websocket-context");

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({ curAgentState: "STOPPED", isArchived: true }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: { sandbox_status: "MISSING" },
  }),
}));

vi.mock("#/hooks/query/use-task-polling", () => ({
  useTaskPolling: () => ({
    isTask: false,
    taskStatus: null,
  }),
}));

vi.mock("#/hooks/mutation/use-unified-stop-conversation", () => ({
  useUnifiedPauseConversationSandbox: () => ({
    mutate: vi.fn(),
  }),
}));

vi.mock("#/hooks/mutation/use-unified-start-conversation", () => ({
  useUnifiedResumeConversationSandbox: () => ({
    mutate: vi.fn(),
  }),
}));

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

// Mock the V1 Messages component to simplify testing
vi.mock("#/components/v1/chat", () => ({
  Messages: ({ messages }: { messages: unknown[] }) => (
    <div data-testid="v1-messages">
      {messages.map((_, index) => (
        <div key={index} data-testid="v1-message-item" />
      ))}
    </div>
  ),
  shouldRenderEvent: () => true,
  hasUserEvent: () => false,
}));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

const renderWithProviders = (ui: React.ReactElement) => {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <I18nextProvider i18n={i18n}>
        <MemoryRouter initialEntries={["/test-conversation-id"]}>
          <Routes>
            <Route path="/:conversationId" element={ui} />
          </Routes>
        </MemoryRouter>
      </I18nextProvider>
    </QueryClientProvider>,
  );
};

describe("ArchivedConversationView", () => {
  beforeEach(() => {
    // Reset event store before each test
    useEventStore.setState({
      events: [],
      uiEvents: [],
    });

    // Default: websocket context with no loading
    vi.mocked(useConversationWebSocket).mockReturnValue({
      isLoadingHistory: false,
      connectionState: "OPEN",
      sendMessage: vi.fn(),
    });
  });

  describe("loading state", () => {
    it("shows loading spinner when loading history and no events exist", () => {
      vi.mocked(useConversationWebSocket).mockReturnValue({
        isLoadingHistory: true,
        connectionState: "OPEN",
        sendMessage: vi.fn(),
      });

      useEventStore.setState({
        events: [],
        uiEvents: [],
      });

      renderWithProviders(<ArchivedConversationView />);

      expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
    });

    it("does not show loading spinner when events exist", () => {
      vi.mocked(useConversationWebSocket).mockReturnValue({
        isLoadingHistory: true,
        connectionState: "OPEN",
        sendMessage: vi.fn(),
      });

      const userEvent = createUserMessageEvent("evt-1");
      useEventStore.setState({
        events: [userEvent],
        uiEvents: [userEvent],
      });

      renderWithProviders(<ArchivedConversationView />);

      expect(screen.queryByTestId("loading-spinner")).not.toBeInTheDocument();
    });
  });

  describe("archived banner", () => {
    it("renders the archived banner", () => {
      renderWithProviders(<ArchivedConversationView />);

      expect(screen.getByTestId("archived-banner")).toBeInTheDocument();
      // The i18n key is rendered as-is when translations aren't fully loaded
      expect(
        screen.getByText(/ARCHIVED_READ_ONLY|This conversation is archived/i),
      ).toBeInTheDocument();
    });
  });

  describe("messages display", () => {
    it("shows empty state message when no events exist and not loading", () => {
      vi.mocked(useConversationWebSocket).mockReturnValue({
        isLoadingHistory: false,
        connectionState: "OPEN",
        sendMessage: vi.fn(),
      });

      useEventStore.setState({
        events: [],
        uiEvents: [],
      });

      renderWithProviders(<ArchivedConversationView />);

      // The i18n key is rendered as-is when translations aren't fully loaded
      expect(
        screen.getByText(/NO_HISTORY_AVAILABLE|No history available/i),
      ).toBeInTheDocument();
    });

    it("renders V1 messages when events exist", () => {
      const userEvent = createUserMessageEvent("evt-1");
      useEventStore.setState({
        events: [userEvent],
        uiEvents: [userEvent],
      });

      renderWithProviders(<ArchivedConversationView />);

      expect(screen.getByTestId("v1-messages")).toBeInTheDocument();
      expect(screen.getAllByTestId("v1-message-item")).toHaveLength(1);
    });

    it("renders multiple messages correctly", () => {
      const events = [
        createUserMessageEvent("evt-1"),
        createUserMessageEvent("evt-2"),
        createUserMessageEvent("evt-3"),
      ];
      useEventStore.setState({
        events,
        uiEvents: events,
      });

      renderWithProviders(<ArchivedConversationView />);

      expect(screen.getByTestId("v1-messages")).toBeInTheDocument();
      expect(screen.getAllByTestId("v1-message-item")).toHaveLength(3);
    });
  });

  describe("layout structure", () => {
    it("does not show loading spinner when conversation is loaded", () => {
      const userEvent = createUserMessageEvent("evt-1");
      useEventStore.setState({
        events: [userEvent],
        uiEvents: [userEvent],
      });

      renderWithProviders(<ArchivedConversationView />);

      // Should not have any loading spinner in the main view
      expect(screen.queryByTestId("loading-spinner")).not.toBeInTheDocument();
    });

    it("renders the complete layout with header and banner", () => {
      const userEvent = createUserMessageEvent("evt-1");
      useEventStore.setState({
        events: [userEvent],
        uiEvents: [userEvent],
      });

      renderWithProviders(<ArchivedConversationView />);

      // Should have archived banner
      expect(screen.getByTestId("archived-banner")).toBeInTheDocument();

      // Should have messages
      expect(screen.getByTestId("v1-messages")).toBeInTheDocument();
    });
  });
});
