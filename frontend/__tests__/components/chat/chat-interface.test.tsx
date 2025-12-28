import {
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  test,
  vi,
} from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderWithProviders } from "test-utils";
import type { Message } from "#/message";
import { SUGGESTIONS } from "#/utils/suggestions";
import { ChatInterface } from "#/components/features/chat/chat-interface";
import { useWsClient } from "#/context/ws-client-provider";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { useConfig } from "#/hooks/query/use-config";
import { useGetTrajectory } from "#/hooks/mutation/use-get-trajectory";
import { useUnifiedUploadFiles } from "#/hooks/mutation/use-unified-upload-files";
import { OpenHandsAction } from "#/types/core/actions";
import { useEventStore } from "#/stores/use-event-store";

vi.mock("#/context/ws-client-provider");
vi.mock("#/hooks/query/use-config");
vi.mock("#/hooks/mutation/use-get-trajectory");
vi.mock("#/hooks/mutation/use-unified-upload-files");

// Mock React Router hooks at the top level
vi.mock("react-router", async () => {
  const actual = await vi.importActual("react-router");
  return {
    ...actual,
    useNavigate: () => vi.fn(),
    useParams: () => ({ conversationId: "test-conversation-id" }),
    useRouteLoaderData: vi.fn(() => ({})),
  };
});

// Mock other hooks that might be used by the component
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

// Helper function to render with Router context
const renderChatInterfaceWithRouter = () =>
  renderWithProviders(
    <MemoryRouter>
      <ChatInterface />
    </MemoryRouter>,
  );

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const renderChatInterface = (messages: Message[]) =>
  renderWithProviders(
    <MemoryRouter>
      <ChatInterface />
    </MemoryRouter>,
  );

// Helper function to render with QueryClientProvider and Router (for newer tests)
const renderWithQueryClient = (
  ui: React.ReactElement,
  queryClient: QueryClient,
) =>
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );

describe("ChatInterface - Chat Suggestions", () => {
  // Create a new QueryClient for each test
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    (useWsClient as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      send: vi.fn(),
      isLoadingMessages: false,
      parsedEvents: [],
    });

    useOptimisticUserMessageStore.setState({
      optimisticUserMessage: null,
    });

    useErrorMessageStore.setState({
      errorMessage: null,
    });

    (useConfig as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { APP_MODE: "local" },
    });
    (useGetTrajectory as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useUnifiedUploadFiles as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      mutateAsync: vi
        .fn()
        .mockResolvedValue({ skipped_files: [], uploaded_files: [] }),
      isLoading: false,
    });
  });

  test("should show chat suggestions when there are no events", () => {
    (useWsClient as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      send: vi.fn(),
      isLoadingMessages: false,
      parsedEvents: [],
    });

    renderWithQueryClient(<ChatInterface />, queryClient);

    // Check if ChatSuggestions is rendered
    expect(screen.getByTestId("chat-suggestions")).toBeInTheDocument();
  });

  test("should show chat suggestions when there are only environment events", () => {
    const environmentEvent: OpenHandsAction = {
      id: 1,
      source: "environment",
      action: "system",
      args: {
        content: "source .openhands/setup.sh",
        tools: null,
        openhands_version: null,
        agent_class: null,
      },
      message: "Running setup script",
      timestamp: "2025-07-01T00:00:00Z",
    };

    (useWsClient as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      send: vi.fn(),
      isLoadingMessages: false,
      parsedEvents: [environmentEvent],
    });

    renderWithQueryClient(<ChatInterface />, queryClient);

    // Check if ChatSuggestions is still rendered with environment events
    expect(screen.getByTestId("chat-suggestions")).toBeInTheDocument();
  });

  test("should hide chat suggestions when there is a user message", () => {
    const mockUserEvent: OpenHandsAction = {
      id: 1,
      source: "user",
      action: "message",
      args: {
        content: "Hello",
        image_urls: [],
        file_urls: [],
      },
      message: "Hello",
      timestamp: "2025-07-01T00:00:00Z",
    };

    useEventStore.setState({
      events: [mockUserEvent],
      uiEvents: [],
      addEvent: vi.fn(),
      clearEvents: vi.fn(),
    });

    renderWithQueryClient(<ChatInterface />, queryClient);

    // Check if ChatSuggestions is not rendered with user events
    expect(screen.queryByTestId("chat-suggestions")).not.toBeInTheDocument();
  });

  test("should hide chat suggestions when there is an optimistic user message", () => {
    useOptimisticUserMessageStore.setState({
      optimisticUserMessage: "Optimistic message",
    });

    renderWithQueryClient(<ChatInterface />, queryClient);

    // Check if ChatSuggestions is not rendered with optimistic user message
    expect(screen.queryByTestId("chat-suggestions")).not.toBeInTheDocument();
  });
});

describe("ChatInterface - Empty state", () => {
  const { send: sendMock } = vi.hoisted(() => ({
    send: vi.fn(),
  }));

  const { useWsClient: useWsClientMock } = vi.hoisted(() => ({
    useWsClient: vi.fn(() => ({
      send: sendMock,
      status: "CONNECTED",
      isLoadingMessages: false,
      parsedEvents: [],
    })),
  }));

  beforeAll(() => {
    vi.mock("#/context/socket", async (importActual) => ({
      ...(await importActual<typeof import("#/context/ws-client-provider")>()),
      useWsClient: useWsClientMock,
    }));
  });

  beforeEach(() => {
    (useWsClient as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      send: sendMock,
      status: "CONNECTED",
      isLoadingMessages: false,
      parsedEvents: [],
    });

    useOptimisticUserMessageStore.setState({
      optimisticUserMessage: null,
    });

    useErrorMessageStore.setState({
      errorMessage: null,
    });
    (useConfig as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { APP_MODE: "local" },
    });
    (useGetTrajectory as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      mutateAsync: vi.fn(),
      isLoading: false,
    });
    (useUnifiedUploadFiles as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      mutateAsync: vi
        .fn()
        .mockResolvedValue({ skipped_files: [], uploaded_files: [] }),
      isLoading: false,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it.todo("should render suggestions if empty");

  it("should render the default suggestions", () => {
    renderChatInterfaceWithRouter();

    const suggestions = screen.getByTestId("chat-suggestions");
    const repoSuggestions = Object.keys(SUGGESTIONS.repo);

    // check that there are at most 4 suggestions displayed
    const displayedSuggestions = within(suggestions).getAllByRole("button");
    expect(displayedSuggestions.length).toBeLessThanOrEqual(4);

    // Check that each displayed suggestion is one of the repo suggestions
    displayedSuggestions.forEach((suggestion) => {
      expect(repoSuggestions).toContain(suggestion.textContent);
    });
  });

  it.fails(
    "should load the a user message to the input when selecting",
    async () => {
      // this is to test that the message is in the UI before the socket is called
      useWsClientMock.mockImplementation(() => ({
        send: sendMock,
        status: "CONNECTED",
        isLoadingMessages: false,
        parsedEvents: [],
      }));
      const user = userEvent.setup();
      renderChatInterfaceWithRouter();

      const suggestions = screen.getByTestId("chat-suggestions");
      const displayedSuggestions = within(suggestions).getAllByRole("button");
      const input = screen.getByTestId("chat-input");

      await user.click(displayedSuggestions[0]);

      // user message loaded to input
      expect(screen.queryByTestId("chat-suggestions")).toBeInTheDocument();
      expect(input).toHaveValue(displayedSuggestions[0].textContent);
    },
  );

  it.fails(
    "should send the message to the socket only if the runtime is active",
    async () => {
      useWsClientMock.mockImplementation(() => ({
        send: sendMock,
        status: "CONNECTED",
        isLoadingMessages: false,
        parsedEvents: [],
      }));
      const user = userEvent.setup();
      const { rerender } = renderChatInterfaceWithRouter();

      const suggestions = screen.getByTestId("chat-suggestions");
      const displayedSuggestions = within(suggestions).getAllByRole("button");

      await user.click(displayedSuggestions[0]);
      expect(sendMock).not.toHaveBeenCalled();

      useWsClientMock.mockImplementation(() => ({
        send: sendMock,
        status: "CONNECTED",
        isLoadingMessages: false,
        parsedEvents: [],
      }));
      rerender(
        <MemoryRouter>
          <ChatInterface />
        </MemoryRouter>,
      );

      await waitFor(() =>
        expect(sendMock).toHaveBeenCalledWith(expect.any(String)),
      );
    },
  );
});

describe.skip("ChatInterface - General functionality", () => {
  beforeAll(() => {
    // mock useScrollToBottom hook
    vi.mock("#/hooks/useScrollToBottom", () => ({
      useScrollToBottom: vi.fn(() => ({
        scrollDomToBottom: vi.fn(),
        onChatBodyScroll: vi.fn(),
        hitBottom: vi.fn(),
      })),
    }));
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render messages", () => {
    const messages: Message[] = [
      {
        sender: "user",
        content: "Hello",
        imageUrls: [],
        timestamp: new Date().toISOString(),
        pending: true,
      },
      {
        sender: "assistant",
        content: "Hi",
        imageUrls: [],
        timestamp: new Date().toISOString(),
        pending: true,
      },
    ];
    renderChatInterface(messages);

    expect(screen.getAllByTestId(/-message/)).toHaveLength(2);
  });

  it("should render a chat input", () => {
    const messages: Message[] = [];
    renderChatInterface(messages);

    expect(screen.getByTestId("chat-input")).toBeInTheDocument();
  });

  it("should call socket send when submitting a message", async () => {
    const user = userEvent.setup();
    const messages: Message[] = [];
    renderChatInterface(messages);

    const input = screen.getByTestId("chat-input");
    await user.type(input, "Hello");
    await user.keyboard("{Enter}");

    // spy on send and expect to have been called
  });

  it("should render an image carousel with a message", () => {
    let messages: Message[] = [
      {
        sender: "assistant",
        content: "Here are some images",
        imageUrls: [],
        timestamp: new Date().toISOString(),
        pending: true,
      },
    ];
    const { rerender } = renderChatInterface(messages);

    expect(screen.queryByTestId("image-carousel")).not.toBeInTheDocument();

    messages = [
      {
        sender: "assistant",
        content: "Here are some images",
        imageUrls: ["image1", "image2"],
        timestamp: new Date().toISOString(),
        pending: true,
      },
    ];

    rerender(
      <MemoryRouter>
        <ChatInterface />
      </MemoryRouter>,
    );

    const imageCarousel = screen.getByTestId("image-carousel");
    expect(imageCarousel).toBeInTheDocument();
    expect(within(imageCarousel).getAllByTestId("image-preview")).toHaveLength(
      2,
    );
  });

  it("should render a 'continue' action when there are more than 2 messages and awaiting user input", () => {
    const messages: Message[] = [
      {
        sender: "assistant",
        content: "Hello",
        imageUrls: [],
        timestamp: new Date().toISOString(),
        pending: true,
      },
      {
        sender: "user",
        content: "Hi",
        imageUrls: [],
        timestamp: new Date().toISOString(),
        pending: true,
      },
    ];
    const { rerender } = renderChatInterface(messages);
    expect(
      screen.queryByTestId("continue-action-button"),
    ).not.toBeInTheDocument();

    messages.push({
      sender: "assistant",
      content: "How can I help you?",
      imageUrls: [],
      timestamp: new Date().toISOString(),
      pending: true,
    });

    rerender(
      <MemoryRouter>
        <ChatInterface />
      </MemoryRouter>,
    );

    expect(screen.getByTestId("continue-action-button")).toBeInTheDocument();
  });

  it("should render inline errors", () => {
    const messages: Message[] = [
      {
        sender: "assistant",
        content: "Hello",
        imageUrls: [],
        timestamp: new Date().toISOString(),
        pending: true,
      },
      {
        type: "error",
        content: "Something went wrong",
        sender: "assistant",
        timestamp: new Date().toISOString(),
      },
    ];
    renderChatInterface(messages);

    const error = screen.getByTestId("error-message");
    expect(within(error).getByText("Something went wrong")).toBeInTheDocument();
  });

  it("should render both GitHub buttons initially when ghToken is available", () => {
    // Note: This test may need adjustment since useRouteLoaderData is now globally mocked

    const messages: Message[] = [
      {
        sender: "assistant",
        content: "Hello",
        imageUrls: [],
        timestamp: new Date().toISOString(),
        pending: true,
      },
    ];
    renderChatInterface(messages);

    const pushButton = screen.getByRole("button", { name: "Push to Branch" });
    const prButton = screen.getByRole("button", { name: "Push & Create PR" });

    expect(pushButton).toBeInTheDocument();
    expect(prButton).toBeInTheDocument();
    expect(pushButton).toHaveTextContent("Push to Branch");
    expect(prButton).toHaveTextContent("Push & Create PR");
  });

  it("should render only 'Push changes to PR' button after PR is created", async () => {
    // Note: This test may need adjustment since useRouteLoaderData is now globally mocked

    const messages: Message[] = [
      {
        sender: "assistant",
        content: "Hello",
        imageUrls: [],
        timestamp: new Date().toISOString(),
        pending: true,
      },
    ];
    const { rerender } = renderChatInterface(messages);
    const user = userEvent.setup();

    // Click the "Push & Create PR" button
    const prButton = screen.getByRole("button", { name: "Push & Create PR" });
    await user.click(prButton);

    // Re-render to trigger state update
    rerender(
      <MemoryRouter>
        <ChatInterface />
      </MemoryRouter>,
    );

    // Verify only one button is shown
    const pushToPrButton = screen.getByRole("button", {
      name: "Push changes to PR",
    });
    expect(pushToPrButton).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Push to Branch" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Push & Create PR" }),
    ).not.toBeInTheDocument();
  });

  it("should render feedback actions if there are more than 3 messages", () => {
    const messages: Message[] = [
      {
        sender: "assistant",
        content: "Hello",
        imageUrls: [],
        timestamp: new Date().toISOString(),
        pending: true,
      },
      {
        sender: "user",
        content: "Hi",
        imageUrls: [],
        timestamp: new Date().toISOString(),
        pending: true,
      },
      {
        sender: "assistant",
        content: "How can I help you?",
        imageUrls: [],
        timestamp: new Date().toISOString(),
        pending: true,
      },
    ];
    const { rerender } = renderChatInterface(messages);
    expect(screen.queryByTestId("feedback-actions")).not.toBeInTheDocument();

    messages.push({
      sender: "user",
      content: "I need help",
      imageUrls: [],
      timestamp: new Date().toISOString(),
      pending: true,
    });

    rerender(
      <MemoryRouter>
        <ChatInterface />
      </MemoryRouter>,
    );

    expect(screen.getByTestId("feedback-actions")).toBeInTheDocument();
  });
});
