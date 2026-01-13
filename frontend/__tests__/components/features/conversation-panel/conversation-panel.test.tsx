import { screen, waitFor, within } from "@testing-library/react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { createRoutesStub } from "react-router";
import React from "react";
import { renderWithProviders } from "test-utils";
import { ConversationPanel } from "#/components/features/conversation-panel/conversation-panel";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { Conversation } from "#/api/open-hands.types";

// Mock the unified stop conversation hook
const mockStopConversationMutate = vi.fn();
vi.mock("#/hooks/mutation/use-unified-stop-conversation", () => ({
  useUnifiedPauseConversationSandbox: () => ({
    mutate: mockStopConversationMutate,
  }),
}));

// Mock toast handlers to prevent unhandled rejection errors
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
  TOAST_OPTIONS: {},
}));

describe("ConversationPanel", () => {
  const onCloseMock = vi.fn();
  const RouterStub = createRoutesStub([
    {
      Component: () => <ConversationPanel onClose={onCloseMock} />,
      path: "/",
    },
    {
      // Add route to prevent "No routes matched location" warning
      Component: () => null,
      path: "/conversations/:conversationId",
    },
  ]);

  const renderConversationPanel = () => renderWithProviders(<RouterStub />);

  beforeAll(() => {
    vi.mock("react-router", async (importOriginal) => ({
      ...(await importOriginal<typeof import("react-router")>()),
      Link: ({ children }: React.PropsWithChildren) => children,
      useNavigate: vi.fn(() => vi.fn()),
      useLocation: vi.fn(() => ({ pathname: "/conversation" })),
      useParams: vi.fn(() => ({ conversationId: "2" })),
    }));
  });

  const mockConversations: Conversation[] = [
    {
      conversation_id: "1",
      title: "Conversation 1",
      selected_repository: null,
      git_provider: null,
      selected_branch: null,
      last_updated_at: "2021-10-01T12:00:00Z",
      created_at: "2021-10-01T12:00:00Z",
      status: "STOPPED" as const,
      runtime_status: null,
      url: null,
      session_api_key: null,
    },
    {
      conversation_id: "2",
      title: "Conversation 2",
      selected_repository: null,
      git_provider: null,
      selected_branch: null,
      last_updated_at: "2021-10-02T12:00:00Z",
      created_at: "2021-10-02T12:00:00Z",
      status: "STOPPED" as const,
      runtime_status: null,
      url: null,
      session_api_key: null,
    },
    {
      conversation_id: "3",
      title: "Conversation 3",
      selected_repository: null,
      git_provider: null,
      selected_branch: null,
      last_updated_at: "2021-10-03T12:00:00Z",
      created_at: "2021-10-03T12:00:00Z",
      status: "STOPPED" as const,
      runtime_status: null,
      url: null,
      session_api_key: null,
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    mockStopConversationMutate.mockClear();
    // Setup default mock for getUserConversations
    vi.spyOn(ConversationService, "getUserConversations").mockResolvedValue({
      results: [...mockConversations],
      next_page_id: null,
    });
  });

  it("should render the conversations", async () => {
    renderConversationPanel();
    const cards = await screen.findAllByTestId("conversation-card");

    // NOTE that we filter out conversations that don't have a created_at property
    // (mock data has 4 conversations, but only 3 have a created_at property)
    expect(cards).toHaveLength(3);
  });

  it("should display an empty state when there are no conversations", async () => {
    const getUserConversationsSpy = vi.spyOn(
      ConversationService,
      "getUserConversations",
    );
    getUserConversationsSpy.mockResolvedValue({
      results: [],
      next_page_id: null,
    });

    renderConversationPanel();

    const emptyState = await screen.findByText("CONVERSATION$NO_CONVERSATIONS");
    expect(emptyState).toBeInTheDocument();
  });

  it("should handle an error when fetching conversations", async () => {
    const getUserConversationsSpy = vi.spyOn(
      ConversationService,
      "getUserConversations",
    );
    getUserConversationsSpy.mockRejectedValue(
      new Error("Failed to fetch conversations"),
    );

    renderConversationPanel();

    const error = await screen.findByText("Failed to fetch conversations");
    expect(error).toBeInTheDocument();
  });

  it("should cancel deleting a conversation", async () => {
    const user = userEvent.setup();
    renderConversationPanel();

    let cards = await screen.findAllByTestId("conversation-card");
    // Delete button should not be visible initially (context menu is closed)
    // The context menu is always in the DOM but hidden by CSS classes on the parent div
    const contextMenuParent = within(cards[0]).queryByTestId(
      "context-menu",
    )?.parentElement;
    if (contextMenuParent) {
      expect(contextMenuParent).toHaveClass("opacity-0", "invisible");
    }

    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);
    const deleteButton = within(cards[0]).getByTestId("delete-button");

    // Click the first delete button
    await user.click(deleteButton);

    // Cancel the deletion
    const cancelButton = screen.getByRole("button", { name: /cancel/i });
    await user.click(cancelButton);

    expect(
      screen.queryByRole("button", { name: /cancel/i }),
    ).not.toBeInTheDocument();

    // Ensure the conversation is not deleted
    cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);
  });

  it("should delete a conversation", async () => {
    const user = userEvent.setup();
    const mockData: Conversation[] = [
      {
        conversation_id: "1",
        title: "Conversation 1",
        selected_repository: null,
        git_provider: null,
        selected_branch: null,
        last_updated_at: "2021-10-01T12:00:00Z",
        created_at: "2021-10-01T12:00:00Z",
        status: "STOPPED" as const,
        runtime_status: null,
        url: null,
        session_api_key: null,
      },
      {
        conversation_id: "2",
        title: "Conversation 2",
        selected_repository: null,
        git_provider: null,
        selected_branch: null,
        last_updated_at: "2021-10-02T12:00:00Z",
        created_at: "2021-10-02T12:00:00Z",
        status: "STOPPED" as const,
        runtime_status: null,
        url: null,
        session_api_key: null,
      },
      {
        conversation_id: "3",
        title: "Conversation 3",
        selected_repository: null,
        git_provider: null,
        selected_branch: null,
        last_updated_at: "2021-10-03T12:00:00Z",
        created_at: "2021-10-03T12:00:00Z",
        status: "STOPPED" as const,
        runtime_status: null,
        url: null,
        session_api_key: null,
      },
    ];

    const getUserConversationsSpy = vi.spyOn(
      ConversationService,
      "getUserConversations",
    );
    getUserConversationsSpy.mockImplementation(async () => ({
      results: mockData,
      next_page_id: null,
    }));

    const deleteUserConversationSpy = vi.spyOn(
      ConversationService,
      "deleteUserConversation",
    );
    deleteUserConversationSpy.mockImplementation(async (id: string) => {
      const index = mockData.findIndex((conv) => conv.conversation_id === id);
      if (index !== -1) {
        mockData.splice(index, 1);
      }
    });

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);

    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);
    const deleteButton = within(cards[0]).getByTestId("delete-button");

    // Click the first delete button
    await user.click(deleteButton);

    // Confirm the deletion
    const confirmButton = screen.getByRole("button", { name: /confirm/i });
    await user.click(confirmButton);

    expect(
      screen.queryByRole("button", { name: /confirm/i }),
    ).not.toBeInTheDocument();

    // Wait for the cards to update
    await waitFor(() => {
      const updatedCards = screen.getAllByTestId("conversation-card");
      expect(updatedCards).toHaveLength(2);
    });
  });

  it("should call onClose after clicking a card", async () => {
    const user = userEvent.setup();
    renderConversationPanel();
    const cards = await screen.findAllByTestId("conversation-card");
    const firstCard = cards[1];

    await user.click(firstCard);

    expect(onCloseMock).toHaveBeenCalledOnce();
  });

  it("should refetch data on rerenders", async () => {
    const user = userEvent.setup();
    const getUserConversationsSpy = vi.spyOn(
      ConversationService,
      "getUserConversations",
    );
    getUserConversationsSpy.mockResolvedValue({
      results: [...mockConversations],
      next_page_id: null,
    });

    function PanelWithToggle() {
      const [isOpen, setIsOpen] = React.useState(true);
      return (
        <>
          <button type="button" onClick={() => setIsOpen((prev) => !prev)}>
            Toggle
          </button>
          {isOpen && <ConversationPanel onClose={onCloseMock} />}
        </>
      );
    }

    const MyRouterStub = createRoutesStub([
      {
        Component: PanelWithToggle,
        path: "/",
      },
    ]);

    renderWithProviders(<MyRouterStub />);

    const toggleButton = screen.getByText("Toggle");

    // Initial render
    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);

    // Toggle off
    await user.click(toggleButton);
    expect(screen.queryByTestId("conversation-card")).not.toBeInTheDocument();

    // Toggle on
    await user.click(toggleButton);
    const newCards = await screen.findAllByTestId("conversation-card");
    expect(newCards).toHaveLength(3);
  });

  it("should cancel stopping a conversation", async () => {
    const user = userEvent.setup();

    // Create mock data with a RUNNING conversation
    const mockRunningConversations: Conversation[] = [
      {
        conversation_id: "1",
        title: "Running Conversation",
        selected_repository: null,
        git_provider: null,
        selected_branch: null,
        last_updated_at: "2021-10-01T12:00:00Z",
        created_at: "2021-10-01T12:00:00Z",
        status: "RUNNING" as const,
        runtime_status: null,
        url: null,
        session_api_key: null,
      },
      {
        conversation_id: "2",
        title: "Starting Conversation",
        selected_repository: null,
        git_provider: null,
        selected_branch: null,
        last_updated_at: "2021-10-02T12:00:00Z",
        created_at: "2021-10-02T12:00:00Z",
        status: "STARTING" as const,
        runtime_status: null,
        url: null,
        session_api_key: null,
      },
      {
        conversation_id: "3",
        title: "Stopped Conversation",
        selected_repository: null,
        git_provider: null,
        selected_branch: null,
        last_updated_at: "2021-10-03T12:00:00Z",
        created_at: "2021-10-03T12:00:00Z",
        status: "STOPPED" as const,
        runtime_status: null,
        url: null,
        session_api_key: null,
      },
    ];

    const getUserConversationsSpy = vi.spyOn(
      ConversationService,
      "getUserConversations",
    );
    getUserConversationsSpy.mockResolvedValue({
      results: mockRunningConversations,
      next_page_id: null,
    });

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);

    // Click ellipsis on the first card (RUNNING status)
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    // Stop button should be available for RUNNING conversation
    const stopButton = within(cards[0]).getByTestId("stop-button");
    expect(stopButton).toBeInTheDocument();

    // Click the stop button
    await user.click(stopButton);

    // Cancel the stopping action
    const cancelButton = screen.getByRole("button", { name: /cancel/i });
    await user.click(cancelButton);

    expect(
      screen.queryByRole("button", { name: /cancel/i }),
    ).not.toBeInTheDocument();

    // Ensure the conversation status hasn't changed
    const updatedCards = await screen.findAllByTestId("conversation-card");
    expect(updatedCards).toHaveLength(3);
  });

  it("should stop a conversation", async () => {
    const user = userEvent.setup();

    const mockData: Conversation[] = [
      {
        conversation_id: "1",
        title: "Running Conversation",
        selected_repository: null,
        git_provider: null,
        selected_branch: null,
        last_updated_at: "2021-10-01T12:00:00Z",
        created_at: "2021-10-01T12:00:00Z",
        status: "RUNNING" as const,
        runtime_status: null,
        url: null,
        session_api_key: null,
      },
      {
        conversation_id: "2",
        title: "Starting Conversation",
        selected_repository: null,
        git_provider: null,
        selected_branch: null,
        last_updated_at: "2021-10-02T12:00:00Z",
        created_at: "2021-10-02T12:00:00Z",
        status: "STARTING" as const,
        runtime_status: null,
        url: null,
        session_api_key: null,
      },
    ];

    const getUserConversationsSpy = vi.spyOn(
      ConversationService,
      "getUserConversations",
    );
    getUserConversationsSpy.mockImplementation(async () => ({
      results: mockData,
      next_page_id: null,
    }));

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(2);

    // Click ellipsis on the first card (RUNNING status)
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const stopButton = within(cards[0]).getByTestId("stop-button");

    // Click the stop button
    await user.click(stopButton);

    // Confirm the stopping action
    const confirmButton = screen.getByRole("button", { name: /confirm/i });
    await user.click(confirmButton);

    expect(
      screen.queryByRole("button", { name: /confirm/i }),
    ).not.toBeInTheDocument();

    // Verify the mutation was called
    expect(mockStopConversationMutate).toHaveBeenCalledWith({
      conversationId: "1",
      version: undefined,
    });
    expect(mockStopConversationMutate).toHaveBeenCalledTimes(1);
  });

  it("should only show stop button for STARTING or RUNNING conversations", async () => {
    const user = userEvent.setup();

    const mockMixedStatusConversations: Conversation[] = [
      {
        conversation_id: "1",
        title: "Running Conversation",
        selected_repository: null,
        git_provider: null,
        selected_branch: null,
        last_updated_at: "2021-10-01T12:00:00Z",
        created_at: "2021-10-01T12:00:00Z",
        status: "RUNNING" as const,
        runtime_status: null,
        url: null,
        session_api_key: null,
      },
      {
        conversation_id: "2",
        title: "Starting Conversation",
        selected_repository: null,
        git_provider: null,
        selected_branch: null,
        last_updated_at: "2021-10-02T12:00:00Z",
        created_at: "2021-10-02T12:00:00Z",
        status: "STARTING" as const,
        runtime_status: null,
        url: null,
        session_api_key: null,
      },
      {
        conversation_id: "3",
        title: "Stopped Conversation",
        selected_repository: null,
        git_provider: null,
        selected_branch: null,
        last_updated_at: "2021-10-03T12:00:00Z",
        created_at: "2021-10-03T12:00:00Z",
        status: "STOPPED" as const,
        runtime_status: null,
        url: null,
        session_api_key: null,
      },
    ];

    const getUserConversationsSpy = vi.spyOn(
      ConversationService,
      "getUserConversations",
    );
    getUserConversationsSpy.mockResolvedValue({
      results: mockMixedStatusConversations,
      next_page_id: null,
    });

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);

    // Test RUNNING conversation - should show stop button
    const runningEllipsisButton = within(cards[0]).getByTestId(
      "ellipsis-button",
    );
    await user.click(runningEllipsisButton);

    expect(within(cards[0]).getByTestId("stop-button")).toBeInTheDocument();

    // Click outside to close the menu
    await user.click(document.body);

    // Wait for context menu to close (check CSS classes on parent div)
    await waitFor(() => {
      const contextMenuParent = within(cards[0]).queryByTestId(
        "context-menu",
      )?.parentElement;
      if (contextMenuParent) {
        expect(contextMenuParent).toHaveClass("opacity-0", "invisible");
      }
    });

    // Test STARTING conversation - should show stop button
    const startingEllipsisButton = within(cards[1]).getByTestId(
      "ellipsis-button",
    );
    await user.click(startingEllipsisButton);

    expect(within(cards[1]).getByTestId("stop-button")).toBeInTheDocument();

    // Click outside to close the menu
    await user.click(document.body);

    // Wait for context menu to close (check CSS classes on parent div)
    await waitFor(() => {
      const contextMenuParent = within(cards[1]).queryByTestId(
        "context-menu",
      )?.parentElement;
      if (contextMenuParent) {
        expect(contextMenuParent).toHaveClass("opacity-0", "invisible");
      }
    });

    // Test STOPPED conversation - should NOT show stop button
    const stoppedEllipsisButton = within(cards[2]).getByTestId(
      "ellipsis-button",
    );
    await user.click(stoppedEllipsisButton);

    expect(
      within(cards[2]).queryByTestId("stop-button"),
    ).not.toBeInTheDocument();
  });

  it("should show edit button in context menu", async () => {
    const user = userEvent.setup();
    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);

    // Click ellipsis to open context menu
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    // Edit button should be visible within the first card's context menu
    const editButton = within(cards[0]).getByTestId("edit-button");
    expect(editButton).toBeInTheDocument();
    expect(editButton).toHaveTextContent("BUTTON$RENAME");
  });

  it("should enter edit mode when edit button is clicked", async () => {
    const user = userEvent.setup();
    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Click ellipsis to open context menu
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    // Click edit button within the first card's context menu
    const editButton = within(cards[0]).getByTestId("edit-button");
    await user.click(editButton);

    // Should find input field instead of title text
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    expect(titleInput).toBeInTheDocument();
    expect(titleInput.tagName).toBe("INPUT");
    expect(titleInput).toHaveValue("Conversation 1");
    expect(titleInput).toHaveFocus();
  });

  it("should successfully update conversation title", async () => {
    const user = userEvent.setup();

    // Mock the updateConversation API call
    const updateConversationSpy = vi.spyOn(
      ConversationService,
      "updateConversation",
    );
    updateConversationSpy.mockResolvedValue(true);

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = within(cards[0]).getByTestId("edit-button");
    await user.click(editButton);

    // Edit the title
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.type(titleInput, "Updated Title");

    // Blur the input to save
    await user.tab();

    // Verify API call was made with correct parameters
    expect(updateConversationSpy).toHaveBeenCalledWith("1", {
      title: "Updated Title",
    });
  });

  it("should save title when Enter key is pressed", async () => {
    const user = userEvent.setup();

    const updateConversationSpy = vi.spyOn(
      ConversationService,
      "updateConversation",
    );
    updateConversationSpy.mockResolvedValue(true);

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = within(cards[0]).getByTestId("edit-button");
    await user.click(editButton);

    // Edit the title and press Enter
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.type(titleInput, "Title Updated via Enter");
    await user.keyboard("{Enter}");

    // Verify API call was made
    expect(updateConversationSpy).toHaveBeenCalledWith("1", {
      title: "Title Updated via Enter",
    });
  });

  it("should trim whitespace from title", async () => {
    const user = userEvent.setup();

    const updateConversationSpy = vi.spyOn(
      ConversationService,
      "updateConversation",
    );
    updateConversationSpy.mockResolvedValue(true);

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = within(cards[0]).getByTestId("edit-button");
    await user.click(editButton);

    // Edit the title with extra whitespace
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.type(titleInput, "   Trimmed Title   ");
    await user.tab();

    // Verify API call was made with trimmed title
    expect(updateConversationSpy).toHaveBeenCalledWith("1", {
      title: "Trimmed Title",
    });
  });

  it("should revert to original title when empty", async () => {
    const user = userEvent.setup();

    const updateConversationSpy = vi.spyOn(
      ConversationService,
      "updateConversation",
    );
    updateConversationSpy.mockResolvedValue(true);

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = within(cards[0]).getByTestId("edit-button");
    await user.click(editButton);

    // Clear the title completely
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.tab();

    // Verify API was not called
    expect(updateConversationSpy).not.toHaveBeenCalled();
  });

  it("should handle API error when updating title", async () => {
    const user = userEvent.setup();

    const updateConversationSpy = vi.spyOn(
      ConversationService,
      "updateConversation",
    );
    updateConversationSpy.mockRejectedValue(new Error("API Error"));

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = within(cards[0]).getByTestId("edit-button");
    await user.click(editButton);

    // Edit the title
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.type(titleInput, "Failed Update");
    await user.tab();

    // Verify API call was made
    expect(updateConversationSpy).toHaveBeenCalledWith("1", {
      title: "Failed Update",
    });

    // Wait for error handling
    await waitFor(() => {
      expect(updateConversationSpy).toHaveBeenCalled();
    });
  });

  it("should close context menu when edit button is clicked", async () => {
    const user = userEvent.setup();
    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Click ellipsis to open context menu
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    // Verify context menu is open within the first card
    const contextMenu = within(cards[0]).getByTestId("context-menu");
    expect(contextMenu).toBeInTheDocument();

    // Click edit button within the first card's context menu
    const editButton = within(cards[0]).getByTestId("edit-button");
    await user.click(editButton);

    // Wait for context menu to close after edit button click (check CSS classes on parent div)
    await waitFor(() => {
      const contextMenuParent = within(cards[0]).queryByTestId(
        "context-menu",
      )?.parentElement;
      if (contextMenuParent) {
        expect(contextMenuParent).toHaveClass("opacity-0", "invisible");
      }
    });
  });

  it("should not call API when title is unchanged", async () => {
    const user = userEvent.setup();

    const updateConversationSpy = vi.spyOn(
      ConversationService,
      "updateConversation",
    );
    updateConversationSpy.mockResolvedValue(true);

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = within(cards[0]).getByTestId("edit-button");
    await user.click(editButton);

    // Don't change the title, just blur
    await user.tab();

    // Verify API was NOT called with the same title (since handleConversationTitleChange will always be called)
    expect(updateConversationSpy).not.toHaveBeenCalledWith("1", {
      title: "Conversation 1",
    });
  });

  it("should handle special characters in title", async () => {
    const user = userEvent.setup();

    const updateConversationSpy = vi.spyOn(
      ConversationService,
      "updateConversation",
    );
    updateConversationSpy.mockResolvedValue(true);

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = within(cards[0]).getByTestId("edit-button");
    await user.click(editButton);

    // Edit the title with special characters
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.type(titleInput, "Special @#$%^&*()_+ Characters");
    await user.tab();

    // Verify API call was made with special characters
    expect(updateConversationSpy).toHaveBeenCalledWith("1", {
      title: "Special @#$%^&*()_+ Characters",
    });
  });

  it("should close delete modal when clicking backdrop", async () => {
    const user = userEvent.setup();
    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Open context menu and click delete
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);
    const deleteButton = within(cards[0]).getByTestId("delete-button");
    await user.click(deleteButton);

    // Modal should be visible
    expect(
      screen.getByRole("button", { name: /confirm/i }),
    ).toBeInTheDocument();

    // Click the backdrop (the dark overlay behind the modal)
    const backdrop = document.querySelector(".bg-black.opacity-60");
    expect(backdrop).toBeInTheDocument();
    await user.click(backdrop!);

    // Modal should be closed
    expect(
      screen.queryByRole("button", { name: /confirm/i }),
    ).not.toBeInTheDocument();
  });

  it("should close stop modal when clicking backdrop", async () => {
    const user = userEvent.setup();

    // Create mock data with a RUNNING conversation
    const mockRunningConversations: Conversation[] = [
      {
        conversation_id: "1",
        title: "Running Conversation",
        selected_repository: null,
        git_provider: null,
        selected_branch: null,
        last_updated_at: "2021-10-01T12:00:00Z",
        created_at: "2021-10-01T12:00:00Z",
        status: "RUNNING" as const,
        runtime_status: null,
        url: null,
        session_api_key: null,
      },
    ];

    vi.spyOn(ConversationService, "getUserConversations").mockResolvedValue({
      results: mockRunningConversations,
      next_page_id: null,
    });

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Open context menu and click stop
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);
    const stopButton = within(cards[0]).getByTestId("stop-button");
    await user.click(stopButton);

    // Modal should be visible
    expect(
      screen.getByRole("button", { name: /confirm/i }),
    ).toBeInTheDocument();

    // Click the backdrop
    const backdrop = document.querySelector(".bg-black.opacity-60");
    expect(backdrop).toBeInTheDocument();
    await user.click(backdrop!);

    // Modal should be closed
    expect(
      screen.queryByRole("button", { name: /confirm/i }),
    ).not.toBeInTheDocument();
  });
});
