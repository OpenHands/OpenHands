import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import {
  setupConversationPanelTest,
  createMockConversation,
  onCloseMock,
  renderConversationPanel,
} from "./conversation-panel-test-utils";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { ExecutionStatus } from "#/types/agent-server/core";
import { useArchivedConversationsStore } from "#/stores/archived-conversations-store";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

const mockStopConversationMutate = vi.fn();
vi.mock("#/hooks/mutation/use-unified-stop-conversation", () => ({
  useUnifiedPauseConversation: () => ({ mutate: mockStopConversationMutate }),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
  TOAST_OPTIONS: {},
}));

describe("ConversationPanel conversation actions", () => {
  setupConversationPanelTest();

  it("should cancel deleting a conversation", async () => {
    const user = userEvent.setup();
    renderConversationPanel();

    let cards = await screen.findAllByTestId("conversation-card");
    // Closed state is observable via the data-context-menu-open attr on the
    // conversation-card root; visual hiding is covered by Playwright.
    expect(cards[0]).toHaveAttribute("data-context-menu-open", "false");

    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);
    const deleteButton = screen.getByTestId("delete-button");

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
    const mockData: import("#/api/conversation-service/agent-server-conversation-service.types").AppConversation[] = [
      createMockConversation({ id: "1", title: "Conversation 1" }),
      createMockConversation({ id: "2", title: "Conversation 2" }),
      createMockConversation({ id: "3", title: "Conversation 3" }),
    ];

    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockImplementation(async () => ({
      items: mockData,
      next_page_id: null,
    }));

    const deleteConversationSpy = vi.spyOn(
      AgentServerConversationService,
      "deleteConversation",
    );
    deleteConversationSpy.mockImplementation(async (id: string) => {
      const index = mockData.findIndex((conv) => conv.id === id);
      if (index !== -1) {
        mockData.splice(index, 1);
      }
    });

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    // Initially shows 3 conversations (no filtering)
    expect(cards).toHaveLength(3);

    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);
    const deleteButton = screen.getByTestId("delete-button");

    // Click the first delete button
    await user.click(deleteButton);

    // Confirm the deletion
    const confirmButton = screen.getByRole("button", { name: /confirm/i });
    await user.click(confirmButton);

    // Verify modal is closed after confirmation
    expect(
      screen.queryByRole("button", { name: /confirm/i }),
    ).not.toBeInTheDocument();
  });

  it("should archive a conversation and remove it from the list", async () => {
    const user = userEvent.setup();
    renderConversationPanel();

    let cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);

    const firstCardTitle = within(cards[0]).getByText("Conversation 1");
    expect(firstCardTitle).toBeInTheDocument();

    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);
    await user.click(screen.getByTestId("archive-button"));

    expect(
      screen.getByText("CONVERSATION$CONFIRM_ARCHIVE"),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /archive/i }));

    await waitFor(() => {
      expect(
        screen.queryByText("CONVERSATION$CONFIRM_ARCHIVE"),
      ).not.toBeInTheDocument();
    });

    cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(2);
    expect(screen.queryByText("Conversation 1")).not.toBeInTheDocument();
  });

  it("shows archived conversations when the preference is on and restores them", async () => {
    // Archiving must stay reversible — that is the whole distinction from
    // deleting, which removes the conversation from the agent server.
    const user = userEvent.setup();
    useArchivedConversationsStore
      .getState()
      .archiveConversation("default-local", "1");
    useConversationPanelPreferencesStore.setState({
      showArchivedConversations: true,
    });

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);
    const archivedCard = cards.find((card) =>
      within(card).queryByText("Conversation 1"),
    )!;
    expect(
      within(archivedCard).getByTestId("conversation-card-archived-chip"),
    ).toBeInTheDocument();

    await user.click(within(archivedCard).getByTestId("ellipsis-button"));
    await user.click(screen.getByTestId("unarchive-button"));

    await waitFor(() => {
      expect(
        useArchivedConversationsStore
          .getState()
          .isArchived("default-local", "1"),
      ).toBe(false);
    });
    expect(
      screen.queryByTestId("conversation-card-archived-chip"),
    ).not.toBeInTheDocument();
  });

  it("should cancel stopping a conversation", async () => {
    const user = userEvent.setup();

    // Create mock data with a RUNNING conversation
    const mockRunningConversations = [
      createMockConversation({
        id: "1",
        title: "Running Conversation",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "2",
        title: "Starting Conversation",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "3",
        title: "Stopped Conversation",
        execution_status: ExecutionStatus.PAUSED,
      }),
    ];

    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockResolvedValue({
      items: mockRunningConversations,
      next_page_id: null,
    });

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);

    // Click ellipsis on the first card (RUNNING status)
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    // Stop button should be available for RUNNING conversation
    const stopButton = screen.getByTestId("stop-button");
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

    const mockData = [
      createMockConversation({
        id: "1",
        title: "Conversation 1",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "2",
        title: "Conversation 2",
        execution_status: ExecutionStatus.FINISHED,
      }),
      createMockConversation({
        id: "3",
        title: "Conversation 3",
        execution_status: ExecutionStatus.FINISHED,
      }),
    ];

    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockImplementation(async () => ({
      items: mockData,
      next_page_id: null,
    }));

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    // Component shows all 3 conversations (no filtering by status)
    expect(cards).toHaveLength(3);

    // Click ellipsis on the first card (RUNNING status)
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const stopButton = screen.getByTestId("stop-button");

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
    });
  });

  it("should only show stop button for STARTING or RUNNING conversations", async () => {
    const user = userEvent.setup();

    const mockMixedStatusConversations = [
      createMockConversation({
        id: "1",
        title: "Running Conversation",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "2",
        title: "Starting Conversation",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "3",
        title: "Stopped Conversation",
        execution_status: ExecutionStatus.PAUSED,
      }),
    ];

    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockResolvedValue({
      items: mockMixedStatusConversations,
      next_page_id: null,
    });

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);

    const getCardByTitle = async (title: string) => {
      const currentCards = await screen.findAllByTestId("conversation-card");
      const card = currentCards.find((candidate) =>
        within(candidate).queryByText(title),
      );
      expect(card).toBeDefined();
      return card as HTMLElement;
    };

    // Test RUNNING conversation - should show stop button
    const runningCard = await getCardByTitle("Running Conversation");
    const runningEllipsisButton =
      within(runningCard).getByTestId("ellipsis-button");
    await user.click(runningEllipsisButton);

    expect(await screen.findByTestId("stop-button")).toBeInTheDocument();

    // Click outside to close the menu
    await user.click(document.body);

    // Wait for context menu to close before opening the next one.
    await waitFor(() => {
      expect(screen.queryByTestId("stop-button")).not.toBeInTheDocument();
    });

    // Test STARTING/RUNNING conversation - should show stop button
    const startingCard = await getCardByTitle("Starting Conversation");
    const startingEllipsisButton =
      within(startingCard).getByTestId("ellipsis-button");
    await user.click(startingEllipsisButton);

    expect(await screen.findByTestId("stop-button")).toBeInTheDocument();

    // Click outside to close the menu
    await user.click(document.body);

    // Wait for context menu to close before opening the next one.
    await waitFor(() => {
      expect(screen.queryByTestId("stop-button")).not.toBeInTheDocument();
    });

    // Test STOPPED conversation - should NOT show stop button
    const stoppedCard = await getCardByTitle("Stopped Conversation");
    const stoppedEllipsisButton =
      within(stoppedCard).getByTestId("ellipsis-button");
    await user.click(stoppedEllipsisButton);

    await waitFor(() => {
      expect(stoppedCard).toHaveAttribute("data-context-menu-open", "true");
    });
    expect(screen.queryByTestId("stop-button")).not.toBeInTheDocument();
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
    const editButton = screen.getByTestId("edit-button");
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
    const editButton = screen.getByTestId("edit-button");
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

    // Mock the updateConversationTitle API call
    const updateConversationTitleSpy = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTitle",
    );
    updateConversationTitleSpy.mockResolvedValue(
      createMockConversation({ id: "1", title: "Updated Title" }),
    );

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Edit the title
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.type(titleInput, "Updated Title");

    // Blur the input to save
    await user.tab();

    // Verify API call was made with correct parameters
    expect(updateConversationTitleSpy).toHaveBeenCalledWith(
      "1",
      "Updated Title",
    );
  });

  it("should save title when Enter key is pressed", async () => {
    const user = userEvent.setup();

    const updateConversationTitleSpy = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTitle",
    );
    updateConversationTitleSpy.mockResolvedValue(
      createMockConversation({ id: "1", title: "Updated Title" }),
    );

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Edit the title and press Enter
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.type(titleInput, "Title Updated via Enter");
    await user.keyboard("{Enter}");

    // Verify API call was made
    expect(updateConversationTitleSpy).toHaveBeenCalledWith(
      "1",
      "Title Updated via Enter",
    );
  });

  it("should trim whitespace from title", async () => {
    const user = userEvent.setup();

    const updateConversationTitleSpy = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTitle",
    );
    updateConversationTitleSpy.mockResolvedValue(
      createMockConversation({ id: "1", title: "Updated Title" }),
    );

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Edit the title with extra whitespace
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.type(titleInput, "   Trimmed Title   ");
    await user.tab();

    // Verify API call was made with trimmed title
    expect(updateConversationTitleSpy).toHaveBeenCalledWith(
      "1",
      "Trimmed Title",
    );
  });

  it("should revert to original title when empty", async () => {
    const user = userEvent.setup();

    const updateConversationTitleSpy = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTitle",
    );
    updateConversationTitleSpy.mockResolvedValue(
      createMockConversation({ id: "1", title: "Updated Title" }),
    );

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Clear the title completely
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.tab();

    // Verify API was not called
    expect(updateConversationTitleSpy).not.toHaveBeenCalled();
  });

  it("should handle API error when updating title", async () => {
    const user = userEvent.setup();

    const updateConversationTitleSpy = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTitle",
    );
    updateConversationTitleSpy.mockRejectedValue(new Error("API Error"));
    // Provide return type for mock

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Edit the title
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.type(titleInput, "Failed Update");
    await user.tab();

    // Verify API call was made
    expect(updateConversationTitleSpy).toHaveBeenCalledWith(
      "1",
      "Failed Update",
    );

    // Wait for error handling
    await waitFor(() => {
      expect(updateConversationTitleSpy).toHaveBeenCalled();
    });
  });

  it("should close context menu when edit button is clicked", async () => {
    const user = userEvent.setup();
    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Click ellipsis to open context menu
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    // Verify context menu is open (portaled to document.body)
    const contextMenu = screen.getByTestId("context-menu");
    expect(contextMenu).toBeInTheDocument();

    // Click edit button within the open context menu
    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Wait for context menu to close after edit button click.
    await waitFor(() => {
      expect(cards[0]).toHaveAttribute("data-context-menu-open", "false");
    });
  });

  it("should not call API when title is unchanged", async () => {
    const user = userEvent.setup();

    const updateConversationTitleSpy = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTitle",
    );
    updateConversationTitleSpy.mockResolvedValue(
      createMockConversation({ id: "1", title: "Updated Title" }),
    );

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Don't change the title, just blur
    await user.tab();

    // Verify API was NOT called with the same title (since handleConversationTitleChange will always be called)
    expect(updateConversationTitleSpy).not.toHaveBeenCalledWith("1", {
      title: "Conversation 1",
    });
  });

  it("should handle special characters in title", async () => {
    const user = userEvent.setup();

    const updateConversationTitleSpy = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTitle",
    );
    updateConversationTitleSpy.mockResolvedValue(
      createMockConversation({ id: "1", title: "Updated Title" }),
    );

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Edit the title with special characters
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.type(titleInput, "Special @#$%^&*()_+ Characters");
    await user.tab();

    // Verify API call was made with special characters
    expect(updateConversationTitleSpy).toHaveBeenCalledWith(
      "1",
      "Special @#$%^&*()_+ Characters",
    );
  });

  it("should close delete modal when clicking backdrop", async () => {
    const user = userEvent.setup();
    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Open context menu and click delete
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);
    const deleteButton = screen.getByTestId("delete-button");
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
    const mockRunningConversations = [
      createMockConversation({
        id: "1",
        title: "Running Conversation",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "2",
        title: "Starting Conversation",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "3",
        title: "Stopped Conversation",
        execution_status: ExecutionStatus.PAUSED,
      }),
    ];

    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: mockRunningConversations,
      next_page_id: null,
    });

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Open context menu and click stop
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);
    const stopButton = screen.getByTestId("stop-button");
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
