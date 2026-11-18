/* eslint-disable i18next/no-literal-string -- test mocks intentionally use literal labels */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { ConversationCard } from "#/components/features/conversation-panel/conversation-card/conversation-card";
import { ConversationLayoutsMenu } from "#/components/features/conversation-panel/conversation-layouts-menu";
import {
  useConversationReadStateStore,
  INITIAL_CONVERSATION_READ_STATE,
} from "#/stores/conversation-read-state-store";

describe("Conversation Card Read / Unread State", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useConversationReadStateStore.setState(INITIAL_CONVERSATION_READ_STATE);
  });

  it("renders unread dot indicator for an unread conversation", () => {
    // Unopened conversation with lastUpdatedAt
    renderWithProviders(
      <ConversationCard
        conversationId="conv-unread-1"
        title="Test Conversation"
        selectedRepository={null}
        lastUpdatedAt={new Date().toISOString()}
      />,
    );

    const dot = screen.getByTestId("conversation-unread-dot");
    expect(dot).toBeInTheDocument();
  });

  it("does not render unread dot for an active conversation", () => {
    renderWithProviders(
      <ConversationCard
        conversationId="conv-active"
        title="Active Conversation"
        selectedRepository={null}
        lastUpdatedAt={new Date().toISOString()}
        isActive
      />,
    );

    expect(
      screen.queryByTestId("conversation-unread-dot"),
    ).not.toBeInTheDocument();
  });

  it("marks conversation as read when card is clicked", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();

    renderWithProviders(
      <ConversationCard
        conversationId="conv-click-to-read"
        title="Unread Conversation"
        selectedRepository={null}
        lastUpdatedAt={new Date().toISOString()}
        onClick={onClick}
      />,
    );

    expect(screen.getByTestId("conversation-unread-dot")).toBeInTheDocument();

    await user.click(screen.getByTestId("conversation-card"));
    expect(onClick).toHaveBeenCalled();

    const state = useConversationReadStateStore.getState();
    expect(state.lastReadAt["conv-click-to-read"]).toBeDefined();
  });

  it("displays 'Mark as read' in context menu when unread, and clicking marks as read", async () => {
    const user = userEvent.setup();
    const onContextMenuToggle = vi.fn();

    renderWithProviders(
      <ConversationCard
        conversationId="conv-ctx-read"
        title="Unread Conversation"
        selectedRepository={null}
        lastUpdatedAt={new Date().toISOString()}
        contextMenuOpen
        onContextMenuToggle={onContextMenuToggle}
      />,
    );

    const markAsReadBtn = screen.getByTestId("mark-as-read-button");
    expect(markAsReadBtn).toBeInTheDocument();
    expect(
      screen.queryByTestId("mark-as-unread-button"),
    ).not.toBeInTheDocument();

    await user.click(markAsReadBtn);

    const state = useConversationReadStateStore.getState();
    expect(state.lastReadAt["conv-ctx-read"]).toBeDefined();
    expect(onContextMenuToggle).toHaveBeenCalledWith(false);
  });

  it("displays 'Mark as unread' in context menu when read, and clicking marks as unread", async () => {
    const user = userEvent.setup();
    const onContextMenuToggle = vi.fn();

    // Mark as read first
    useConversationReadStateStore.getState().markAsRead("conv-ctx-unread");

    renderWithProviders(
      <ConversationCard
        conversationId="conv-ctx-unread"
        title="Read Conversation"
        selectedRepository={null}
        lastUpdatedAt={new Date(Date.now() - 50000).toISOString()}
        contextMenuOpen
        onContextMenuToggle={onContextMenuToggle}
      />,
    );

    const markAsUnreadBtn = screen.getByTestId("mark-as-unread-button");
    expect(markAsUnreadBtn).toBeInTheDocument();
    expect(screen.queryByTestId("mark-as-read-button")).not.toBeInTheDocument();

    await user.click(markAsUnreadBtn);

    const state = useConversationReadStateStore.getState();
    expect(state.manualUnread["conv-ctx-unread"]).toBe(true);
    expect(onContextMenuToggle).toHaveBeenCalledWith(false);
  });

  describe("ConversationLayoutsMenu 'Mark all as read'", () => {
    it("renders 'Mark all as read' option and triggers onMarkAllAsRead when clicked", async () => {
      const user = userEvent.setup();
      const onMarkAllAsRead = vi.fn();
      const setMenuOpen = vi.fn();

      renderWithProviders(
        <ConversationLayoutsMenu
          menuOpen
          setMenuOpen={setMenuOpen}
          menuRef={{ current: null }}
          backendKind="local"
          tagFacets={[]}
          automationNameFacets={[]}
          totalConversationsCount={5}
          onRequestDeleteAll={() => {}}
          onMarkAllAsRead={onMarkAllAsRead}
          hasUnreadConversations
        />,
      );

      const markAllBtn = screen.getByTestId("mark-all-conversations-as-read");
      expect(markAllBtn).toBeInTheDocument();
      expect(markAllBtn).not.toBeDisabled();

      await user.click(markAllBtn);
      expect(onMarkAllAsRead).toHaveBeenCalled();
      expect(setMenuOpen).toHaveBeenCalledWith(false);
    });

    it("disables 'Mark all as read' when there are no unread conversations", () => {
      renderWithProviders(
        <ConversationLayoutsMenu
          menuOpen
          setMenuOpen={() => {}}
          menuRef={{ current: null }}
          backendKind="local"
          tagFacets={[]}
          automationNameFacets={[]}
          totalConversationsCount={5}
          onRequestDeleteAll={() => {}}
          onMarkAllAsRead={() => {}}
          hasUnreadConversations={false}
        />,
      );

      const markAllBtn = screen.getByTestId("mark-all-conversations-as-read");
      expect(markAllBtn).toBeDisabled();
    });
  });
});
