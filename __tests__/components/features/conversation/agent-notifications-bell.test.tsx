import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AgentNotificationsBell } from "#/components/features/conversation/agent-notifications-bell";
import {
  AGENT_NOTIFICATIONS_HISTORY_STORAGE_KEY,
  AGENT_NOTIFICATIONS_SEEN_STORAGE_KEY,
} from "#/components/features/chat/agent-notifications-storage";
import type { AgentNotification } from "#/components/features/chat/agent-notifications.constants";
import { I18nKey } from "#/i18n/declaration";
import { useAgentNotificationsStore } from "#/stores/use-agent-notifications-store";
import { useEventStore, type OHEvent } from "#/stores/use-event-store";
import { renderWithProviders } from "test-utils";

const conversationId = "test-conversation-id";

const history: AgentNotification[] = [
  {
    id: "skill-standup",
    kind: "skill",
    name: "Standup digest helper",
    prompt: 'Save a reusable skill named "Standup digest helper".',
    createdAt: "2026-01-01T00:00:00.000Z",
  },
  {
    id: "workflow-ci",
    kind: "workflow",
    name: "CI failure watchdog",
    prompt: "Create a workflow named CI failure watchdog.",
    createdAt: "2026-01-01T00:00:01.000Z",
  },
];

const mockSend = vi.fn();

vi.mock("#/hooks/use-send-message", () => ({
  useSendMessage: () => ({ send: mockSend }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
}));

vi.mock("#/components/features/chat/agent-notifications.constants", async () => {
  const actual = await vi.importActual<
    typeof import("#/components/features/chat/agent-notifications.constants")
  >("#/components/features/chat/agent-notifications.constants");

  return {
    ...actual,
    isAgentNotificationsStagingEnabled: () => false,
  };
});

describe("AgentNotificationsBell", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.removeItem(AGENT_NOTIFICATIONS_HISTORY_STORAGE_KEY);
    window.localStorage.removeItem(AGENT_NOTIFICATIONS_SEEN_STORAGE_KEY);
    useAgentNotificationsStore.setState({
      historyByConversation: { [conversationId]: [] },
      seenByConversation: { [conversationId]: [] },
    });
    useEventStore.setState({
      events: [],
      eventIds: new Set(),
      uiEvents: [],
      loadedConversationId: null,
    });
  });

  it("renders the bell even when there is no recommendation history", () => {
    renderWithProviders(
      <AgentNotificationsBell conversationId={conversationId} />,
    );

    expect(screen.getByTestId("agent-notifications-bell")).toBeInTheDocument();
    expect(
      screen.queryByTestId("agent-notifications-bell-badge"),
    ).not.toBeInTheDocument();
  });

  it("opens a dropdown with an empty state when history is empty", async () => {
    const user = userEvent.setup();

    renderWithProviders(
      <AgentNotificationsBell conversationId={conversationId} />,
    );

    await user.click(screen.getByTestId("agent-notifications-bell"));

    expect(screen.getByTestId("agent-notifications-bell")).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(
      screen.getByTestId("agent-notifications-bell-dropdown"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("agent-notifications-bell-info")).toBeInTheDocument();
    expect(
      screen.queryByText(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_DESCRIPTION),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("agent-notifications-bell-empty-state"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("agent-notifications-bell-detect"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_EMPTY_TITLE),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("agent-notifications-bell-create-all"),
    ).not.toBeInTheDocument();
  });

  it("shows persisted history in the dropdown and a badge for unseen items", async () => {
    const user = userEvent.setup();

    useAgentNotificationsStore.setState({
      historyByConversation: { [conversationId]: history },
      seenByConversation: { [conversationId]: ["skill-standup"] },
    });

    renderWithProviders(
      <AgentNotificationsBell conversationId={conversationId} />,
    );

    expect(
      screen.getByTestId("agent-notifications-bell-badge"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("agent-notifications-bell"));

    expect(screen.getByTestId("agent-notifications-bell")).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(
      screen.getByTestId("agent-notifications-bell-item-skill-standup"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("agent-notifications-bell-item-workflow-ci"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("agent-notifications-bell-create-all"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("agent-notifications-bell-empty-state"),
    ).not.toBeInTheDocument();
  });

  it("removes a notification from persisted history", async () => {
    const user = userEvent.setup();

    useAgentNotificationsStore.setState({
      historyByConversation: { [conversationId]: history },
      seenByConversation: { [conversationId]: [] },
    });

    renderWithProviders(
      <AgentNotificationsBell conversationId={conversationId} />,
    );

    await user.click(screen.getByTestId("agent-notifications-bell"));
    await user.click(
      screen.getByTestId("agent-notifications-bell-item-remove-skill-standup"),
    );

    expect(
      screen.queryByTestId("agent-notifications-bell-item-skill-standup"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("agent-notifications-bell-item-workflow-ci"),
    ).toBeInTheDocument();
    expect(
      useAgentNotificationsStore.getState().historyByConversation[conversationId],
    ).toEqual([history[1]]);
  });

  it("populates recommendations when Scan conversation is clicked", async () => {
    const user = userEvent.setup();

    const fileEditEvent = {
      id: "file-edit-1",
      source: "agent",
      timestamp: "2026-01-01T00:00:00.000Z",
      tool_name: "file_editor",
      tool_call_id: "call-file-edit-1",
      action: {
        kind: "FileEditorAction",
        command: "str_replace",
        path: "/workspace/project/src/utils/format.ts",
        file_text: null,
        old_str: "a",
        new_str: "b",
        insert_line: null,
        view_range: null,
      },
    } as OHEvent;

    useEventStore.setState({
      events: [fileEditEvent],
      eventIds: new Set(["file-edit-1"]),
      uiEvents: [fileEditEvent],
      loadedConversationId: conversationId,
    });

    renderWithProviders(
      <AgentNotificationsBell conversationId={conversationId} />,
    );

    await user.click(screen.getByTestId("agent-notifications-bell"));
    await user.click(screen.getByTestId("agent-notifications-bell-detect"));

    expect(
      screen.getByTestId("agent-notifications-bell-item-detected-skill-format"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Format helper"),
    ).toBeInTheDocument();
  });
});
