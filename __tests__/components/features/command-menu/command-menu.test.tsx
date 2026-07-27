import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { CommandMenu } from "#/components/features/command-menu";
import { COMMAND_MENU_ROUTE } from "#/components/features/command-menu/command-menu-items";
import * as searchMatchingConversationsModule from "#/api/conversation-service/search-matching-conversations";
import { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ExecutionStatus } from "#/types/agent-server/core";
import { useCommandMenuStore } from "#/stores/command-menu-store";
import { useSidebarStore } from "#/stores/sidebar-store";
import { renderWithProviders } from "../../../../test-utils";

const SEARCH_LABEL_KEY = "COMMAND_MENU$SEARCH_LABEL";
const AUTOMATIONS_TITLE_KEY = "COMMAND_MENU$AUTOMATIONS_TITLE";
const NEW_CHAT_TITLE_KEY = "COMMAND_MENU$NEW_CHAT_TITLE";
const SECRETS_TITLE_KEY = "COMMAND_MENU$SECRETS_SETTINGS_TITLE";
const TOGGLE_SIDEBAR_TITLE_KEY = "COMMAND_MENU$TOGGLE_SIDEBAR_TITLE";
const CONVERSATIONS_GROUP_KEY = "COMMAND_MENU$GROUP_CONVERSATIONS";
const SEARCH_NO_RESULTS_KEY = "CONVERSATION_PANEL$SEARCH_NO_RESULTS";

const navigateMock = vi.fn();

const createMockConversation = (
  overrides: Partial<AppConversation> = {},
): AppConversation => ({
  id: "1",
  title: "Review the project",
  selected_repository: "org/repo",
  git_provider: null,
  selected_branch: "main",
  updated_at: "2026-01-01T00:00:00.000Z",
  created_at: "2026-01-01T00:00:00.000Z",
  execution_status: ExecutionStatus.FINISHED,
  conversation_url: null,
  created_by_user_id: "user",
  metrics: null,
  llm_model: null,
  trigger: null,
  pr_number: [],
  session_api_key: null,
  sandbox_id: null,
  sub_conversation_ids: [],
  ...overrides,
});

function renderCommandMenu(navigate = navigateMock) {
  const view = renderWithProviders(<CommandMenu />, {
    navigation: { navigate },
  });

  return { ...view, navigate };
}

beforeEach(() => {
  navigateMock.mockReset();
  window.localStorage.clear();
  useCommandMenuStore.setState({ isOpen: false });
  useSidebarStore.setState({ collapsed: false });
  vi.restoreAllMocks();
  // Keep command-only tests free of conversation hits unless a case opts in.
  vi.spyOn(
    searchMatchingConversationsModule,
    "searchMatchingConversations",
  ).mockResolvedValue([]);
});

describe("CommandMenu", () => {
  it("opens from the global command-k shortcut and closes with escape", async () => {
    renderCommandMenu();

    fireEvent.keyDown(window, { key: "k", metaKey: true });

    const searchInput = await screen.findByRole("combobox", {
      name: SEARCH_LABEL_KEY,
    });
    await waitFor(() => expect(searchInput).toHaveFocus());
    expect(screen.getByTestId("command-menu")).toBeInTheDocument();

    await userEvent.keyboard("{Escape}");

    await waitFor(() => {
      expect(screen.queryByTestId("command-menu")).not.toBeInTheDocument();
    });
  });

  it("opens from the global ctrl-k shortcut", async () => {
    renderCommandMenu();

    fireEvent.keyDown(window, { key: "k", ctrlKey: true });

    const searchInput = await screen.findByRole("combobox", {
      name: SEARCH_LABEL_KEY,
    });
    await waitFor(() => expect(searchInput).toHaveFocus());
  });

  it("filters commands by page and setting keywords", async () => {
    useCommandMenuStore.getState().open();
    renderCommandMenu();

    await userEvent.type(
      screen.getByRole("combobox", { name: SEARCH_LABEL_KEY }),
      "secrets",
    );

    expect(screen.getByText(SECRETS_TITLE_KEY)).toBeInTheDocument();
    expect(screen.queryByText(NEW_CHAT_TITLE_KEY)).not.toBeInTheDocument();
  });

  it("navigates to the selected command and closes the menu", async () => {
    useCommandMenuStore.getState().open();
    const { navigate } = renderCommandMenu();

    await userEvent.click(screen.getByText(AUTOMATIONS_TITLE_KEY));

    expect(navigate).toHaveBeenCalledWith(COMMAND_MENU_ROUTE.automations);
    await waitFor(() => {
      expect(screen.queryByTestId("command-menu")).not.toBeInTheDocument();
    });
  });

  it("supports arrow-key navigation and enter selection", async () => {
    useCommandMenuStore.getState().open();
    const { navigate } = renderCommandMenu();
    const searchInput = screen.getByRole("combobox", {
      name: SEARCH_LABEL_KEY,
    });

    await userEvent.type(searchInput, "settings");
    await userEvent.keyboard("{ArrowDown}{ArrowUp}{Enter}");

    expect(navigate).toHaveBeenCalledWith(COMMAND_MENU_ROUTE.settings);
    await waitFor(() => {
      expect(screen.queryByTestId("command-menu")).not.toBeInTheDocument();
    });
  });

  it("runs local actions from the menu", async () => {
    useCommandMenuStore.getState().open();
    renderCommandMenu();

    await userEvent.type(
      screen.getByRole("combobox", { name: SEARCH_LABEL_KEY }),
      "toggle",
    );
    await userEvent.click(screen.getByText(TOGGLE_SIDEBAR_TITLE_KEY));

    expect(useSidebarStore.getState().collapsed).toBe(true);
  });

  it("shows matching conversations and opens the selected conversation", async () => {
    const conversations = [
      createMockConversation({ id: "1", title: "Alpha task" }),
      createMockConversation({ id: "99", title: "Enable Figma design export" }),
    ];
    vi.spyOn(
      searchMatchingConversationsModule,
      "searchMatchingConversations",
    ).mockImplementation(async (query) => {
      const trimmed = query.trim().toLowerCase();
      if (!trimmed) {
        return conversations;
      }
      return conversations.filter((conversation) =>
        conversation.title?.toLowerCase().includes(trimmed),
      );
    });

    useCommandMenuStore.getState().open();
    const { navigate } = renderCommandMenu();

    await userEvent.type(
      screen.getByRole("combobox", { name: SEARCH_LABEL_KEY }),
      "figma",
    );

    const section = await screen.findByTestId(
      "command-menu-conversations-section",
      {},
      { timeout: 2000 },
    );
    expect(within(section).getByText(CONVERSATIONS_GROUP_KEY)).toBeInTheDocument();

    const result = await within(section).findByTestId(
      "command-menu-conversation-result",
    );
    expect(result).toHaveAttribute("data-conversation-id", "99");

    await userEvent.click(result);

    expect(navigate).toHaveBeenCalledWith("/conversations/99");
    await waitFor(() => {
      expect(screen.queryByTestId("command-menu")).not.toBeInTheDocument();
    });
  });

  it("shows a no-results state when conversation search finds nothing", async () => {
    vi.spyOn(
      searchMatchingConversationsModule,
      "searchMatchingConversations",
    ).mockResolvedValue([]);

    useCommandMenuStore.getState().open();
    renderCommandMenu();

    await userEvent.type(
      screen.getByRole("combobox", { name: SEARCH_LABEL_KEY }),
      "zzzz-not-found",
    );

    expect(
      await screen.findByText(SEARCH_NO_RESULTS_KEY, {}, { timeout: 2000 }),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("command-menu-conversation-result"),
    ).not.toBeInTheDocument();
  });
});
