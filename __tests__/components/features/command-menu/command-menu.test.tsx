import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  CommandMenu,
  CommandMenuTrigger,
} from "#/components/features/command-menu";
import { COMMAND_MENU_ROUTE } from "#/components/features/command-menu/command-menu-items";
import { useCommandMenuStore } from "#/stores/command-menu-store";
import { useSidebarStore } from "#/stores/sidebar-store";
import { renderWithProviders } from "../../../../test-utils";

const OPEN_LABEL_KEY = "COMMAND_MENU$OPEN_LABEL";
const SEARCH_LABEL_KEY = "COMMAND_MENU$SEARCH_LABEL";
const AUTOMATIONS_TITLE_KEY = "COMMAND_MENU$AUTOMATIONS_TITLE";
const NEW_CHAT_TITLE_KEY = "COMMAND_MENU$NEW_CHAT_TITLE";
const SECRETS_TITLE_KEY = "COMMAND_MENU$SECRETS_SETTINGS_TITLE";
const TOGGLE_SIDEBAR_TITLE_KEY = "COMMAND_MENU$TOGGLE_SIDEBAR_TITLE";

const navigateMock = vi.fn();

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
});

describe("CommandMenuTrigger", () => {
  it("opens the command menu from the sidebar trigger", async () => {
    renderWithProviders(
      <>
        <CommandMenuTrigger collapsed={false} />
        <CommandMenu />
      </>,
    );

    await userEvent.click(screen.getByRole("button", { name: OPEN_LABEL_KEY }));

    expect(
      await screen.findByRole("combobox", { name: SEARCH_LABEL_KEY }),
    ).toBeInTheDocument();
  });
});
