import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  CommandMenu,
  CommandMenuTrigger,
} from "#/components/features/command-menu";
import { COMMAND_MENU_ROUTE } from "#/components/features/command-menu/command-menu-items";
import { getInterfaceCopy } from "#/manifests/automation-interface";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import { useCommandMenuStore } from "#/stores/command-menu-store";
import { useSidebarStore } from "#/stores/sidebar-store";
import { renderWithProviders } from "../../../../test-utils";

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: (namespace?: string) => ({
    t: (key: string) =>
      namespace === "openhands" ? key : `missing-namespace:${key}`,
    i18n: {
      language: "en",
      exists: () => false,
    },
  }),
}));

const OPEN_LABEL_KEY = "COMMAND_MENU$OPEN_LABEL";
const SEARCH_LABEL_KEY = "COMMAND_MENU$SEARCH_LABEL";
const CLEAR_SEARCH_LABEL_KEY = "COMMAND_MENU$CLEAR_SEARCH_LABEL";
const NO_RESULTS_TITLE_KEY = "COMMAND_MENU$NO_RESULTS_TITLE";
const NAVIGATION_GROUP_KEY = "COMMAND_MENU$GROUP_NAVIGATION";
const SETTINGS_GROUP_KEY = "COMMAND_MENU$GROUP_SETTINGS";
const ACTIONS_GROUP_KEY = "COMMAND_MENU$GROUP_ACTIONS";
// The automation interface manifest owns the automations command's copy, so
// its title renders as the manifest literal rather than a translation key.
const AUTOMATIONS_TITLE_KEY = getInterfaceCopy().commandMenuTitle;
const NEW_CHAT_TITLE_KEY = "COMMAND_MENU$NEW_CHAT_TITLE";
const SECRETS_TITLE_KEY = "COMMAND_MENU$SECRETS_SETTINGS_TITLE";
const TOGGLE_SIDEBAR_TITLE_KEY = "COMMAND_MENU$TOGGLE_SIDEBAR_TITLE";
const SEARCH_INPUT_ID = "command-menu-search";
const RESULTS_LISTBOX_ID = "command-menu-results";
const NEW_CHAT_OPTION_ID = "command-menu-option-new-chat";

const navigateMock = vi.fn();

function renderCommandMenu(navigate = navigateMock) {
  const view = renderWithProviders(<CommandMenu />, {
    navigation: { navigate },
  });

  return { ...view, navigate };
}

function withNavigation(navigate: NavigationContextValue["navigate"]) {
  return (
    <NavigationProvider
      value={{
        currentPath: "/",
        conversationId: null,
        isNavigating: false,
        navigate,
      }}
    >
      <CommandMenu />
    </NavigationProvider>
  );
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

  it("ignores unmodified and unrelated global shortcuts", () => {
    renderCommandMenu();

    fireEvent.keyDown(window, { key: "k" });
    fireEvent.keyDown(window, { key: "x", metaKey: true });

    expect(screen.queryByTestId("command-menu")).not.toBeInTheDocument();
    expect(useCommandMenuStore.getState().isOpen).toBe(false);
  });

  it("removes the global shortcut listener when unmounted", () => {
    const { unmount } = renderCommandMenu();
    unmount();

    fireEvent.keyDown(window, { key: "k", ctrlKey: true });

    expect(useCommandMenuStore.getState().isOpen).toBe(false);
  });

  it("cancels deferred focus and safely ignores it after unmount", () => {
    let focusCallback: FrameRequestCallback | undefined;
    const requestFrame = vi
      .spyOn(globalThis, "requestAnimationFrame")
      .mockImplementation((callback) => {
        focusCallback = callback;
        return 42;
      });
    const cancelFrame = vi
      .spyOn(globalThis, "cancelAnimationFrame")
      .mockImplementation(() => undefined);
    useCommandMenuStore.getState().open();

    const { unmount } = renderCommandMenu();
    expect(requestFrame).toHaveBeenCalledOnce();
    unmount();

    expect(cancelFrame).toHaveBeenCalledWith(42);
    expect(focusCallback).toBeDefined();
    expect(() => focusCallback?.(0)).not.toThrow();

    requestFrame.mockRestore();
    cancelFrame.mockRestore();
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
    expect(screen.getByText(SETTINGS_GROUP_KEY)).toBeInTheDocument();
    expect(screen.queryByText(NAVIGATION_GROUP_KEY)).not.toBeInTheDocument();
    expect(screen.queryByText(ACTIONS_GROUP_KEY)).not.toBeInTheDocument();
  });

  it("keeps translated search fields separated", async () => {
    useCommandMenuStore.getState().open();
    renderCommandMenu();

    await userEvent.type(
      screen.getByRole("combobox", { name: SEARCH_LABEL_KEY }),
      "titlecommand",
    );

    expect(screen.getByText(NO_RESULTS_TITLE_KEY)).toBeInTheDocument();
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

  it("wraps arrow-up navigation from the first to the last command", async () => {
    useCommandMenuStore.getState().open();
    renderCommandMenu();
    const searchInput = screen.getByRole("combobox", {
      name: SEARCH_LABEL_KEY,
    });
    const options = screen.getAllByRole("option");

    await userEvent.type(searchInput, "{arrowup}");

    expect(options.at(-1)).toHaveAttribute("aria-selected", "true");
    expect(searchInput).toHaveAttribute(
      "aria-activedescendant",
      options.at(-1)?.id,
    );
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

  it("shows an empty state and ignores selection keys when nothing matches", async () => {
    const user = userEvent.setup();
    useCommandMenuStore.getState().open();
    const { navigate } = renderCommandMenu();
    const searchInput = screen.getByRole("combobox", {
      name: SEARCH_LABEL_KEY,
    });

    await user.type(searchInput, "no matching command");

    expect(screen.getByText(NO_RESULTS_TITLE_KEY)).toBeInTheDocument();
    expect(searchInput).not.toHaveAttribute("aria-activedescendant");

    await user.keyboard("{ArrowDown}{ArrowUp}{Enter}");

    expect(navigate).not.toHaveBeenCalled();
    expect(screen.getByTestId("command-menu")).toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: CLEAR_SEARCH_LABEL_KEY }),
    );

    const firstOption = screen.getAllByRole("option")[0];
    expect(firstOption).toHaveAttribute("aria-selected", "true");
    expect(searchInput).toHaveAttribute(
      "aria-activedescendant",
      NEW_CHAT_OPTION_ID,
    );
  });

  it("scrolls the newly active option into view", () => {
    useCommandMenuStore.getState().open();
    renderCommandMenu();
    const searchInput = screen.getByRole("combobox", {
      name: SEARCH_LABEL_KEY,
    });
    const options = screen.getAllByRole("option");
    const nextOption = options[1];
    const scrollIntoView = vi.fn();
    Object.defineProperty(nextOption, "scrollIntoView", {
      configurable: true,
      value: scrollIntoView,
    });

    fireEvent.keyDown(searchInput, { key: "ArrowDown" });

    expect(searchInput).toHaveAttribute("id", SEARCH_INPUT_ID);
    expect(searchInput).toHaveAttribute("aria-controls", RESULTS_LISTBOX_ID);
    expect(screen.getByRole("listbox")).toHaveAttribute(
      "id",
      RESULTS_LISTBOX_ID,
    );
    expect(options[0]).toHaveAttribute("id", NEW_CHAT_OPTION_ID);
    expect(options[0]).toHaveAttribute("aria-selected", "false");
    expect(nextOption).toHaveAttribute("aria-selected", "true");
    expect(searchInput).toHaveAttribute("aria-activedescendant", nextOption.id);
    expect(scrollIntoView).toHaveBeenCalledWith({ block: "nearest" });
  });

  it("preserves and clamps pointer selection when filtering", async () => {
    const user = userEvent.setup();
    useCommandMenuStore.getState().open();
    renderCommandMenu();
    const searchInput = screen.getByRole("combobox", {
      name: SEARCH_LABEL_KEY,
    });
    const allOptions = screen.getAllByRole("option");

    await user.hover(allOptions[1]);
    expect(allOptions[1]).toHaveAttribute("aria-selected", "true");
    await user.hover(allOptions.at(-1) as HTMLElement);
    expect(allOptions.at(-1)).toHaveAttribute("aria-selected", "true");

    await user.type(searchInput, "settings");

    const filteredOptions = screen.getAllByRole("option");
    expect(filteredOptions.length).toBeLessThan(allOptions.length);
    expect(filteredOptions.at(-1)).toHaveAttribute("aria-selected", "true");
    expect(searchInput).toHaveAttribute(
      "aria-activedescendant",
      filteredOptions.at(-1)?.id,
    );
  });

  it("clears the search and restores the full command list", async () => {
    const user = userEvent.setup();
    useCommandMenuStore.getState().open();
    renderCommandMenu();
    const searchInput = screen.getByRole("combobox", {
      name: SEARCH_LABEL_KEY,
    });

    await user.type(searchInput, "secrets");
    expect(screen.queryByText(NEW_CHAT_TITLE_KEY)).not.toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: CLEAR_SEARCH_LABEL_KEY }),
    );

    expect(searchInput).toHaveValue("");
    expect(searchInput).toHaveFocus();
    expect(screen.getByText(NEW_CHAT_TITLE_KEY)).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: CLEAR_SEARCH_LABEL_KEY }),
    ).not.toBeInTheDocument();
  });

  it("resets search and selection after closing and reopening", async () => {
    const user = userEvent.setup();
    useCommandMenuStore.getState().open();
    renderCommandMenu();
    const searchInput = screen.getByRole("combobox", {
      name: SEARCH_LABEL_KEY,
    });

    await user.type(searchInput, "secrets");
    await user.keyboard("{Escape}");
    await waitFor(() =>
      expect(screen.queryByTestId("command-menu")).not.toBeInTheDocument(),
    );

    act(() => useCommandMenuStore.getState().open());

    const reopenedInput = screen.getByRole("combobox", {
      name: SEARCH_LABEL_KEY,
    });
    expect(reopenedInput).toHaveValue("");
    expect(screen.getAllByRole("option")).toHaveLength(12);
    expect(screen.getAllByRole("option")[0]).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("uses the latest navigation context when a command runs", async () => {
    const firstNavigate = vi.fn();
    const latestNavigate = vi.fn();
    useCommandMenuStore.getState().open();
    const { rerender } = renderWithProviders(withNavigation(firstNavigate));

    rerender(withNavigation(latestNavigate));
    await userEvent.click(screen.getByText(AUTOMATIONS_TITLE_KEY));

    expect(firstNavigate).not.toHaveBeenCalled();
    expect(latestNavigate).toHaveBeenCalledWith(COMMAND_MENU_ROUTE.automations);
  });

  it("renders navigation commands as links and local actions as buttons", () => {
    useCommandMenuStore.getState().open();
    renderCommandMenu();

    expect(screen.getByText(NEW_CHAT_TITLE_KEY).closest("a")).toHaveAttribute(
      "href",
      COMMAND_MENU_ROUTE.conversations,
    );
    expect(
      screen.getByText(TOGGLE_SIDEBAR_TITLE_KEY).closest("button"),
    ).toHaveAttribute("type", "button");
  });

  it("applies the command palette and active-option presentation", () => {
    useCommandMenuStore.getState().open();
    renderCommandMenu();

    const dialog = screen.getByRole("dialog");
    const surface = dialog.children[1] as HTMLElement;
    const [activeOption, inactiveOption] = screen.getAllByRole("option");
    const activeIcon = activeOption.querySelector('[aria-hidden="true"]');
    const inactiveIcon = inactiveOption.querySelector('[aria-hidden="true"]');

    expect(surface).toHaveClass(
      "max-w-2xl",
      "border-[var(--oh-border)]",
      "shadow-[0_24px_90px_rgba(0,0,0,0.52),0_0_0_1px_rgba(255,255,255,0.03)_inset]",
    );
    expect(activeOption).toHaveClass("rounded-xl", "bg-white/[0.09]");
    expect(inactiveOption).toHaveClass("rounded-xl", "text-[var(--oh-muted)]");
    expect(activeIcon).toHaveClass("size-9", "border-[var(--oh-accent)]");
    expect(inactiveIcon).toHaveClass("size-9", "border-[var(--oh-border)]");
  });

  it("leaves modified command links to native browser behavior", () => {
    useCommandMenuStore.getState().open();
    const { navigate } = renderCommandMenu();
    const commandLink = screen.getByText(NEW_CHAT_TITLE_KEY).closest("a");
    expect(commandLink).not.toBeNull();
    commandLink?.addEventListener("click", (event) => event.preventDefault(), {
      once: true,
    });

    fireEvent.click(commandLink as HTMLAnchorElement, { metaKey: true });

    expect(navigate).not.toHaveBeenCalled();
    expect(screen.getByTestId("command-menu")).toBeInTheDocument();
  });

  it("does not render the browser portal when open during server rendering", async () => {
    const serverStoreState = {
      isOpen: true,
      open: () => undefined,
      close: () => undefined,
      toggle: () => undefined,
    };
    vi.resetModules();
    vi.doMock("#/stores/command-menu-store", () => ({
      useCommandMenuStore: (
        selector: (state: typeof serverStoreState) => unknown,
      ) => selector(serverStoreState),
    }));
    const { CommandMenu: ServerCommandMenu } =
      await import("#/components/features/command-menu/command-menu");
    const originalDocument = globalThis.document;
    Object.defineProperty(globalThis, "document", {
      configurable: true,
      value: undefined,
    });

    try {
      expect(renderToStaticMarkup(<ServerCommandMenu />)).toBe("");
    } finally {
      Object.defineProperty(globalThis, "document", {
        configurable: true,
        value: originalDocument,
      });
      vi.doUnmock("#/stores/command-menu-store");
      vi.resetModules();
    }
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
