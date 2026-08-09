import {
  render,
  screen,
  fireEvent,
  within,
  waitFor,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Sidebar } from "#/components/features/sidebar/sidebar";
import { SidebarMobileNavProvider } from "#/components/features/sidebar/sidebar-mobile-nav-context";
import { SidebarMobileMenuBar } from "#/components/features/sidebar/sidebar-mobile-menu-bar";
import { useSidebarStore } from "#/stores/sidebar-store";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import translations from "#/i18n/translation.json";

// The global `useTranslation` mock in `vitest.setup.ts` returns the key
// as-is. Override it here so `t(...)` resolves keys via the source-of-truth
// `translation.json` (English values), letting the test assert real
// user-facing labels rather than raw keys.
vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...(actual as object),
    useTranslation: () => ({
      t: (key: string) => {
        const entry = (translations as Record<string, Record<string, string>>)[
          key
        ];
        return entry?.en ?? key;
      },
      i18n: { language: "en", exists: () => false },
    }),
  };
});

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({ data: { feature_flags: {} } }),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({
    data: { email_verified: true },
    error: null,
    isError: false,
    isFetching: false,
  }),
  getErrorStatus: () => undefined,
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackendContext: () => ({
    backends: [{ id: "local", name: "Local", kind: "local" }],
    active: {
      backend: { id: "local", name: "Local", kind: "local" },
      orgId: null,
    },
    setActive: vi.fn(),
  }),
}));

vi.mock("#/hooks/query/use-backends-health", () => ({
  useBackendsHealth: () => ({
    local: { isConnected: true },
  }),
}));

vi.mock("#/components/shared/buttons/styled-tooltip", () => ({
  StyledTooltip: ({ children }: { children: unknown }) => children,
}));

vi.mock("#/components/shared/buttons/openhands-logo-button", () => ({
  OpenHandsLogoButton: () => <div data-testid="logo-button" />,
}));

vi.mock("#/components/features/sidebar/user-actions", () => ({
  UserActions: () => <div data-testid="user-actions" />,
}));

vi.mock("#/components/features/conversation-panel/conversation-panel", () => ({
  ConversationPanel: () => null,
}));

vi.mock(
  "#/components/features/conversation-panel/conversation-panel-wrapper",
  () => ({
    ConversationPanelWrapper: () => null,
  }),
);

vi.mock("#/components/shared/modals/settings/settings-modal", () => ({
  SettingsModal: () => null,
}));

vi.mock("#/components/features/backends/backend-selector", () => ({
  BackendSelector: ({
    onSelectOption,
    onOpenAddBackend,
    onOpenManageBackends,
  }: {
    onSelectOption?: () => void;
    onOpenAddBackend?: () => void;
    onOpenManageBackends?: () => void;
  } = {}) => (
    <div data-testid="backend-selector">
      {/*
        Mimic a backend OPTION row in the dropdown menu — same role/tag the
        real Dropdown emits. Clicking this should not bubble up to the rail
        collapse handler.
       */}
      <ul>
        <li
          data-testid="mock-backend-option"
          role="option"
          aria-selected={false}
          onClick={() => onSelectOption?.()}
        >
          Switch backend
        </li>
      </ul>
      <button
        type="button"
        data-testid="mock-add-backend"
        onClick={() => onOpenAddBackend?.()}
      >
        Add Backend
      </button>
      <button
        type="button"
        data-testid="mock-manage-backends"
        onClick={() => onOpenManageBackends?.()}
      >
        Manage Backends
      </button>
    </div>
  ),
}));

vi.mock("#/components/features/backends/add-backend-modal", () => ({
  AddBackendModal: ({ onClose }: { onClose: () => void }) => (
    <div data-testid="add-backend-modal">
      <button
        type="button"
        data-testid="add-backend-modal-close"
        onClick={onClose}
      >
        Close
      </button>
    </div>
  ),
}));

vi.mock("#/components/features/backends/manage-backends-modal", () => ({
  ManageBackendsModal: ({ onClose }: { onClose: () => void }) => (
    <div data-testid="manage-backends-modal">
      <button
        type="button"
        data-testid="manage-backends-modal-close"
        onClick={onClose}
      >
        Close
      </button>
    </div>
  ),
}));

vi.mock(
  "#/components/features/conversation-panel/new-conversation-button",
  () => ({
    NewConversationButton: () => <div data-testid="new-conversation-button" />,
  }),
);

vi.mock("#/components/features/sidebar/sidebar-conversation-list", () => ({
  SidebarConversationList: () => (
    <div data-testid="sidebar-conversation-list" />
  ),
}));

vi.mock("#/hooks/use-settings-nav-items", () => ({
  useSettingsNavItems: () => [],
}));

function getDesktopSidebar(collapsed?: boolean): HTMLElement {
  const selector =
    collapsed === undefined
      ? "aside[data-collapsed]"
      : `aside[data-collapsed="${collapsed ? "true" : "false"}"]`;
  const sidebar = document.querySelector(selector);
  if (!(sidebar instanceof HTMLElement)) {
    throw new Error(`Desktop sidebar not found: ${selector}`);
  }
  return sidebar;
}

function renderSidebar(currentPath: string) {
  const navigate = vi.fn();
  const value: NavigationContextValue = {
    currentPath,
    conversationId: null,
    isNavigating: false,
    navigate,
  };

  const rendered = render(
    <QueryClientProvider client={new QueryClient()}>
      <NavigationProvider value={value}>
        <SidebarMobileNavProvider>
          <Sidebar />
          <SidebarMobileMenuBar />
        </SidebarMobileNavProvider>
      </NavigationProvider>
    </QueryClientProvider>,
  );

  return { ...rendered, navigate };
}

describe("Sidebar", () => {
  beforeEach(() => {
    window.localStorage.clear();
    // Zustand store is a module singleton; reset it so collapsed state from
    // a prior test doesn't bleed into this one.
    useSidebarStore.setState({ collapsed: false });
  });

  afterEach(() => {
    window.localStorage.clear();
    useSidebarStore.setState({ collapsed: false });
  });

  it("opens and closes the mobile navigation drawer from the menu button", async () => {
    renderSidebar("/conversations");

    expect(
      screen.queryByTestId("sidebar-mobile-drawer"),
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("sidebar-mobile-menu-toggle"));
    const drawer = screen.getByTestId("sidebar-mobile-drawer");
    expect(drawer).toBeInTheDocument();
    expect(
      within(drawer).getByTestId("sidebar-conversation-list"),
    ).toBeInTheDocument();

    fireEvent.click(within(drawer).getByTestId("sidebar-mobile-drawer-close"));
    await waitFor(() => {
      expect(
        screen.queryByTestId("sidebar-mobile-drawer"),
      ).not.toBeInTheDocument();
    });
  });

  it("toggles between expanded and collapsed states and persists the choice", () => {
    const { unmount } = renderSidebar("/conversations");

    const sidebar = getDesktopSidebar(false);
    expect(sidebar.dataset.collapsed).toBe("false");

    const toggle = screen.getByTestId("sidebar-collapse-toggle");
    fireEvent.click(toggle);

    expect(sidebar.dataset.collapsed).toBe("true");

    // The choice survives a remount via localStorage.
    unmount();
    renderSidebar("/conversations");
    expect(getDesktopSidebar(true).dataset.collapsed).toBe("true");
  });

  it("expands the sidebar when the toggle is clicked from the collapsed state", () => {
    // Arrange: simulate a user whose sidebar was previously collapsed.
    useSidebarStore.setState({ collapsed: true });
    renderSidebar("/conversations");

    // Act
    fireEvent.click(screen.getByTestId("sidebar-collapse-toggle"));

    // Assert: state flips back to expanded.
    expect(getDesktopSidebar(false).dataset.collapsed).toBe("false");
  });

  it("expands the sidebar when collapsed rail empty space is clicked", () => {
    useSidebarStore.setState({ collapsed: true });
    renderSidebar("/conversations");

    const sidebar = getDesktopSidebar(true);
    expect(sidebar.dataset.collapsed).toBe("true");

    fireEvent.click(sidebar);
    expect(sidebar.dataset.collapsed).toBe("false");
  });

  it("shows collapsed server/settings action icons when sidebar is collapsed", () => {
    useSidebarStore.setState({ collapsed: true });
    renderSidebar("/conversations");

    expect(screen.getByTestId("collapsed-settings-link")).toBeInTheDocument();
    expect(
      screen.getByTestId("collapsed-backend-selector-link"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("backend-status-dot")).toBeInTheDocument();
  });

  it("navigates to settings when collapsed settings icon is clicked", () => {
    useSidebarStore.setState({ collapsed: true });
    const { navigate } = renderSidebar("/conversations");

    fireEvent.click(screen.getByTestId("collapsed-settings-link"));
    expect(navigate).toHaveBeenCalledWith("/settings", { replace: false });
  });

  it("opens the backend popover when hovering the collapsed backend icon", async () => {
    useSidebarStore.setState({ collapsed: true });
    renderSidebar("/conversations");

    expect(screen.queryByTestId("backend-selector")).not.toBeInTheDocument();
    const trigger = screen.getByTestId("collapsed-backend-selector-link");
    const wrapper = trigger.parentElement;
    if (!wrapper) throw new Error("Popover wrapper not found");
    fireEvent.mouseEnter(wrapper);
    expect(await screen.findByTestId("backend-selector")).toBeInTheDocument();
  });

  it("does NOT expand the sidebar when a backend option in the popover is clicked", async () => {
    // Bug: clicking a backend <li role='option'> bubbled up to the aside's
    // rail-collapse handler (which only bails on a/button/[role=button]),
    // so selecting a backend would expand the sidebar mid-switch.
    useSidebarStore.setState({ collapsed: true });
    renderSidebar("/conversations");

    const sidebar = getDesktopSidebar(true);
    expect(sidebar.dataset.collapsed).toBe("true");

    const trigger = screen.getByTestId("collapsed-backend-selector-link");
    const popoverContainer = trigger.parentElement;
    if (!popoverContainer) throw new Error("Popover container not found");
    fireEvent.mouseEnter(popoverContainer);
    const option = await screen.findByTestId("mock-backend-option");

    fireEvent.click(option);

    // Sidebar should remain collapsed — selecting a backend should not
    // collapse-toggle the rail.
    expect(sidebar.dataset.collapsed).toBe("true");
  });

  it("keeps the Add Backend modal open after the popover closes", async () => {
    // Bug: modal state lived inside BackendSelector; once the cursor moved
    // out of the popover toward the centred modal, mouseLeave closed the
    // popover, which unmounted BackendSelector and tore the modal down with
    // it. Modal state must live above the popover to survive its unmount.
    useSidebarStore.setState({ collapsed: true });
    renderSidebar("/conversations");

    const trigger = screen.getByTestId("collapsed-backend-selector-link");
    const popoverContainer = trigger.parentElement;
    if (!popoverContainer) throw new Error("Popover container not found");
    fireEvent.mouseEnter(popoverContainer);

    fireEvent.click(await screen.findByTestId("mock-add-backend"));

    // Cursor moves toward the modal -> popover times out and closes.
    fireEvent.mouseLeave(popoverContainer);
    await new Promise((resolve) => {
      setTimeout(resolve, 200);
    });

    expect(screen.queryByTestId("backend-selector")).not.toBeInTheDocument();
    expect(await screen.findByTestId("add-backend-modal")).toBeInTheDocument();
  });

  it("keeps the Manage Backends modal open after the popover closes", async () => {
    useSidebarStore.setState({ collapsed: true });
    renderSidebar("/conversations");

    const trigger = screen.getByTestId("collapsed-backend-selector-link");
    const popoverContainer = trigger.parentElement;
    if (!popoverContainer) throw new Error("Popover container not found");
    fireEvent.mouseEnter(popoverContainer);

    fireEvent.click(await screen.findByTestId("mock-manage-backends"));
    fireEvent.mouseLeave(popoverContainer);
    await new Promise((resolve) => {
      setTimeout(resolve, 200);
    });

    expect(screen.queryByTestId("backend-selector")).not.toBeInTheDocument();
    expect(
      await screen.findByTestId("manage-backends-modal"),
    ).toBeInTheDocument();
  });

  it("does not bubble mouse events to window when the collapsed backend icon is clicked, so the downshift-driven popover is not torn down mid-hover", () => {
    // Bug: while the popover was open, left-clicking the tray icon closed
    // the dropdown menu because downshift attaches its outside-click logic
    // to window-level mousedown/mouseup. The tray icon is a sibling of the
    // Dropdown (not one of its tracked elements), so the event reached
    // downshift and was treated as "outside". The fix stops propagation on
    // the button so neither event reaches the window listeners that close
    // the menu.
    useSidebarStore.setState({ collapsed: true });
    const windowMouseDown = vi.fn();
    const windowMouseUp = vi.fn();
    window.addEventListener("mousedown", windowMouseDown);
    window.addEventListener("mouseup", windowMouseUp);

    try {
      renderSidebar("/conversations");
      const trigger = screen.getByTestId("collapsed-backend-selector-link");
      const wrapper = trigger.parentElement;
      if (!wrapper) throw new Error("Popover wrapper not found");
      fireEvent.mouseEnter(wrapper);

      fireEvent.mouseDown(trigger);
      fireEvent.mouseUp(trigger);

      expect(windowMouseDown).not.toHaveBeenCalled();
      expect(windowMouseUp).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener("mousedown", windowMouseDown);
      window.removeEventListener("mouseup", windowMouseUp);
    }
  });

  it("renders icons for every top-level nav item so they remain meaningful in the collapsed rail", () => {
    renderSidebar("/conversations");

    for (const testId of [
      "sidebar-conversations-link",
      "sidebar-automations-link",
      "sidebar-skills-link",
    ]) {
      const link = screen.getByTestId(testId);
      expect(link.querySelector("svg")).not.toBeNull();
    }
  });

  it("renders the renamed top-level nav labels", () => {
    // Arrange
    renderSidebar("/conversations");

    // Act + Assert: each top-level nav link surfaces its new user-facing label.
    expect(screen.getByTestId("sidebar-conversations-link")).toHaveTextContent(
      "New Chat",
    );
    expect(screen.getByTestId("sidebar-skills-link")).toHaveTextContent(
      "Customize",
    );
    expect(screen.getByTestId("sidebar-automations-link")).toHaveTextContent(
      "Automate",
    );
  });
});
