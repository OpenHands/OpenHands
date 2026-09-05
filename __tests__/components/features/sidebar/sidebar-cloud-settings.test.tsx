import {
  render,
  screen,
  fireEvent,
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
} from "#/contexts/navigation-context";
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

const cloudBackendMock = {
  backend: {
    id: "cloud-1",
    name: "Cloud Backend",
    kind: "cloud" as const,
    host: "https://cloud.example.com",
    apiKey: "test-key",
  },
  orgId: "org-1",
};

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
    backends: [cloudBackendMock.backend],
    active: cloudBackendMock,
    setActive: vi.fn(),
  }),
  useActiveBackend: () => cloudBackendMock,
}));

vi.mock("#/hooks/query/use-backends-health", () => ({
  useBackendsHealth: () => ({
    "cloud-1": { isConnected: true },
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

vi.mock("#/components/features/sidebar/sidebar-onboarding-checklist", () => ({
  SidebarOnboardingChecklist: () => (
    <div data-testid="sidebar-onboarding-checklist" />
  ),
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

vi.mock("#/components/features/settings/agent-canvas-version-tile", () => ({
  AgentCanvasVersionTile: ({
    hideWhenUpToDate,
  }: {
    hideWhenUpToDate?: boolean;
  } = {}) =>
    hideWhenUpToDate ? (
      <button type="button" data-testid="agent-canvas-version-tile">
        Agent Canvas version
      </button>
    ) : null,
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

vi.mock("#/components/features/sidebar/sidebar-conversation-list", () => ({
  SidebarConversationList: () => (
    <div data-testid="sidebar-conversation-list" />
  ),
}));

vi.mock("#/hooks/use-settings-nav-items", () => ({
  useSettingsNavItems: () => [],
}));

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

describe("Sidebar with Cloud Backend", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useSidebarStore.setState({ collapsed: false });
  });

  afterEach(() => {
    window.localStorage.clear();
    useSidebarStore.setState({ collapsed: false });
  });

  it("opens cloud settings in new tab when connected to cloud backend", () => {
    useSidebarStore.setState({ collapsed: true });
    renderSidebar("/conversations");

    const settingsLink = screen.getByTestId("collapsed-settings-link");
    expect(settingsLink).toBeInTheDocument();
    expect(settingsLink).toHaveAttribute("target", "_blank");
    expect(settingsLink).toHaveAttribute(
      "href",
      "https://cloud.example.com/settings",
    );
  });

  it("cloud settings link has correct rel attribute", () => {
    useSidebarStore.setState({ collapsed: true });
    renderSidebar("/conversations");

    const settingsLink = screen.getByTestId("collapsed-settings-link");
    expect(settingsLink).toHaveAttribute("rel", "noopener noreferrer");
  });
});
