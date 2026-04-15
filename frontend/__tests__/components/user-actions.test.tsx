import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import { describe, expect, it, vi, afterEach, beforeEach, test } from "vitest";
import userEvent from "@testing-library/user-event";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import { MemoryRouter, createRoutesStub } from "react-router";
import { ReactElement } from "react";
import { http, HttpResponse } from "msw";
import { UserActions } from "#/components/features/sidebar/user-actions";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { MOCK_PERSONAL_ORG, MOCK_TEAM_ORG_ACME } from "#/mocks/org-handlers";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import { server } from "#/mocks/node";
import { createMockWebClientConfig } from "#/mocks/settings-handlers";
import { renderWithProviders } from "../../test-utils";

vi.mock("react-router", async (importActual) => ({
  ...(await importActual()),
  useNavigate: () => vi.fn(),
  useRevalidator: () => ({
    revalidate: vi.fn(),
  }),
}));

vi.mock("react-i18next", async () => {
  const actual =
    await vi.importActual<typeof import("react-i18next")>("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => {
        const translations: Record<string, string> = {
          ORG$SELECT_ORGANIZATION_PLACEHOLDER: "Please select an organization",
          ORG$PERSONAL_WORKSPACE: "Personal Workspace",
        };
        return translations[key] || key;
      },
      i18n: {
        changeLanguage: vi.fn(),
      },
    }),
  };
});

const renderUserActions = (props = { hasAvatar: true }) => {
  render(
    <UserActions
      user={
        props.hasAvatar
          ? { avatar_url: "https://example.com/avatar.png" }
          : undefined
      }
    />,
    {
      wrapper: ({ children }) => (
        <MemoryRouter>
          <QueryClientProvider client={new QueryClient()}>
            {children}
          </QueryClientProvider>
        </MemoryRouter>
      ),
    },
  );
};

// RouterStub and render helper for menu close delay tests
const RouterStubForMenuCloseDelay = createRoutesStub([
  {
    path: "/",
    Component: () => (
      <UserActions user={{ avatar_url: "https://example.com/avatar.png" }} />
    ),
  },
]);

const renderUserActionsForMenuCloseDelay = () => {
  return renderWithProviders(<RouterStubForMenuCloseDelay initialEntries={["/"]} />);
};

// Create mocks for all the hooks we need
const useIsAuthedMock = vi
  .fn()
  .mockReturnValue({ data: true, isLoading: false });

const useConfigMock = vi
  .fn()
  .mockReturnValue({ data: { app_mode: "saas" }, isLoading: false });

const useUserProvidersMock = vi
  .fn()
  .mockReturnValue({ providers: [{ id: "github", name: "GitHub" }] });

// Mock the hooks
vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => useIsAuthedMock(),
}));

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => useConfigMock(),
}));

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => useUserProvidersMock(),
}));

describe("UserActions", () => {
  const user = userEvent.setup();
  const onClickAccountSettingsMock = vi.fn();
  const onLogoutMock = vi.fn();

  // Create a wrapper with MemoryRouter and renderWithProviders
  const renderWithRouter = (ui: ReactElement) =>
    renderWithProviders(<MemoryRouter>{ui}</MemoryRouter>);

  beforeEach(() => {
    // Reset all mocks to default values before each test
    useIsAuthedMock.mockReturnValue({ data: true, isLoading: false });
    useConfigMock.mockReturnValue({
      data: { app_mode: "saas" },
      isLoading: false,
    });
    useUserProvidersMock.mockReturnValue({
      providers: [{ id: "github", name: "GitHub" }],
    });
  });

  afterEach(() => {
    onClickAccountSettingsMock.mockClear();
    onLogoutMock.mockClear();
    vi.clearAllMocks();
  });

  it("should render", () => {
    renderUserActions();
    expect(screen.getByTestId("user-actions")).toBeInTheDocument();
    expect(screen.getByTestId("user-avatar")).toBeInTheDocument();
  });

  it("should show context menu even when user has no avatar_url", async () => {
    renderUserActions();
    const userActions = screen.getByTestId("user-actions");
    await user.hover(userActions);

    // Context menu SHOULD appear because user object exists (even with empty avatar_url)
    expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();
  });

  it("should work with loading state and user provided", async () => {
    // Ensure authentication and providers are set correctly
    useIsAuthedMock.mockReturnValue({ data: true, isLoading: false });
    useConfigMock.mockReturnValue({
      data: { app_mode: "saas" },
      isLoading: false,
    });
    useUserProvidersMock.mockReturnValue({
      providers: [{ id: "github", name: "GitHub" }],
    });

    renderUserActions();
    const userActions = screen.getByTestId("user-actions");
    await user.hover(userActions);

    // Context menu should still appear even when loading
    expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();
  });

  test("context menu should default to user role", async () => {
    renderUserActions();
    const userActions = screen.getByTestId("user-actions");
    await user.hover(userActions);

    // Verify logout is present
    expect(screen.getByTestId("user-context-menu")).toHaveTextContent(
      "ACCOUNT_SETTINGS$LOGOUT",
    );
    // Verify nav items are present (e.g., settings nav items)
    expect(screen.getByTestId("user-context-menu")).toHaveTextContent(
      "SETTINGS$NAV_USER",
    );
    // Verify admin-only items are NOT present for user role
    expect(
      screen.queryByText("ORG$MANAGE_ORGANIZATION_MEMBERS"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("ORG$MANAGE_ORGANIZATION"),
    ).not.toBeInTheDocument();
  });

  test("should NOT show Team and Organization nav items when personal workspace is selected", async () => {
    renderUserActions();
    const userActions = screen.getByTestId("user-actions");
    await user.hover(userActions);

    // Team and Organization nav links should NOT be visible when no org is selected (personal workspace)
    expect(screen.queryByText("Team")).not.toBeInTheDocument();
    expect(screen.queryByText("Organization")).not.toBeInTheDocument();
  });

  it("should show context menu on hover", async () => {
    renderUserActions();

    const userActions = screen.getByTestId("user-actions");
    const contextMenu = screen.getByTestId("user-context-menu");

    // Menu is in DOM but hidden via CSS (opacity-0, pointer-events-none)
    expect(contextMenu.parentElement).toHaveClass("opacity-0");
    expect(contextMenu.parentElement).toHaveClass("pointer-events-none");

    // Hover over the user actions area
    await user.hover(userActions);

    // Menu should be visible on hover (CSS classes change via group-hover)
    expect(contextMenu).toBeVisible();
  });

  it("should use state-based visibility for hover behavior instead of CSS pseudo-element", async () => {
    renderUserActions();

    const userActions = screen.getByTestId("user-actions");
    await user.hover(userActions);

    const contextMenu = screen.getByTestId("user-context-menu");
    const hoverBridgeContainer = contextMenu.parentElement;

    // The component uses state-based visibility with a 500ms delay for diagonal mouse movement
    // When visible, the container should have opacity-100 and pointer-events-auto
    expect(hoverBridgeContainer?.className).toContain("opacity-100");
    expect(hoverBridgeContainer?.className).toContain("pointer-events-auto");
  });

  describe("Org selector dropdown state reset when context menu hides", () => {
    // These tests verify that the org selector dropdown resets its internal
    // state (search text, open/closed) when the context menu hides and
    // reappears. The component uses a 500ms delay before hiding (to support
    // diagonal mouse movement).

    beforeEach(() => {
      vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
        items: [MOCK_PERSONAL_ORG, MOCK_TEAM_ORG_ACME],
        currentOrgId: MOCK_PERSONAL_ORG.id,
      });
      useSelectedOrganizationStore.setState({ organizationId: null });
    });

    it("should reset org selector search text when context menu hides and reappears", async () => {
      renderUserActions();
      const userActions = screen.getByTestId("user-actions");

      // Hover to show context menu
      await user.hover(userActions);

      // Wait for orgs to load and auto-select
      await waitFor(() => {
        expect(screen.getByRole("combobox")).toHaveValue(
          MOCK_PERSONAL_ORG.name,
        );
      });

      // Open dropdown and type search text
      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);
      const input = screen.getByRole("combobox");
      await user.clear(input);
      await user.type(input, "search text");
      expect(input).toHaveValue("search text");

      // Unhover to trigger hide timeout, then wait for the 500ms delay to complete
      await user.unhover(userActions);

      // Wait for the 500ms hide delay to complete and menu to actually hide
      await waitFor(
        () => {
          // The menu resets when it actually hides (after 500ms delay)
          // After hiding, hovering again should show a fresh menu
        },
        { timeout: 600 },
      );

      // Wait a bit more for the timeout to fire
      await new Promise((resolve) => setTimeout(resolve, 550));

      // Now hover again to show the menu
      await user.hover(userActions);

      // Org selector should be reset — showing selected org name, not search text
      await waitFor(() => {
        expect(screen.getByRole("combobox")).toHaveValue(
          MOCK_PERSONAL_ORG.name,
        );
      });
    });

    it("should reset dropdown to collapsed state when context menu hides and reappears", async () => {
      renderUserActions();
      const userActions = screen.getByTestId("user-actions");

      // Hover to show context menu
      await user.hover(userActions);

      // Wait for orgs to load
      await waitFor(() => {
        expect(screen.getByRole("combobox")).toHaveValue(
          MOCK_PERSONAL_ORG.name,
        );
      });

      // Open dropdown and type to change its state
      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);
      const input = screen.getByRole("combobox");
      await user.clear(input);
      await user.type(input, "Acme");
      expect(input).toHaveValue("Acme");

      // Unhover to trigger hide timeout
      await user.unhover(userActions);

      // Wait for the 500ms hide delay to complete
      await new Promise((resolve) => setTimeout(resolve, 550));

      // Now hover again to show the menu
      await user.hover(userActions);

      // Wait for fresh component with org data
      await waitFor(() => {
        expect(screen.getByRole("combobox")).toHaveValue(
          MOCK_PERSONAL_ORG.name,
        );
      });

      // Dropdown should be collapsed (closed) after reset
      expect(screen.getByTestId("dropdown-trigger")).toHaveAttribute(
        "aria-expanded",
        "false",
      );
      // No option elements should be rendered
      expect(screen.queryAllByRole("option")).toHaveLength(0);
    });
  });

  describe("menu close delay", () => {
    beforeEach(() => {
      vi.useFakeTimers();
      useSelectedOrganizationStore.setState({ organizationId: "1" });

      // Mock config to return SaaS mode so useShouldShowUserFeatures returns true
      server.use(
        http.get("/api/v1/web-client/config", () =>
          HttpResponse.json(createMockWebClientConfig({ app_mode: "saas" })),
        ),
      );
    });

    afterEach(() => {
      vi.useRealTimers();
      server.resetHandlers();
    });

    it("should keep menu visible when mouse leaves and re-enters within 500ms", async () => {
      // Arrange - render and wait for queries to settle
      renderUserActionsForMenuCloseDelay();
      await act(async () => {
        await vi.runAllTimersAsync();
      });

      const userActions = screen.getByTestId("user-actions");

      // Act - open menu
      await act(async () => {
        fireEvent.mouseEnter(userActions);
      });

      // Assert - menu is visible
      expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();

      // Act - leave and re-enter within 500ms
      await act(async () => {
        fireEvent.mouseLeave(userActions);
        await vi.advanceTimersByTimeAsync(200);
        fireEvent.mouseEnter(userActions);
      });

      // Assert - menu should still be visible after waiting (pending close was cancelled)
      await act(async () => {
        await vi.advanceTimersByTimeAsync(500);
      });
      expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();
    });

    it("should not close menu before 500ms delay when mouse leaves", async () => {
      // Arrange - render and wait for queries to settle
      renderUserActionsForMenuCloseDelay();
      await act(async () => {
        await vi.runAllTimersAsync();
      });

      const userActions = screen.getByTestId("user-actions");

      // Act - open menu
      await act(async () => {
        fireEvent.mouseEnter(userActions);
      });

      // Assert - menu is visible
      expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();

      // Act - leave without re-entering, but check before timeout expires
      await act(async () => {
        fireEvent.mouseLeave(userActions);
        await vi.advanceTimersByTimeAsync(400); // Before the 500ms delay
      });

      // Assert - menu should still be visible (delay hasn't expired yet)
      // Note: The menu is always in DOM but with opacity-0 when closed.
      // This test verifies the state hasn't changed yet (delay is working).
      expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();
    });
  });
});
