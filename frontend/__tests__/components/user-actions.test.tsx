import { render, screen, waitFor, act } from "@testing-library/react";
import { describe, expect, it, vi, afterEach, beforeEach, test } from "vitest";
import userEvent from "@testing-library/user-event";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { ReactElement } from "react";
import { UserActions } from "#/components/features/sidebar/user-actions";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { MOCK_PERSONAL_ORG, MOCK_TEAM_ORG_ACME } from "#/mocks/org-handlers";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
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

  it("should not show context menu before clicking", () => {
    renderUserActions();
    expect(screen.queryByTestId("user-context-menu")).not.toBeInTheDocument();
  });

  it("should show context menu when avatar is clicked", async () => {
    renderUserActions();
    const userActions = screen.getByTestId("user-actions");
    await user.click(userActions);

    expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();
  });

  it("should show context menu even when user has no avatar_url", async () => {
    renderUserActions();
    const userActions = screen.getByTestId("user-actions");
    await user.click(userActions);

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
    await user.click(userActions);

    // Context menu should still appear even when loading
    expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();
  });

  test("context menu should default to user role", async () => {
    renderUserActions();
    const userActions = screen.getByTestId("user-actions");
    await user.click(userActions);

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
    await user.click(userActions);

    // Team and Organization nav links should NOT be visible when no org is selected (personal workspace)
    expect(screen.queryByText("Team")).not.toBeInTheDocument();
    expect(screen.queryByText("Organization")).not.toBeInTheDocument();
  });

  it("should toggle context menu on repeated clicks", async () => {
    renderUserActions();
    const userActions = screen.getByTestId("user-actions");

    // First click — opens
    await user.click(userActions);
    expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();

    // Second click — closes
    await user.click(userActions);
    expect(screen.queryByTestId("user-context-menu")).not.toBeInTheDocument();
  });

  it("should close context menu when clicking outside", async () => {
    renderUserActions();
    const userActions = screen.getByTestId("user-actions");

    // Open menu
    await user.click(userActions);
    expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();

    // Click outside
    await user.click(document.body);
    expect(screen.queryByTestId("user-context-menu")).not.toBeInTheDocument();
  });

  describe("Org selector dropdown state reset when context menu hides", () => {
    // These tests verify that the org selector dropdown resets its internal
    // state (search text, open/closed) when the context menu hides and
    // reappears (via remount triggered by menuResetCount key).

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

      // Click to show context menu
      await user.click(userActions);

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

      // Click outside to close menu (triggers remount counter increment)
      await user.click(document.body);

      // Click again to reopen
      await user.click(userActions);

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

      // Click to show context menu
      await user.click(userActions);

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

      // Click outside to close menu
      await user.click(document.body);

      // Click again to reopen
      await user.click(userActions);

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
});
