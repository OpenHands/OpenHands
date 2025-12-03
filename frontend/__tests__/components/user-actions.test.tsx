import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, afterEach, beforeEach, test } from "vitest";
import userEvent from "@testing-library/user-event";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { ReactElement } from "react";
import { UserActions } from "#/components/features/sidebar/user-actions";
import { renderWithProviders } from "../../test-utils";

vi.mock("react-router", async (importActual) => ({
  ...(await importActual()),
  useNavigate: () => vi.fn(),
  useRevalidator: () => ({
    revalidate: vi.fn(),
  }),
}));

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
  .mockReturnValue({ data: { APP_MODE: "saas" }, isLoading: false });

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
      data: { APP_MODE: "saas" },
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

  it("should NOT show context menu when user is not authenticated and avatar is clicked", async () => {
    // Set isAuthed to false for this test
    useIsAuthedMock.mockReturnValue({ data: false, isLoading: false });
    // Keep other mocks with default values
    useConfigMock.mockReturnValue({
      data: { APP_MODE: "saas" },
      isLoading: false,
    });
    useUserProvidersMock.mockReturnValue({
      providers: [{ id: "github", name: "GitHub" }],
    });

    renderUserActions();

    const userAvatar = screen.getByTestId("user-avatar");
    await user.click(userAvatar);

    // Context menu should NOT appear because user is not authenticated
    expect(screen.queryByTestId("user-context-menu")).not.toBeInTheDocument();
  });

  it("should NOT show context menu when user is undefined and avatar is hovered", async () => {
    renderUserActions({ hasAvatar: false });
    const userActions = screen.getByTestId("user-actions");
    await user.hover(userActions);

    // Context menu should NOT appear because user is undefined
    expect(screen.queryByTestId("user-context-menu")).not.toBeInTheDocument();
  });

  it("should show context menu even when user has no avatar_url", async () => {
    renderUserActions();
    const userActions = screen.getByTestId("user-actions");
    await user.hover(userActions);

    // Context menu SHOULD appear because user object exists (even with empty avatar_url)
    expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();
  });

  it("should NOT be able to access logout when user is not authenticated", async () => {
    // Set isAuthed to false for this test
    useIsAuthedMock.mockReturnValue({ data: false, isLoading: false });
    // Keep other mocks with default values
    useConfigMock.mockReturnValue({
      data: { APP_MODE: "saas" },
      isLoading: false,
    });
    useUserProvidersMock.mockReturnValue({
      providers: [{ id: "github", name: "GitHub" }],
    });

    renderWithRouter(<UserActions />);

    const userAvatar = screen.getByTestId("user-avatar");
    await user.click(userAvatar);

    // Context menu should NOT appear because user is not authenticated
    expect(screen.queryByTestId("user-context-menu")).not.toBeInTheDocument();

    // Logout option should NOT be accessible when user is not authenticated
    expect(
      screen.queryByText("ACCOUNT_SETTINGS$LOGOUT"),
    ).not.toBeInTheDocument();
  });

  it("should handle user prop changing from undefined to defined", async () => {
    // Start with no authentication
    useIsAuthedMock.mockReturnValue({ data: false, isLoading: false });
    // Keep other mocks with default values
    useConfigMock.mockReturnValue({
      data: { APP_MODE: "saas" },
      isLoading: false,
    });
    useUserProvidersMock.mockReturnValue({
      providers: [{ id: "github", name: "GitHub" }],
    });

    const { unmount } = renderWithRouter(<UserActions />);

    // Initially no user and not authenticated - menu should not appear
    const userActions = screen.getByTestId("user-actions");
    await user.hover(userActions);
    expect(screen.queryByTestId("user-context-menu")).not.toBeInTheDocument();

    // Unmount the first component
    unmount();

    // Set authentication to true for the new render
    useIsAuthedMock.mockReturnValue({ data: true, isLoading: false });
    // Ensure config and providers are set correctly
    useConfigMock.mockReturnValue({
      data: { APP_MODE: "saas" },
      isLoading: false,
    });
    useUserProvidersMock.mockReturnValue({
      providers: [{ id: "github", name: "GitHub" }],
    });

    // Render a new component with user prop and authentication
    renderWithRouter(
      <UserActions user={{ avatar_url: "https://example.com/avatar.png" }} />,
    );

    // Component should render correctly
    expect(screen.getByTestId("user-actions")).toBeInTheDocument();
    expect(screen.getByTestId("user-avatar")).toBeInTheDocument();

    // Menu should now work with user defined and authenticated
    const userActionsEl = screen.getByTestId("user-actions");
    await user.hover(userActionsEl);

    expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();
  });

  it("should handle user prop changing from defined to undefined", async () => {
    // Start with authentication and providers
    useIsAuthedMock.mockReturnValue({ data: true, isLoading: false });
    useConfigMock.mockReturnValue({
      data: { APP_MODE: "saas" },
      isLoading: false,
    });
    useUserProvidersMock.mockReturnValue({
      providers: [{ id: "github", name: "GitHub" }],
    });

    const { rerender } = renderWithRouter(
      <UserActions user={{ avatar_url: "https://example.com/avatar.png" }} />,
    );

    // Hover to open menu
    const userActions = screen.getByTestId("user-actions");
    await user.hover(userActions);
    expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();

    // Set authentication to false for the rerender
    useIsAuthedMock.mockReturnValue({ data: false, isLoading: false });
    // Keep other mocks with default values
    useConfigMock.mockReturnValue({
      data: { APP_MODE: "saas" },
      isLoading: false,
    });
    useUserProvidersMock.mockReturnValue({
      providers: [{ id: "github", name: "GitHub" }],
    });

    // Remove user prop - menu should disappear because user is no longer authenticated
    rerender(
      <MemoryRouter>
        <UserActions />
      </MemoryRouter>,
    );

    // Context menu should NOT be visible when user becomes unauthenticated
    expect(screen.queryByTestId("user-context-menu")).not.toBeInTheDocument();

    // Logout option should not be accessible
    expect(
      screen.queryByText("ACCOUNT_SETTINGS$LOGOUT"),
    ).not.toBeInTheDocument();
  });

  it("should work with loading state and user provided", async () => {
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
    expect(screen.queryByText("ORG$MANAGE_ACCOUNT")).not.toBeInTheDocument();
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

    // Menu should not be visible initially
    expect(screen.queryByTestId("user-context-menu")).not.toBeInTheDocument();

    // Hover over the user actions area
    await user.hover(userActions);

    // Menu should appear on hover
    expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();

    // Move mouse away
    await user.unhover(userActions);

    // Menu should disappear when not hovering
    expect(screen.queryByTestId("user-context-menu")).not.toBeInTheDocument();
  });
});
