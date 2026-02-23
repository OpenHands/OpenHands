import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, test, vi } from "vitest";
import { AccountSettingsContextMenu } from "#/components/features/context-menu/account-settings-context-menu";
import { MemoryRouter } from "react-router";
import { renderWithProviders } from "../../../test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const mockTrackAddTeamMembersButtonClick = vi.fn();
const mockUseFeatureFlagEnabled = vi.fn();

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackAddTeamMembersButtonClick: mockTrackAddTeamMembersButtonClick,
  }),
}));

vi.mock("posthog-js/react", () => ({
  useFeatureFlagEnabled: () => mockUseFeatureFlagEnabled(),
}));

describe("AccountSettingsContextMenu", () => {
  const user = userEvent.setup();
  const onClickAccountSettingsMock = vi.fn();
  const onLogoutMock = vi.fn();
  const onCloseMock = vi.fn();

  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
  });

  // Create a wrapper with MemoryRouter and renderWithProviders
  const renderWithRouter = (ui: React.ReactElement) => {
    return renderWithProviders(<MemoryRouter>{ui}</MemoryRouter>);
  };

  const renderWithSaasConfig = (ui: React.ReactElement) => {
    queryClient.setQueryData(["config"], { APP_MODE: "saas" });
    return render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>{ui}</MemoryRouter>
      </QueryClientProvider>
    );
  };

  const renderWithOssConfig = (ui: React.ReactElement) => {
    queryClient.setQueryData(["config"], { APP_MODE: "oss" });
    return render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>{ui}</MemoryRouter>
      </QueryClientProvider>
    );
  };

  afterEach(() => {
    onClickAccountSettingsMock.mockClear();
    onLogoutMock.mockClear();
    onCloseMock.mockClear();
    mockTrackAddTeamMembersButtonClick.mockClear();
    mockUseFeatureFlagEnabled.mockClear();
  });

  it("should always render the right options", () => {
    renderWithRouter(
      <AccountSettingsContextMenu
        onLogout={onLogoutMock}
        onClose={onCloseMock}
      />,
    );

    expect(
      screen.getByTestId("account-settings-context-menu"),
    ).toBeInTheDocument();
    expect(screen.getByText("SIDEBAR$DOCS")).toBeInTheDocument();
    expect(screen.getByText("ACCOUNT_SETTINGS$LOGOUT")).toBeInTheDocument();
  });

  it("should render Documentation link with correct attributes", () => {
    renderWithRouter(
      <AccountSettingsContextMenu
        onLogout={onLogoutMock}
        onClose={onCloseMock}
      />,
    );

    const documentationLink = screen.getByText("SIDEBAR$DOCS").closest("a");
    expect(documentationLink).toHaveAttribute("href", "https://docs.openhands.dev");
    expect(documentationLink).toHaveAttribute("target", "_blank");
    expect(documentationLink).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("should call onLogout when the logout option is clicked", async () => {
    renderWithRouter(
      <AccountSettingsContextMenu
        onLogout={onLogoutMock}
        onClose={onCloseMock}
      />,
    );

    const logoutOption = screen.getByText("ACCOUNT_SETTINGS$LOGOUT");
    await user.click(logoutOption);

    expect(onLogoutMock).toHaveBeenCalledOnce();
  });

  test("logout button is always enabled", async () => {
    renderWithRouter(
      <AccountSettingsContextMenu
        onLogout={onLogoutMock}
        onClose={onCloseMock}
      />,
    );

    const logoutOption = screen.getByText("ACCOUNT_SETTINGS$LOGOUT");
    await user.click(logoutOption);

    expect(onLogoutMock).toHaveBeenCalledOnce();
  });

  it("should call onClose when clicking outside of the element", async () => {
    renderWithRouter(
      <AccountSettingsContextMenu
        onLogout={onLogoutMock}
        onClose={onCloseMock}
      />,
    );

    const accountSettingsButton = screen.getByText("ACCOUNT_SETTINGS$LOGOUT");
    await user.click(accountSettingsButton);
    await user.click(document.body);

    expect(onCloseMock).toHaveBeenCalledOnce();
  });

  it("should show Add Team Members button in SaaS mode when feature flag is enabled", () => {
    mockUseFeatureFlagEnabled.mockReturnValue(true);
    renderWithSaasConfig(
      <AccountSettingsContextMenu
        onLogout={onLogoutMock}
        onClose={onCloseMock}
      />,
    );

    expect(screen.getByTestId("add-team-members-button")).toBeInTheDocument();
    expect(screen.getByText("SETTINGS$NAV_ADD_TEAM_MEMBERS")).toBeInTheDocument();
  });

  it("should not show Add Team Members button in SaaS mode when feature flag is disabled", () => {
    mockUseFeatureFlagEnabled.mockReturnValue(false);
    renderWithSaasConfig(
      <AccountSettingsContextMenu
        onLogout={onLogoutMock}
        onClose={onCloseMock}
      />,
    );

    expect(screen.queryByTestId("add-team-members-button")).not.toBeInTheDocument();
    expect(screen.queryByText("SETTINGS$NAV_ADD_TEAM_MEMBERS")).not.toBeInTheDocument();
  });

  it("should not show Add Team Members button in OSS mode even when feature flag is enabled", () => {
    mockUseFeatureFlagEnabled.mockReturnValue(true);
    renderWithOssConfig(
      <AccountSettingsContextMenu
        onLogout={onLogoutMock}
        onClose={onCloseMock}
      />,
    );

    expect(screen.queryByTestId("add-team-members-button")).not.toBeInTheDocument();
    expect(screen.queryByText("SETTINGS$NAV_ADD_TEAM_MEMBERS")).not.toBeInTheDocument();
  });

  it("should call tracking function and onClose when Add Team Members button is clicked", async () => {
    mockUseFeatureFlagEnabled.mockReturnValue(true);
    renderWithSaasConfig(
      <AccountSettingsContextMenu
        onLogout={onLogoutMock}
        onClose={onCloseMock}
      />,
    );

    const addTeamMembersButton = screen.getByTestId("add-team-members-button");
    await user.click(addTeamMembersButton);

    expect(mockTrackAddTeamMembersButtonClick).toHaveBeenCalledOnce();
    expect(onCloseMock).toHaveBeenCalledOnce();
  });
});
