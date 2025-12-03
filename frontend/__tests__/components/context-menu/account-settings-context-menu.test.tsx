import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, test, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { AccountSettingsContextMenu } from "#/components/features/context-menu/account-settings-context-menu";
import { renderWithProviders } from "../../../test-utils";

// Mock useConfig hook
const useConfigMock = vi.fn();

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => useConfigMock(),
}));

describe("AccountSettingsContextMenu", () => {
  const user = userEvent.setup();
  const onClickAccountSettingsMock = vi.fn();
  const onLogoutMock = vi.fn();
  const onCloseMock = vi.fn();

  // Create a wrapper with MemoryRouter and renderWithProviders
  const renderWithRouter = (ui: React.ReactElement) =>
    renderWithProviders(<MemoryRouter>{ui}</MemoryRouter>);

  beforeEach(() => {
    // Default to SaaS mode for tests
    useConfigMock.mockReturnValue({
      data: { APP_MODE: "saas" },
      isLoading: false,
    });
  });

  afterEach(() => {
    onClickAccountSettingsMock.mockClear();
    onLogoutMock.mockClear();
    onCloseMock.mockClear();
    vi.clearAllMocks();
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
    expect(documentationLink).toHaveAttribute(
      "href",
      "https://docs.openhands.dev",
    );
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

  describe("Team menu item", () => {
    it("should display 'Organization Members' text for the team menu item in SaaS mode", () => {
      renderWithRouter(
        <AccountSettingsContextMenu
          onLogout={onLogoutMock}
          onClose={onCloseMock}
        />,
      );

      const organizationMembersText = screen.getByText("Organization Members");
      expect(organizationMembersText).toBeInTheDocument();
    });

    it("should link to '/settings/organization-members' for the Organization Members menu item", () => {
      renderWithRouter(
        <AccountSettingsContextMenu
          onLogout={onLogoutMock}
          onClose={onCloseMock}
        />,
      );

      const organizationMembersText = screen.getByText("Organization Members");
      const teamLink = organizationMembersText.closest("a");
      expect(teamLink).toHaveAttribute(
        "href",
        "/settings/organization-members",
      );
    });

    it("should not display Organization Members menu item in OSS mode", () => {
      useConfigMock.mockReturnValue({
        data: { APP_MODE: "oss" },
        isLoading: false,
      });

      renderWithRouter(
        <AccountSettingsContextMenu
          onLogout={onLogoutMock}
          onClose={onCloseMock}
        />,
      );

      const organizationMembersText = screen.queryByText(
        "Organization Members",
      );
      expect(organizationMembersText).not.toBeInTheDocument();
    });
  });
});
