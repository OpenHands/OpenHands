import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, test, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { UserContextMenu } from "#/components/features/user/user-context-menu";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { GetComponentPropTypes } from "#/utils/get-component-prop-types";
import { INITIAL_MOCK_ORGS } from "#/mocks/org-handlers";
import AuthService from "#/api/auth-service/auth-service.api";
import { SAAS_NAV_ITEMS, OSS_NAV_ITEMS } from "#/constants/settings-nav";
import OptionService from "#/api/option-service/option-service.api";

type UserContextMenuProps = GetComponentPropTypes<typeof UserContextMenu>;

function UserContextMenuWithRootOutlet({
  type,
  onClose,
}: UserContextMenuProps) {
  return (
    <div>
      <div data-testid="portal-root" id="portal-root" />
      <UserContextMenu type={type} onClose={onClose} />
    </div>
  );
}

const renderUserContextMenu = ({ type, onClose }: UserContextMenuProps) =>
  render(<UserContextMenuWithRootOutlet type={type} onClose={onClose} />, {
    wrapper: ({ children }) => (
      <MemoryRouter>
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      </MemoryRouter>
    ),
  });

const { navigateMock } = vi.hoisted(() => ({
  navigateMock: vi.fn(),
}));

vi.mock("react-router", async (importActual) => ({
  ...(await importActual()),
  useNavigate: () => navigateMock,
  useRevalidator: () => ({
    revalidate: vi.fn(),
  }),
}));

describe("UserContextMenu", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render the default context items for a user", () => {
    renderUserContextMenu({ type: "user", onClose: vi.fn });

    screen.getByTestId("org-selector");
    screen.getByText("ACCOUNT_SETTINGS$LOGOUT");

    expect(
      screen.queryByText("ORG$INVITE_ORGANIZATION_MEMBER"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("ORG$MANAGE_ORGANIZATION_MEMBERS"),
    ).not.toBeInTheDocument();
    expect(screen.queryByText("ORG$MANAGE_ACCOUNT")).not.toBeInTheDocument();
  });

  it("should render navigation items from SAAS_NAV_ITEMS (except organization-members/org)", () => {
    renderUserContextMenu({ type: "user", onClose: vi.fn });

    // Verify that navigation items are rendered (except organization-members/org which are filtered out)
    SAAS_NAV_ITEMS.filter(
      (item) =>
        item.to !== "/settings/organization-members" &&
        item.to !== "/settings/org",
    ).forEach((item) => {
      expect(screen.getByText(item.text)).toBeInTheDocument();
    });
  });

  it("should not display Organization Members menu item for regular users (filtered out)", () => {
    renderUserContextMenu({ type: "user", onClose: vi.fn });

    // Organization Members is filtered out from nav items for all users
    expect(screen.queryByText("Organization Members")).not.toBeInTheDocument();
  });

  it("should render a documentation link", () => {
    renderUserContextMenu({ type: "user", onClose: vi.fn });

    const docsLink = screen.getByText("SIDEBAR$DOCS").closest("a");
    expect(docsLink).toHaveAttribute("href", "https://docs.openhands.dev");
    expect(docsLink).toHaveAttribute("target", "_blank");
  });

  describe("OSS mode", () => {
    beforeEach(() => {
      vi.spyOn(OptionService, "getConfig").mockResolvedValue({
        APP_MODE: "oss",
        GITHUB_CLIENT_ID: "test",
        POSTHOG_CLIENT_KEY: "test",
        FEATURE_FLAGS: {
          ENABLE_BILLING: false,
          HIDE_LLM_SETTINGS: false,
          ENABLE_JIRA: false,
          ENABLE_JIRA_DC: false,
          ENABLE_LINEAR: false,
        },
      });
    });

    it("should render OSS_NAV_ITEMS when in OSS mode", async () => {
      renderUserContextMenu({ type: "user", onClose: vi.fn });

      // Wait for the config to load and OSS nav items to appear
      await waitFor(() => {
        OSS_NAV_ITEMS.forEach((item) => {
          expect(screen.getByText(item.text)).toBeInTheDocument();
        });
      });

      // Verify SAAS-only items are NOT rendered (e.g., Billing)
      expect(
        screen.queryByText("SETTINGS$NAV_BILLING"),
      ).not.toBeInTheDocument();
    });

    it("should not display Organization Members menu item in OSS mode", async () => {
      renderUserContextMenu({ type: "user", onClose: vi.fn });

      // Wait for the config to load
      await waitFor(() => {
        expect(screen.getByText("SETTINGS$NAV_LLM")).toBeInTheDocument();
      });

      // Verify Organization Members is NOT rendered in OSS mode
      expect(
        screen.queryByText("Organization Members"),
      ).not.toBeInTheDocument();
    });
  });

  describe("HIDE_LLM_SETTINGS feature flag", () => {
    it("should hide LLM settings link when HIDE_LLM_SETTINGS is true", async () => {
      vi.spyOn(OptionService, "getConfig").mockResolvedValue({
        APP_MODE: "saas",
        GITHUB_CLIENT_ID: "test",
        POSTHOG_CLIENT_KEY: "test",
        FEATURE_FLAGS: {
          ENABLE_BILLING: false,
          HIDE_LLM_SETTINGS: true,
          ENABLE_JIRA: false,
          ENABLE_JIRA_DC: false,
          ENABLE_LINEAR: false,
        },
      });

      renderUserContextMenu({ type: "user", onClose: vi.fn });

      await waitFor(() => {
        // Other nav items should still be visible
        expect(screen.getByText("SETTINGS$NAV_USER")).toBeInTheDocument();
        // LLM settings (to: "/settings") should NOT be visible
        expect(
          screen.queryByText("COMMON$LANGUAGE_MODEL_LLM"),
        ).not.toBeInTheDocument();
      });
    });

    it("should show LLM settings link when HIDE_LLM_SETTINGS is false", async () => {
      vi.spyOn(OptionService, "getConfig").mockResolvedValue({
        APP_MODE: "saas",
        GITHUB_CLIENT_ID: "test",
        POSTHOG_CLIENT_KEY: "test",
        FEATURE_FLAGS: {
          ENABLE_BILLING: false,
          HIDE_LLM_SETTINGS: false,
          ENABLE_JIRA: false,
          ENABLE_JIRA_DC: false,
          ENABLE_LINEAR: false,
        },
      });

      renderUserContextMenu({ type: "user", onClose: vi.fn });

      await waitFor(() => {
        expect(
          screen.getByText("COMMON$LANGUAGE_MODEL_LLM"),
        ).toBeInTheDocument();
      });
    });
  });

  it("should render additional context items when user is an admin", () => {
    renderUserContextMenu({ type: "admin", onClose: vi.fn });

    screen.getByTestId("org-selector");
    screen.getByText("ORG$INVITE_ORGANIZATION_MEMBER");
    screen.getByText("ORG$MANAGE_ORGANIZATION_MEMBERS");
    screen.getByText("ORG$MANAGE_ACCOUNT");
  });

  it("should render additional context items when user is an owner", () => {
    renderUserContextMenu({ type: "owner", onClose: vi.fn });

    screen.getByTestId("org-selector");
    screen.getByText("ORG$INVITE_ORGANIZATION_MEMBER");
    screen.getByText("ORG$MANAGE_ORGANIZATION_MEMBERS");
    screen.getByText("ORG$MANAGE_ACCOUNT");
  });

  it("should call the logout handler when Logout is clicked", async () => {
    const logoutSpy = vi.spyOn(AuthService, "logout");
    renderUserContextMenu({ type: "user", onClose: vi.fn });

    const logoutButton = screen.getByText("ACCOUNT_SETTINGS$LOGOUT");
    await userEvent.click(logoutButton);

    expect(logoutSpy).toHaveBeenCalledOnce();
  });

  it("should have correct navigation links for nav items", () => {
    renderUserContextMenu({ type: "user", onClose: vi.fn });

    // Test a few representative nav items have the correct href
    const userLink = screen.getByText("SETTINGS$NAV_USER").closest("a");
    expect(userLink).toHaveAttribute("href", "/settings/user");

    const billingLink = screen.getByText("SETTINGS$NAV_BILLING").closest("a");
    expect(billingLink).toHaveAttribute("href", "/settings/billing");

    const integrationsLink = screen
      .getByText("SETTINGS$NAV_INTEGRATIONS")
      .closest("a");
    expect(integrationsLink).toHaveAttribute("href", "/settings/integrations");
  });

  it("should navigate to /settings/organization-members when Manage Organization Members is clicked", async () => {
    renderUserContextMenu({ type: "admin", onClose: vi.fn });

    const manageOrganizationMembersButton = screen.getByText(
      "ORG$MANAGE_ORGANIZATION_MEMBERS",
    );
    await userEvent.click(manageOrganizationMembersButton);

    expect(navigateMock).toHaveBeenCalledExactlyOnceWith(
      "/settings/organization-members",
    );
  });

  it("should navigate to /settings/org when Manage Account is clicked", async () => {
    renderUserContextMenu({ type: "admin", onClose: vi.fn });

    const manageAccountButton = screen.getByText("ORG$MANAGE_ACCOUNT");
    await userEvent.click(manageAccountButton);

    expect(navigateMock).toHaveBeenCalledExactlyOnceWith("/settings/org");
  });

  it("should call the onClose handler when clicking outside the context menu", async () => {
    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "user", onClose: onCloseMock });

    const contextMenu = screen.getByTestId("user-context-menu");
    await userEvent.click(contextMenu);

    expect(onCloseMock).not.toHaveBeenCalled();

    // Simulate clicking outside the context menu
    await userEvent.click(document.body);

    expect(onCloseMock).toHaveBeenCalled();
  });

  it("should call the onClose handler after each action", async () => {
    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "owner", onClose: onCloseMock });

    const logoutButton = screen.getByText("ACCOUNT_SETTINGS$LOGOUT");
    await userEvent.click(logoutButton);
    expect(onCloseMock).toHaveBeenCalledTimes(1);

    const manageOrganizationMembersButton = screen.getByText(
      "ORG$MANAGE_ORGANIZATION_MEMBERS",
    );
    await userEvent.click(manageOrganizationMembersButton);
    expect(onCloseMock).toHaveBeenCalledTimes(2);

    const manageAccountButton = screen.getByText("ORG$MANAGE_ACCOUNT");
    await userEvent.click(manageAccountButton);
    expect(onCloseMock).toHaveBeenCalledTimes(3);
  });

  it("should render the invite user modal when Invite Organization Member is clicked", async () => {
    const inviteMembersBatchSpy = vi.spyOn(
      organizationService,
      "inviteMembers",
    );
    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "admin", onClose: onCloseMock });

    const inviteButton = screen.getByText("ORG$INVITE_ORGANIZATION_MEMBER");
    await userEvent.click(inviteButton);

    const portalRoot = screen.getByTestId("portal-root");
    expect(within(portalRoot).getByTestId("invite-modal")).toBeInTheDocument();

    await userEvent.click(within(portalRoot).getByText("BUTTON$CANCEL"));
    expect(inviteMembersBatchSpy).not.toHaveBeenCalled();
  });

  test("the user can change orgs", async () => {
    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "user", onClose: onCloseMock });

    const orgSelector = screen.getByTestId("org-selector");
    expect(orgSelector).toBeInTheDocument();

    // Simulate changing the organization
    await userEvent.click(orgSelector);
    const orgOption = screen.getByText(INITIAL_MOCK_ORGS[1].name);
    await userEvent.click(orgOption);

    expect(onCloseMock).not.toHaveBeenCalled();

    // Verify that the dropdown shows the selected organization
    // The dropdown should now display the selected org name
    expect(orgSelector).toHaveValue(INITIAL_MOCK_ORGS[1].name);
  });

  it("should have Personal Account as the default selected option with null value", async () => {
    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "user", onClose: onCloseMock });

    const orgSelector = screen.getByTestId("org-selector");

    // Should default to "Personal Account" when orgId is null
    expect(orgSelector).toHaveValue("Personal Account");

    // Click to open dropdown
    await userEvent.click(orgSelector);

    // Should have "Personal Account" as an option
    const personalAccountOption = screen.getByText("Personal Account");
    expect(personalAccountOption).toBeInTheDocument();

    // Select an organization
    const orgOption = screen.getByText(INITIAL_MOCK_ORGS[1].name);
    await userEvent.click(orgOption);

    // Should now show the selected organization
    expect(orgSelector).toHaveValue(INITIAL_MOCK_ORGS[1].name);

    // Click to open dropdown again
    await userEvent.click(orgSelector);

    // Click on Personal Account to go back
    const personalAccountOptionAgain = screen.getByText("Personal Account");
    await userEvent.click(personalAccountOptionAgain);

    // Should show "Personal Account" after going back
    expect(orgSelector).toHaveValue("Personal Account");
  });
});
