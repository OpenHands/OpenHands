import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, test, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { UserContextMenu } from "#/components/features/user/user-context-menu";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { GetComponentPropTypes } from "#/utils/get-component-prop-types";
import {
  INITIAL_MOCK_ORGS,
  MOCK_PERSONAL_ORG,
  MOCK_TEAM_ORG_ACME,
} from "#/mocks/org-handlers";
import AuthService from "#/api/auth-service/auth-service.api";
import { SAAS_NAV_ITEMS, OSS_NAV_ITEMS } from "#/constants/settings-nav";
import OptionService from "#/api/option-service/option-service.api";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";

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

vi.mock("react-i18next", async () => {
  const actual =
    await vi.importActual<typeof import("react-i18next")>("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => {
        const translations: Record<string, string> = {
          "ORG$SELECT_ORGANIZATION_PLACEHOLDER": "Please select an organization",
          "ORG$PERSONAL_WORKSPACE": "Personal Workspace",
        };
        return translations[key] || key;
      },
      i18n: {
        changeLanguage: vi.fn(),
      },
    }),
  };
});

describe("UserContextMenu", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    navigateMock.mockClear();
  });

  it("should render the default context items for a user", () => {
    renderUserContextMenu({ type: "member", onClose: vi.fn });

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

  it("should render navigation items from SAAS_NAV_ITEMS (except organization-members/org)", async () => {
    vi.spyOn(OptionService, "getConfig").mockResolvedValue({
      APP_MODE: "saas",
      GITHUB_CLIENT_ID: "test",
      POSTHOG_CLIENT_KEY: "test",
      FEATURE_FLAGS: {
        ENABLE_BILLING: false,
        HIDE_LLM_SETTINGS: false,
        HIDE_BILLING: false,
        ENABLE_JIRA: false,
        ENABLE_JIRA_DC: false,
        ENABLE_LINEAR: false,
      },
    });

    renderUserContextMenu({ type: "member", onClose: vi.fn });

    // Wait for config to load and verify that navigation items are rendered (except organization-members/org which are filtered out)
    const expectedItems = SAAS_NAV_ITEMS.filter(
      (item) =>
        item.to !== "/settings/org-members" && item.to !== "/settings/org",
    );

    await waitFor(() => {
      expectedItems.forEach((item) => {
        expect(screen.getByText(item.text)).toBeInTheDocument();
      });
    });
  });

  it("should not display Organization Members menu item for regular users (filtered out)", () => {
    renderUserContextMenu({ type: "member", onClose: vi.fn });

    // Organization Members is filtered out from nav items for all users
    expect(screen.queryByText("Organization Members")).not.toBeInTheDocument();
  });

  it("should render a documentation link", () => {
    renderUserContextMenu({ type: "member", onClose: vi.fn });

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
          HIDE_BILLING: false,
          ENABLE_JIRA: false,
          ENABLE_JIRA_DC: false,
          ENABLE_LINEAR: false,
        },
      });
    });

    it("should render OSS_NAV_ITEMS when in OSS mode", async () => {
      renderUserContextMenu({ type: "member", onClose: vi.fn });

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
      renderUserContextMenu({ type: "member", onClose: vi.fn });

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
          HIDE_BILLING: false,
          ENABLE_JIRA: false,
          ENABLE_JIRA_DC: false,
          ENABLE_LINEAR: false,
        },
      });

      renderUserContextMenu({ type: "member", onClose: vi.fn });

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
          HIDE_BILLING: false,
          ENABLE_JIRA: false,
          ENABLE_JIRA_DC: false,
          ENABLE_LINEAR: false,
        },
      });

      renderUserContextMenu({ type: "member", onClose: vi.fn });

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
    renderUserContextMenu({ type: "member", onClose: vi.fn });

    const logoutButton = screen.getByText("ACCOUNT_SETTINGS$LOGOUT");
    await userEvent.click(logoutButton);

    expect(logoutSpy).toHaveBeenCalledOnce();
  });

  it("should have correct navigation links for nav items", async () => {
    vi.spyOn(OptionService, "getConfig").mockResolvedValue({
      APP_MODE: "saas",
      GITHUB_CLIENT_ID: "test",
      POSTHOG_CLIENT_KEY: "test",
      FEATURE_FLAGS: {
        ENABLE_BILLING: false,
        HIDE_LLM_SETTINGS: false,
        HIDE_BILLING: false,
        ENABLE_JIRA: false,
        ENABLE_JIRA_DC: false,
        ENABLE_LINEAR: false,
      },
    });

    renderUserContextMenu({ type: "member", onClose: vi.fn });

    // Wait for config to load and test a few representative nav items have the correct href
    await waitFor(() => {
      const userLink = screen.getByText("SETTINGS$NAV_USER").closest("a");
      expect(userLink).toHaveAttribute("href", "/settings/user");
    });

    const billingLink = screen.getByText("SETTINGS$NAV_BILLING").closest("a");
    expect(billingLink).toHaveAttribute("href", "/settings/billing");

    const integrationsLink = screen
      .getByText("SETTINGS$NAV_INTEGRATIONS")
      .closest("a");
    expect(integrationsLink).toHaveAttribute("href", "/settings/integrations");
  });

  it("should navigate to /settings/org-members when Manage Organization Members is clicked", async () => {
    // Mock a team org so org management buttons are visible (not personal org)
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue([
      MOCK_TEAM_ORG_ACME,
    ]);

    renderUserContextMenu({ type: "admin", onClose: vi.fn });

    // Wait for orgs to load so org management buttons are visible
    const manageOrganizationMembersButton = await screen.findByText(
      "ORG$MANAGE_ORGANIZATION_MEMBERS",
    );
    await userEvent.click(manageOrganizationMembersButton);

    expect(navigateMock).toHaveBeenCalledExactlyOnceWith(
      "/settings/org-members",
    );
  });

  it("should navigate to /settings/org when Manage Account is clicked", async () => {
    // Mock a team org so org management buttons are visible (not personal org)
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue([
      MOCK_TEAM_ORG_ACME,
    ]);

    renderUserContextMenu({ type: "admin", onClose: vi.fn });

    // Wait for orgs to load so org management buttons are visible
    const manageAccountButton = await screen.findByText("ORG$MANAGE_ACCOUNT");
    await userEvent.click(manageAccountButton);

    expect(navigateMock).toHaveBeenCalledExactlyOnceWith("/settings/org");
  });

  it("should call the onClose handler when clicking outside the context menu", async () => {
    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "member", onClose: onCloseMock });

    const contextMenu = screen.getByTestId("user-context-menu");
    await userEvent.click(contextMenu);

    expect(onCloseMock).not.toHaveBeenCalled();

    // Simulate clicking outside the context menu
    await userEvent.click(document.body);

    expect(onCloseMock).toHaveBeenCalled();
  });

  it("should call the onClose handler after each action", async () => {
    // Mock a team org so org management buttons are visible
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue([
      MOCK_TEAM_ORG_ACME,
    ]);

    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "owner", onClose: onCloseMock });

    const logoutButton = screen.getByText("ACCOUNT_SETTINGS$LOGOUT");
    await userEvent.click(logoutButton);
    expect(onCloseMock).toHaveBeenCalledTimes(1);

    // Wait for orgs to load so org management buttons are visible
    const manageOrganizationMembersButton = await screen.findByText(
      "ORG$MANAGE_ORGANIZATION_MEMBERS",
    );
    await userEvent.click(manageOrganizationMembersButton);
    expect(onCloseMock).toHaveBeenCalledTimes(2);

    const manageAccountButton = screen.getByText("ORG$MANAGE_ACCOUNT");
    await userEvent.click(manageAccountButton);
    expect(onCloseMock).toHaveBeenCalledTimes(3);
  });

  describe("Personal org vs team org visibility", () => {
    it("should not show Organization and Organization Members settings items when personal org is selected", async () => {
      vi.spyOn(organizationService, "getOrganizations").mockResolvedValue([
        MOCK_PERSONAL_ORG,
      ]);
      vi.spyOn(organizationService, "getMe").mockResolvedValue({
        org_id: "1",
        user_id: "99",
        email: "me@test.com",
        role: "admin",
        llm_api_key: "**********",
        max_iterations: 20,
        llm_model: "gpt-4",
        llm_api_key_for_byor: null,
        llm_base_url: "https://api.openai.com",
        status: "active",
      });

      // Pre-select the personal org in the Zustand store
      useSelectedOrganizationStore.setState({ organizationId: "1" });

      renderUserContextMenu({ type: "admin", onClose: vi.fn });

      // Wait for org selector to load and org management buttons to disappear
      // (they disappear when personal org is selected)
      await waitFor(() => {
        expect(
          screen.queryByText("ORG$MANAGE_ORGANIZATION_MEMBERS"),
        ).not.toBeInTheDocument();
      });

      expect(screen.queryByText("ORG$MANAGE_ACCOUNT")).not.toBeInTheDocument();
    });

    it("should not show Billing settings item when team org is selected", async () => {
      vi.spyOn(organizationService, "getOrganizations").mockResolvedValue([
        MOCK_TEAM_ORG_ACME,
      ]);
      vi.spyOn(organizationService, "getMe").mockResolvedValue({
        org_id: "1",
        user_id: "99",
        email: "me@test.com",
        role: "admin",
        llm_api_key: "**********",
        max_iterations: 20,
        llm_model: "gpt-4",
        llm_api_key_for_byor: null,
        llm_base_url: "https://api.openai.com",
        status: "active",
      });

      renderUserContextMenu({ type: "admin", onClose: vi.fn });

      // Wait for org selector to load and billing to disappear
      // (billing disappears when team org is selected)
      await waitFor(() => {
        expect(
          screen.queryByText("SETTINGS$NAV_BILLING"),
        ).not.toBeInTheDocument();
      });
    });
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
    const user = userEvent.setup();
    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "member", onClose: onCloseMock });

    const orgSelector = screen.getByTestId("org-selector");
    expect(orgSelector).toBeInTheDocument();

    // Wait for organizations to load (indicated by org name appearing in the dropdown)
    // INITIAL_MOCK_ORGS[0] is a personal org, so it displays "Personal Workspace"
    await waitFor(() => {
      expect(screen.getByRole("combobox")).toHaveValue("Personal Workspace");
    });

    // Open the dropdown by clicking the trigger
    const trigger = screen.getByTestId("dropdown-trigger");
    await user.click(trigger);

    // Select a different organization
    const orgOption = screen.getByRole("option", {
      name: INITIAL_MOCK_ORGS[1].name,
    });
    await user.click(orgOption);

    expect(onCloseMock).not.toHaveBeenCalled();

    // Verify that the dropdown shows the selected organization
    expect(screen.getByRole("combobox")).toHaveValue(INITIAL_MOCK_ORGS[1].name);
  });
});
