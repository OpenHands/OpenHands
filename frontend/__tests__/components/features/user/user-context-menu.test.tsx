import { render, screen, waitFor } from "@testing-library/react";
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
import { OrganizationMember } from "#/types/org";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import { createMockWebClientConfig } from "#/mocks/settings-handlers";

// Mock useBreakpoint hook
vi.mock("#/hooks/use-breakpoint", () => ({
  useBreakpoint: vi.fn(() => false), // Default to desktop (not mobile)
}));

// Mock useTracking hook for CTA
vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackSaasSelfhostedInquiry: vi.fn(),
  }),
}));

// Import the mocked modules
import * as breakpoint from "#/hooks/use-breakpoint";

type UserContextMenuProps = GetComponentPropTypes<typeof UserContextMenu>;

function UserContextMenuWithRootOutlet({
  type,
  onClose,
  onOpenInviteModal,
}: UserContextMenuProps) {
  return (
    <div>
      <div data-testid="portal-root" id="portal-root" />
      <UserContextMenu
        type={type}
        onClose={onClose}
        onOpenInviteModal={onOpenInviteModal}
      />
    </div>
  );
}

const renderUserContextMenu = ({
  type,
  onClose,
  onOpenInviteModal,
}: UserContextMenuProps) =>
  render(
    <UserContextMenuWithRootOutlet
      type={type}
      onClose={onClose}
      onOpenInviteModal={onOpenInviteModal}
    />,
    {
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

// Mock useIsAuthed to return authenticated state
vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => ({ data: true }),
}));

const createMockUser = (
  overrides: Partial<OrganizationMember> = {},
): OrganizationMember => ({
  org_id: "org-1",
  user_id: "user-1",
  email: "test@example.com",
  role: "member",
  llm_api_key: "",
  max_iterations: 100,
  llm_model: "gpt-4",
  llm_base_url: "",
  status: "active",
  ...overrides,
});

const seedActiveUser = (user: Partial<OrganizationMember>) => {
  useSelectedOrganizationStore.setState({ organizationId: "org-1" });
  vi.spyOn(organizationService, "getMe").mockResolvedValue(
    createMockUser(user),
  );
};

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

describe("UserContextMenu", () => {
  beforeEach(() => {
    // Ensure clean state at the start of each test
    vi.restoreAllMocks();
    useSelectedOrganizationStore.setState({ organizationId: null });
    // Reset breakpoint mock to desktop by default
    vi.mocked(breakpoint.useBreakpoint).mockReturnValue(false);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    navigateMock.mockClear();
    // Reset Zustand store to ensure clean state between tests
    useSelectedOrganizationStore.setState({ organizationId: null });
  });

  it("should render the default context items for a user", async () => {
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(
      createMockWebClientConfig({ app_mode: "saas" }),
    );

    renderUserContextMenu({ type: "member", onClose: vi.fn, onOpenInviteModal: vi.fn });

    screen.getByTestId("org-selector");

    // Wait for config to load so logout button appears
    await waitFor(() => {
      expect(screen.getByText("ACCOUNT_SETTINGS$LOGOUT")).toBeInTheDocument();
    });

    expect(
      screen.queryByText("ORG$INVITE_ORG_MEMBERS"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("ORG$ORGANIZATION_MEMBERS"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("COMMON$ORGANIZATION"),
    ).not.toBeInTheDocument();
  });

  it("should render navigation items from SAAS_NAV_ITEMS (except organization-members/org)", async () => {
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(
      createMockWebClientConfig({
        app_mode: "saas",
        feature_flags: {
          enable_billing: true,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
          hide_users_page: false,
          hide_billing_page: false,
          hide_integrations_page: false,
        },
      }),
    );

    renderUserContextMenu({ type: "member", onClose: vi.fn, onOpenInviteModal: vi.fn });

    // In SaaS, personal LLM/Condenser/Verification routes are hidden in favor
    // of /settings/org-defaults/* (visible only when an org is selected, which
    // this test does not seed). Org-only and billing routes are also filtered.
    const personalLlmPaths = new Set([
      "/settings",
      "/settings/condenser",
      "/settings/verification",
    ]);
    const expectedItems = SAAS_NAV_ITEMS.filter(
      (item) =>
        item.to !== "/settings/org-members" &&
        item.to !== "/settings/org" &&
        item.to !== "/settings/billing" &&
        !item.to.startsWith("/settings/org-defaults") &&
        !personalLlmPaths.has(item.to),
    );

    await waitFor(() => {
      expectedItems.forEach((item) => {
        expect(screen.getAllByText(item.text).length).toBeGreaterThan(0);
      });
    });
  });

  it("should render navigation items from SAAS_NAV_ITEMS when user role is admin (except organization-members/org)", async () => {
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(
      createMockWebClientConfig({
        app_mode: "saas",
        feature_flags: {
          enable_billing: true,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
          hide_users_page: false,
          hide_billing_page: false,
          hide_integrations_page: false,
        },
      }),
    );

    seedActiveUser({ role: "admin" });

    renderUserContextMenu({ type: "admin", onClose: vi.fn, onOpenInviteModal: vi.fn });

    // Wait for config to load and verify that navigation items are rendered (except organization-members/org which are filtered out)
    const expectedItems = SAAS_NAV_ITEMS.filter(
      (item) =>
        item.to !== "/settings/org-members" && item.to !== "/settings/org",
    );

    await waitFor(() => {
      expectedItems.forEach((item) => {
        expect(screen.getAllByText(item.text).length).toBeGreaterThan(0);
      });
    });
  });

  it("should not display Organization Members menu item for regular users (filtered out)", () => {
    renderUserContextMenu({ type: "member", onClose: vi.fn, onOpenInviteModal: vi.fn });

    // Organization Members is filtered out from nav items for all users
    expect(screen.queryByText("Organization Members")).not.toBeInTheDocument();
  });

  it("should render a documentation link", () => {
    renderUserContextMenu({ type: "member", onClose: vi.fn, onOpenInviteModal: vi.fn });

    const docsLink = screen.getByText("SIDEBAR$DOCS").closest("a");
    expect(docsLink).toHaveAttribute("href", "https://docs.openhands.dev");
    expect(docsLink).toHaveAttribute("target", "_blank");
  });

  describe("OSS mode", () => {
    beforeEach(() => {
      vi.spyOn(OptionService, "getConfig").mockResolvedValue(
        createMockWebClientConfig({
          app_mode: "oss",
          feature_flags: {
            enable_billing: false,
            hide_llm_settings: false,
            enable_jira: false,
            enable_jira_dc: false,
            enable_linear: false,
            hide_users_page: false,
            hide_billing_page: false,
            hide_integrations_page: false,
          },
        }),
      );
    });

    it("should render OSS_NAV_ITEMS when in OSS mode", async () => {
      renderUserContextMenu({ type: "member", onClose: vi.fn, onOpenInviteModal: vi.fn });

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
      renderUserContextMenu({ type: "member", onClose: vi.fn, onOpenInviteModal: vi.fn });

      // Wait for the config to load
      await waitFor(() => {
        expect(screen.getByText("SETTINGS$NAV_LLM")).toBeInTheDocument();
      });

      // Verify Organization Members is NOT rendered in OSS mode
      expect(
        screen.queryByText("Organization Members"),
      ).not.toBeInTheDocument();
    });

    it("should not display logout button in OSS mode", async () => {
      renderUserContextMenu({ type: "member", onClose: vi.fn, onOpenInviteModal: vi.fn });

      // Wait for the config to load
      await waitFor(() => {
        expect(screen.getByText("SETTINGS$NAV_LLM")).toBeInTheDocument();
      });

      // Verify logout button is NOT rendered in OSS mode
      expect(
        screen.queryByText("ACCOUNT_SETTINGS$LOGOUT"),
      ).not.toBeInTheDocument();
    });
  });

  describe("HIDE_LLM_SETTINGS feature flag", () => {
    it("should hide LLM settings link when HIDE_LLM_SETTINGS is true", async () => {
      vi.spyOn(OptionService, "getConfig").mockResolvedValue(
        createMockWebClientConfig({
          app_mode: "saas",
          feature_flags: {
            enable_billing: false,
            hide_llm_settings: true,
            enable_jira: false,
            enable_jira_dc: false,
            enable_linear: false,
            hide_users_page: false,
            hide_billing_page: false,
            hide_integrations_page: false,
          },
        }),
      );

      renderUserContextMenu({ type: "member", onClose: vi.fn, onOpenInviteModal: vi.fn });

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
      vi.spyOn(OptionService, "getConfig").mockResolvedValue(
        createMockWebClientConfig({
          app_mode: "saas",
          feature_flags: {
            enable_billing: false,
            hide_llm_settings: false,
            enable_jira: false,
            enable_jira_dc: false,
            enable_linear: false,
            hide_users_page: false,
            hide_billing_page: false,
            hide_integrations_page: false,
          },
        }),
      );
      // The LLM nav item now lives under /settings/org-defaults, which only
      // appears when an org is selected. Seed a personal org for that.
      vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
        items: [MOCK_PERSONAL_ORG],
        currentOrgId: MOCK_PERSONAL_ORG.id,
      });
      useSelectedOrganizationStore.setState({
        organizationId: MOCK_PERSONAL_ORG.id,
      });

      renderUserContextMenu({ type: "member", onClose: vi.fn, onOpenInviteModal: vi.fn });

      await waitFor(() => {
        expect(
          screen.getByText("COMMON$LANGUAGE_MODEL_LLM"),
        ).toBeInTheDocument();
      });
    });
  });

  it("should render additional context items when user is an admin", async () => {
    // Mock SaaS mode and a team org so org management items are visible
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(
      createMockWebClientConfig({ app_mode: "saas" }),
    );
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
      items: [MOCK_TEAM_ORG_ACME],
      currentOrgId: MOCK_TEAM_ORG_ACME.id,
    });
    useSelectedOrganizationStore.setState({
      organizationId: MOCK_TEAM_ORG_ACME.id,
    });
    vi.spyOn(organizationService, "getMe").mockResolvedValue(
      createMockUser({ role: "admin", org_id: MOCK_TEAM_ORG_ACME.id }),
    );

    renderUserContextMenu({ type: "admin", onClose: vi.fn, onOpenInviteModal: vi.fn });

    screen.getByTestId("org-selector");
    // Wait for orgs to load so org management items appear
    await waitFor(() => {
      expect(screen.getByText("ORG$INVITE_ORG_MEMBERS")).toBeInTheDocument();
    });
    // Note: Organization and Org Members links may or may not appear depending on
    // permission checks in useSettingsNavItems. The key test is that Invite button appears.
  });

  it("should render additional context items when user is an owner", async () => {
    // Mock SaaS mode and a team org so org management items are visible
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(
      createMockWebClientConfig({ app_mode: "saas" }),
    );
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
      items: [MOCK_TEAM_ORG_ACME],
      currentOrgId: MOCK_TEAM_ORG_ACME.id,
    });
    useSelectedOrganizationStore.setState({
      organizationId: MOCK_TEAM_ORG_ACME.id,
    });
    vi.spyOn(organizationService, "getMe").mockResolvedValue(
      createMockUser({ role: "owner", org_id: MOCK_TEAM_ORG_ACME.id }),
    );

    renderUserContextMenu({ type: "owner", onClose: vi.fn, onOpenInviteModal: vi.fn });

    screen.getByTestId("org-selector");
    // Wait for orgs to load so org management items appear
    await waitFor(() => {
      expect(screen.getByText("ORG$INVITE_ORG_MEMBERS")).toBeInTheDocument();
    });
    // Note: Organization and Org Members links may or may not appear depending on
    // permission checks in useSettingsNavItems. The key test is that Invite button appears.
  });

  it("should call the logout handler when Logout is clicked", async () => {
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(
      createMockWebClientConfig({ app_mode: "saas" }),
    );

    const logoutSpy = vi.spyOn(AuthService, "logout");
    renderUserContextMenu({ type: "member", onClose: vi.fn, onOpenInviteModal: vi.fn });

    // Wait for config to load so logout button appears
    const logoutButton = await screen.findByText("ACCOUNT_SETTINGS$LOGOUT");
    await userEvent.click(logoutButton);

    expect(logoutSpy).toHaveBeenCalledOnce();
  });

  it("should have correct navigation links for nav items", async () => {
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(
      createMockWebClientConfig({
        app_mode: "saas",
        feature_flags: {
          enable_billing: true, // Enable billing so billing link is shown
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
          hide_users_page: false,
          hide_billing_page: false,
          hide_integrations_page: false,
        },
      }),
    );

    seedActiveUser({ role: "admin" });

    renderUserContextMenu({ type: "admin", onClose: vi.fn, onOpenInviteModal: vi.fn });

    // Wait for config to load and test a few representative nav items have the correct href
    await waitFor(() => {
      const userLink = screen.getByText("SETTINGS$NAV_USER").closest("a");
      expect(userLink).toHaveAttribute("href", "/settings/user");
    });

    await waitFor(() => {
      const billingLink = screen.getByText("SETTINGS$NAV_BILLING").closest("a");
      expect(billingLink).toHaveAttribute("href", "/settings/billing");
    });

    await waitFor(() => {
      const integrationsLink = screen
        .getByText("SETTINGS$NAV_INTEGRATIONS")
        .closest("a");
      expect(integrationsLink).toHaveAttribute(
        "href",
        "/settings/integrations",
      );
    });
  });

  it("should have correct link for Organization Members nav item when visible", async () => {
    // Mock SaaS mode and a team org so org management items are visible
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(
      createMockWebClientConfig({ app_mode: "saas" }),
    );
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
      items: [MOCK_TEAM_ORG_ACME],
      currentOrgId: MOCK_TEAM_ORG_ACME.id,
    });
    useSelectedOrganizationStore.setState({
      organizationId: MOCK_TEAM_ORG_ACME.id,
    });
    vi.spyOn(organizationService, "getMe").mockResolvedValue(
      createMockUser({ role: "admin", org_id: MOCK_TEAM_ORG_ACME.id }),
    );

    renderUserContextMenu({ type: "admin", onClose: vi.fn, onOpenInviteModal: vi.fn });

    // Wait for nav items to load. The Org Members link may appear if permissions are met.
    await waitFor(() => {
      const orgMembersLink = screen.queryByText("SETTINGS$NAV_ORG_MEMBERS");
      if (orgMembersLink) {
        expect(orgMembersLink.closest("a")).toHaveAttribute(
          "href",
          "/settings/org-members",
        );
      }
    });
  });

  it("should have correct link for Organization nav item when visible", async () => {
    // Mock SaaS mode and a team org so org management items are visible
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(
      createMockWebClientConfig({ app_mode: "saas" }),
    );
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
      items: [MOCK_TEAM_ORG_ACME],
      currentOrgId: MOCK_TEAM_ORG_ACME.id,
    });
    useSelectedOrganizationStore.setState({
      organizationId: MOCK_TEAM_ORG_ACME.id,
    });
    vi.spyOn(organizationService, "getMe").mockResolvedValue(
      createMockUser({ role: "admin", org_id: MOCK_TEAM_ORG_ACME.id }),
    );

    renderUserContextMenu({ type: "admin", onClose: vi.fn, onOpenInviteModal: vi.fn });

    // Wait for nav items to load. The Organization link may appear if permissions are met.
    await waitFor(() => {
      const orgLink = screen.queryByText("SETTINGS$NAV_ORGANIZATION");
      if (orgLink) {
        expect(orgLink.closest("a")).toHaveAttribute("href", "/settings/org");
      }
    });
  });

  it("should call the onClose handler when clicking outside the context menu", async () => {
    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "member", onClose: onCloseMock, onOpenInviteModal: vi.fn });

    const contextMenu = screen.getByTestId("user-context-menu");
    await userEvent.click(contextMenu);

    expect(onCloseMock).not.toHaveBeenCalled();

    // Simulate clicking outside the context menu
    await userEvent.click(document.body);

    expect(onCloseMock).toHaveBeenCalled();
  });

  it("should call the onClose handler after each action", async () => {
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(
      createMockWebClientConfig({ app_mode: "saas" }),
    );

    // Mock a team org so org management items are visible
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
      items: [MOCK_TEAM_ORG_ACME],
      currentOrgId: MOCK_TEAM_ORG_ACME.id,
    });
    seedActiveUser({ role: "owner" });

    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "owner", onClose: onCloseMock, onOpenInviteModal: vi.fn });

    // Wait for config to load so logout button appears
    const logoutButton = await screen.findByText("ACCOUNT_SETTINGS$LOGOUT");
    await userEvent.click(logoutButton);
    expect(onCloseMock).toHaveBeenCalledTimes(1);

    // Wait for orgs to load so org management items are visible
    // Click on Organization Members link (now it's a Link, not a button)
    const orgMembersLink = await screen.findByText("SETTINGS$NAV_ORG_MEMBERS");
    await userEvent.click(orgMembersLink);
    expect(onCloseMock).toHaveBeenCalledTimes(2);

    // Click on Organization link
    const orgLink = screen.getByText("SETTINGS$NAV_ORGANIZATION");
    await userEvent.click(orgLink);
    expect(onCloseMock).toHaveBeenCalledTimes(3);
  });

  describe("Personal org vs team org visibility", () => {
    it("should not show Organization and Organization Members settings items when personal org is selected", async () => {
      vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
        items: [MOCK_PERSONAL_ORG],
        currentOrgId: MOCK_PERSONAL_ORG.id,
      });
      vi.spyOn(organizationService, "getMe").mockResolvedValue({
        org_id: "1",
        user_id: "99",
        email: "me@test.com",
        role: "admin",
        llm_api_key: "**********",
        max_iterations: 20,
        llm_model: "gpt-4",
        llm_base_url: "https://api.openai.com",
        status: "active",
      });

      // Pre-select the personal org in the Zustand store
      useSelectedOrganizationStore.setState({ organizationId: "1" });

      renderUserContextMenu({ type: "admin", onClose: vi.fn, onOpenInviteModal: vi.fn });

      // Wait for org selector to load and org management buttons to disappear
      // (they disappear when personal org is selected)
      await waitFor(() => {
        expect(
          screen.queryByText("ORG$ORGANIZATION_MEMBERS"),
        ).not.toBeInTheDocument();
      });

      expect(
        screen.queryByText("COMMON$ORGANIZATION"),
      ).not.toBeInTheDocument();
    });

    it("should not show Billing settings item when team org is selected", async () => {
      vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
        items: [MOCK_TEAM_ORG_ACME],
        currentOrgId: MOCK_TEAM_ORG_ACME.id,
      });
      vi.spyOn(organizationService, "getMe").mockResolvedValue({
        org_id: "1",
        user_id: "99",
        email: "me@test.com",
        role: "admin",
        llm_api_key: "**********",
        max_iterations: 20,
        llm_model: "gpt-4",
        llm_base_url: "https://api.openai.com",
        status: "active",
      });

      renderUserContextMenu({ type: "admin", onClose: vi.fn, onOpenInviteModal: vi.fn });

      // Wait for org selector to load and billing to disappear
      // (billing disappears when team org is selected)
      await waitFor(() => {
        expect(
          screen.queryByText("SETTINGS$NAV_BILLING"),
        ).not.toBeInTheDocument();
      });
    });
  });

  it("should call onOpenInviteModal and onClose when Invite Organization Member is clicked", async () => {
    // Mock a team org so org management items are visible (not personal org)
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
      items: [MOCK_TEAM_ORG_ACME],
      currentOrgId: MOCK_TEAM_ORG_ACME.id,
    });
    useSelectedOrganizationStore.setState({
      organizationId: MOCK_TEAM_ORG_ACME.id,
    });
    vi.spyOn(organizationService, "getMe").mockResolvedValue(
      createMockUser({ role: "admin", org_id: MOCK_TEAM_ORG_ACME.id }),
    );

    const onCloseMock = vi.fn();
    const onOpenInviteModalMock = vi.fn();
    renderUserContextMenu({
      type: "admin",
      onClose: onCloseMock,
      onOpenInviteModal: onOpenInviteModalMock,
    });

    // Wait for orgs to load so org management items are visible
    const inviteButton = await screen.findByText("ORG$INVITE_ORG_MEMBERS");
    await userEvent.click(inviteButton);

    expect(onOpenInviteModalMock).toHaveBeenCalledOnce();
    expect(onCloseMock).toHaveBeenCalledOnce();
  });

  test("the user can change orgs", async () => {
    // Mock SaaS mode and organizations for this test
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(
      createMockWebClientConfig({
        app_mode: "saas",
        feature_flags: {
          enable_billing: true,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
          hide_users_page: false,
          hide_billing_page: false,
          hide_integrations_page: false,
        },
      }),
    );
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
      items: INITIAL_MOCK_ORGS,
      currentOrgId: INITIAL_MOCK_ORGS[0].id,
    });

    const user = userEvent.setup();
    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "member", onClose: onCloseMock, onOpenInviteModal: vi.fn });

    // Wait for org selector to appear (it may take a moment for config to load)
    const orgSelector = await screen.findByTestId("org-selector");
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

  describe("Context Menu CTA", () => {
    it("should render the CTA component in SaaS Cloud mode on desktop", async () => {
      vi.spyOn(OptionService, "getConfig").mockResolvedValue(
        createMockWebClientConfig({
          app_mode: "saas",
          feature_flags: {
            deployment_mode: "cloud",
            enable_billing: false,
            hide_llm_settings: false,
            enable_jira: false,
            enable_jira_dc: false,
            enable_linear: false,
            hide_users_page: false,
            hide_billing_page: false,
            hide_integrations_page: false,
          },
        }),
      );

      renderUserContextMenu({ type: "member", onClose: vi.fn, onOpenInviteModal: vi.fn });

      await waitFor(() => {
        expect(screen.getByTestId("context-menu-cta")).toBeInTheDocument();
      });
      expect(screen.getByText("CTA$ENTERPRISE_TITLE")).toBeInTheDocument();
      expect(screen.getByText("CTA$LEARN_MORE")).toBeInTheDocument();
    });

    it("should not render the CTA component in OSS mode", async () => {
      vi.spyOn(OptionService, "getConfig").mockResolvedValue(
        createMockWebClientConfig({ app_mode: "oss" }),
      );

      renderUserContextMenu({ type: "member", onClose: vi.fn, onOpenInviteModal: vi.fn });

      await waitFor(() => {
        expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();
      });
      expect(screen.queryByTestId("context-menu-cta")).not.toBeInTheDocument();
    });

    it("should not render the CTA component on mobile even in SaaS Cloud mode", async () => {
      vi.spyOn(OptionService, "getConfig").mockResolvedValue(
        createMockWebClientConfig({
          app_mode: "saas",
          feature_flags: {
            deployment_mode: "cloud",
            enable_billing: false,
            hide_llm_settings: false,
            enable_jira: false,
            enable_jira_dc: false,
            enable_linear: false,
            hide_users_page: false,
            hide_billing_page: false,
            hide_integrations_page: false,
          },
        }),
      );
      // Set mobile mode
      vi.mocked(breakpoint.useBreakpoint).mockReturnValue(true);

      renderUserContextMenu({ type: "member", onClose: vi.fn, onOpenInviteModal: vi.fn });

      await waitFor(() => {
        expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();
      });

      expect(screen.queryByTestId("context-menu-cta")).not.toBeInTheDocument();
    });

    it("should not render the CTA component in SaaS Self-hosted mode", async () => {
      vi.spyOn(OptionService, "getConfig").mockResolvedValue(
        createMockWebClientConfig({
          app_mode: "saas",
          feature_flags: {
            deployment_mode: "self_hosted",
            enable_billing: false,
            hide_llm_settings: false,
            enable_jira: false,
            enable_jira_dc: false,
            enable_linear: false,
            hide_users_page: false,
            hide_billing_page: false,
            hide_integrations_page: false,
          },
        }),
      );

      renderUserContextMenu({ type: "member", onClose: vi.fn, onOpenInviteModal: vi.fn });

      await waitFor(() => {
        expect(screen.getByTestId("user-context-menu")).toBeInTheDocument();
      });

      expect(screen.queryByTestId("context-menu-cta")).not.toBeInTheDocument();
    });
  });
});
