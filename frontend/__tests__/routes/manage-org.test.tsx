import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { createRoutesStub } from "react-router";
import { selectOrganization } from "test-utils";
import ManageOrg from "#/routes/manage-org";
import { organizationService } from "#/api/organization-service/organization-service.api";
import SettingsScreen, { clientLoader } from "#/routes/settings";
import {
  resetOrgMockData,
  MOCK_TEAM_ORG_ACME,
  INITIAL_MOCK_ORGS,
} from "#/mocks/org-handlers";
import OptionService from "#/api/option-service/option-service.api";
import BillingService from "#/api/billing-service/billing-service.api";
import { OrganizationMember } from "#/types/org";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import { createMockWebClientConfig } from "#/mocks/settings-handlers";

const mockQueryClient = vi.hoisted(() => {
  const { QueryClient } = require("@tanstack/react-query");
  return new QueryClient();
});

vi.mock("#/query-client-config", () => ({
  queryClient: mockQueryClient,
}));

function ManageOrgWithPortalRoot() {
  return (
    <div>
      <ManageOrg />
      <div data-testid="portal-root" id="portal-root" />
    </div>
  );
}

const RouteStub = createRoutesStub([
  {
    Component: () => <div data-testid="home-screen" />,
    path: "/",
  },
  {
    // @ts-expect-error - type mismatch
    loader: clientLoader,
    Component: SettingsScreen,
    path: "/settings",
    HydrateFallback: () => <div>Loading...</div>,
    children: [
      {
        Component: ManageOrgWithPortalRoot,
        path: "/settings/org",
      },
    ],
  },
]);

let queryClient: QueryClient;

const renderManageOrg = () =>
  render(<RouteStub initialEntries={["/settings/org"]} />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });

const { navigateMock } = vi.hoisted(() => ({
  navigateMock: vi.fn(),
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

vi.mock("react-router", async () => ({
  ...(await vi.importActual("react-router")),
  useNavigate: () => navigateMock,
}));

vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => ({ data: true }),
}));

describe("Manage Org Route", () => {
  const getMeSpy = vi.spyOn(organizationService, "getMe");

  // Test data constants
  const TEST_USERS: Record<"OWNER" | "ADMIN", OrganizationMember> = {
    OWNER: {
      org_id: "1",
      user_id: "1",
      email: "test@example.com",
      role: "owner",
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_api_key_for_byor: null,
      llm_base_url: "https://api.openai.com",
      status: "active",
    },
    ADMIN: {
      org_id: "1",
      user_id: "1",
      email: "test@example.com",
      role: "admin",
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_api_key_for_byor: null,
      llm_base_url: "https://api.openai.com",
      status: "active",
    },
  };

  // Helper function to set up user mock
  const setupUserMock = (userData: {
    org_id: string;
    user_id: string;
    email: string;
    role: "owner" | "admin" | "member";
    llm_api_key: string;
    max_iterations: number;
    llm_model: string;
    llm_api_key_for_byor: string | null;
    llm_base_url: string;
    status: "active" | "invited" | "inactive";
  }) => {
    getMeSpy.mockResolvedValue(userData);
  };

  beforeEach(() => {
    // Set Zustand store to a team org so clientLoader's org route protection allows access
    useSelectedOrganizationStore.setState({
      organizationId: MOCK_TEAM_ORG_ACME.id,
    });
    // Seed organizations into the module-level queryClient used by clientLoader
    mockQueryClient.setQueryData(["organizations"], {
      items: [MOCK_TEAM_ORG_ACME],
      currentOrgId: MOCK_TEAM_ORG_ACME.id,
    });

    queryClient = new QueryClient();
    // Pre-seed organizations so org selector renders immediately (avoids flaky race with API fetch)
    queryClient.setQueryData(["organizations"], {
      items: INITIAL_MOCK_ORGS,
      currentOrgId: MOCK_TEAM_ORG_ACME.id,
    });

    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    getConfigSpy.mockResolvedValue(
      createMockWebClientConfig({
        app_mode: "saas",
        feature_flags: {
          enable_billing: true, // Enable billing by default so billing UI is shown
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

    // Set default mock for user (owner role has all permissions)
    setupUserMock(TEST_USERS.OWNER);
  });

  afterEach(() => {
    vi.clearAllMocks();
    // Reset organization mock data to ensure clean state between tests
    resetOrgMockData();
    // Reset Zustand store to ensure clean state between tests
    useSelectedOrganizationStore.setState({ organizationId: null });
    // Clear module-level queryClient used by clientLoader
    mockQueryClient.clear();
    // Clear test queryClient
    queryClient?.clear();
  });

  it("should render the available credits", async () => {
    renderManageOrg();
    await screen.findByTestId("manage-org-screen");

    await selectOrganization({ orgIndex: 0 });

    await waitFor(() => {
      const credits = screen.getByTestId("available-credits");
      expect(credits).toHaveTextContent("100");
    });
  });

  it("should render account details", async () => {
    renderManageOrg();
    await screen.findByTestId("manage-org-screen");

    await selectOrganization({ orgIndex: 0 });

    await waitFor(() => {
      const orgName = screen.getByTestId("org-name");
      expect(orgName).toHaveTextContent("Personal Workspace");
    });
  });

  it("should be able to add credits", async () => {
    const createCheckoutSessionSpy = vi.spyOn(
      BillingService,
      "createCheckoutSession",
    );

    renderManageOrg();
    await screen.findByTestId("manage-org-screen");

    await selectOrganization({ orgIndex: 0 }); // user is owner in org 1

    expect(screen.queryByTestId("add-credits-form")).not.toBeInTheDocument();
    // Simulate adding credits — wait for permissions-dependent button
    const addCreditsButton = await waitFor(() => screen.getByText(/add/i));
    await userEvent.click(addCreditsButton);

    const addCreditsForm = screen.getByTestId("add-credits-form");
    expect(addCreditsForm).toBeInTheDocument();

    const amountInput = within(addCreditsForm).getByTestId("amount-input");
    const nextButton = within(addCreditsForm).getByRole("button", {
      name: /next/i,
    });

    await userEvent.type(amountInput, "1000");
    await userEvent.click(nextButton);

    // expect redirect to payment page
    expect(createCheckoutSessionSpy).toHaveBeenCalledWith(1000);

    await waitFor(() =>
      expect(screen.queryByTestId("add-credits-form")).not.toBeInTheDocument(),
    );
  });

  it("should close the modal when clicking cancel", async () => {
    const createCheckoutSessionSpy = vi.spyOn(
      BillingService,
      "createCheckoutSession",
    );
    renderManageOrg();
    await screen.findByTestId("manage-org-screen");

    await selectOrganization({ orgIndex: 0 }); // user is owner in org 1

    expect(screen.queryByTestId("add-credits-form")).not.toBeInTheDocument();
    // Simulate adding credits — wait for permissions-dependent button
    const addCreditsButton = await waitFor(() => screen.getByText(/add/i));
    await userEvent.click(addCreditsButton);

    const addCreditsForm = screen.getByTestId("add-credits-form");
    expect(addCreditsForm).toBeInTheDocument();

    const cancelButton = within(addCreditsForm).getByRole("button", {
      name: /close/i,
    });

    await userEvent.click(cancelButton);

    expect(screen.queryByTestId("add-credits-form")).not.toBeInTheDocument();
    expect(createCheckoutSessionSpy).not.toHaveBeenCalled();
  });

  it("should show add credits option for ADMIN role", async () => {
    renderManageOrg();
    await screen.findByTestId("manage-org-screen");

    await selectOrganization({ orgIndex: 3 }); // user is admin in org 4 (All Hands AI)

    // Verify credits are shown
    await waitFor(() => {
      const credits = screen.getByTestId("available-credits");
      expect(credits).toBeInTheDocument();
    });

    // Verify add credits button is present (admins can add credits)
    const addButton = screen.getByText(/add/i);
    expect(addButton).toBeInTheDocument();
  });

  describe("actions", () => {
    it("should be able to update the organization name", async () => {
      const updateOrgNameSpy = vi.spyOn(
        organizationService,
        "updateOrganization",
      );
      const getConfigSpy = vi.spyOn(OptionService, "getConfig");

      getConfigSpy.mockResolvedValue(
        createMockWebClientConfig({
          app_mode: "saas", // required to enable getMe
        }),
      );

      renderManageOrg();
      await screen.findByTestId("manage-org-screen");

      await selectOrganization({ orgIndex: 0 });

      const orgName = screen.getByTestId("org-name");
      await waitFor(() =>
        expect(orgName).toHaveTextContent("Personal Workspace"),
      );

      expect(
        screen.queryByTestId("update-org-name-form"),
      ).not.toBeInTheDocument();

      const changeOrgNameButton = within(orgName).getByRole("button", {
        name: /change/i,
      });
      await userEvent.click(changeOrgNameButton);

      const orgNameForm = screen.getByTestId("update-org-name-form");
      const orgNameInput = within(orgNameForm).getByRole("textbox");
      const saveButton = within(orgNameForm).getByRole("button", {
        name: /save/i,
      });

      await userEvent.type(orgNameInput, "New Org Name");
      await userEvent.click(saveButton);

      expect(updateOrgNameSpy).toHaveBeenCalledWith({
        orgId: "1",
        name: "New Org Name",
      });

      await waitFor(() => {
        expect(
          screen.queryByTestId("update-org-name-form"),
        ).not.toBeInTheDocument();
        expect(orgName).toHaveTextContent("New Org Name");
      });
    });

    it("should NOT allow roles other than owners to change org name", async () => {
      // Set admin role before rendering
      setupUserMock(TEST_USERS.ADMIN);

      renderManageOrg();
      await screen.findByTestId("manage-org-screen");

      await selectOrganization({ orgIndex: 3 }); // user is admin in org 4 (All Hands AI)

      const orgName = screen.getByTestId("org-name");
      const changeOrgNameButton = within(orgName).queryByRole("button", {
        name: /change/i,
      });
      expect(changeOrgNameButton).not.toBeInTheDocument();
    });

    it("should NOT allow roles other than owners to delete an organization", async () => {
      setupUserMock(TEST_USERS.ADMIN);

      const getConfigSpy = vi.spyOn(OptionService, "getConfig");
      getConfigSpy.mockResolvedValue(
        createMockWebClientConfig({
          app_mode: "saas", // required to enable getMe
        }),
      );

      renderManageOrg();
      await screen.findByTestId("manage-org-screen");

      await selectOrganization({ orgIndex: 3 }); // user is admin in org 4 (All Hands AI)

      const deleteOrgButton = screen.queryByRole("button", {
        name: /ORG\$DELETE_ORGANIZATION/i,
      });
      expect(deleteOrgButton).not.toBeInTheDocument();
    });

    it("should be able to delete an organization", async () => {
      const deleteOrgSpy = vi.spyOn(organizationService, "deleteOrganization");

      renderManageOrg();
      await screen.findByTestId("manage-org-screen");

      await selectOrganization({ orgIndex: 0 });

      expect(
        screen.queryByTestId("delete-org-confirmation"),
      ).not.toBeInTheDocument();

      const deleteOrgButton = await waitFor(() =>
        screen.getByRole("button", {
          name: /ORG\$DELETE_ORGANIZATION/i,
        }),
      );
      await userEvent.click(deleteOrgButton);

      const deleteConfirmation = screen.getByTestId("delete-org-confirmation");
      const confirmButton = within(deleteConfirmation).getByRole("button", {
        name: /BUTTON\$CONFIRM/i,
      });

      await userEvent.click(confirmButton);

      expect(deleteOrgSpy).toHaveBeenCalledWith({ orgId: "1" });
      expect(
        screen.queryByTestId("delete-org-confirmation"),
      ).not.toBeInTheDocument();

      // expect to have navigated to home screen
      await screen.findByTestId("home-screen");
    });

    it.todo("should be able to update the organization billing info");
  });

  describe("Role-based delete organization permission behavior", () => {
    it("should show delete organization button when user has canDeleteOrganization permission (Owner role)", async () => {
      setupUserMock(TEST_USERS.OWNER);

      renderManageOrg();
      await screen.findByTestId("manage-org-screen");

      await selectOrganization({ orgIndex: 0 });

      const deleteButton = await screen.findByRole("button", {
        name: /ORG\$DELETE_ORGANIZATION/i,
      });

      expect(deleteButton).toBeInTheDocument();
      expect(deleteButton).not.toBeDisabled();
    });

    it("should not show delete organization button when user lacks canDeleteOrganization permission ('Admin' role)", async () => {
      setupUserMock({
        org_id: "1",
        user_id: "1",
        email: "test@example.com",
        role: "admin",
        llm_api_key: "**********",
        max_iterations: 20,
        llm_model: "gpt-4",
        llm_api_key_for_byor: null,
        llm_base_url: "https://api.openai.com",
        status: "active",
      });

      renderManageOrg();
      await screen.findByTestId("manage-org-screen");

      await selectOrganization({ orgIndex: 0 });

      const deleteButton = screen.queryByRole("button", {
        name: /ORG\$DELETE_ORGANIZATION/i,
      });

      expect(deleteButton).not.toBeInTheDocument();
    });

    it("should not show delete organization button when user lacks canDeleteOrganization permission ('Member' role)", async () => {
      setupUserMock({
        org_id: "1",
        user_id: "1",
        email: "test@example.com",
        role: "member",
        llm_api_key: "**********",
        max_iterations: 20,
        llm_model: "gpt-4",
        llm_api_key_for_byor: null,
        llm_base_url: "https://api.openai.com",
        status: "active",
      });

      // Members lack view_billing permission, so the clientLoader redirects away from /settings/org
      renderManageOrg();

      // The manage-org screen should NOT be accessible — clientLoader redirects
      await waitFor(() => {
        expect(
          screen.queryByTestId("manage-org-screen"),
        ).not.toBeInTheDocument();
      });
    });

    it("should open delete confirmation modal when delete button is clicked (with permission)", async () => {
      setupUserMock(TEST_USERS.OWNER);

      renderManageOrg();
      await screen.findByTestId("manage-org-screen");

      await selectOrganization({ orgIndex: 0 });

      expect(
        screen.queryByTestId("delete-org-confirmation"),
      ).not.toBeInTheDocument();

      const deleteButton = await screen.findByRole("button", {
        name: /ORG\$DELETE_ORGANIZATION/i,
      });
      await userEvent.click(deleteButton);

      expect(screen.getByTestId("delete-org-confirmation")).toBeInTheDocument();
    });
  });

  describe("enable_billing feature flag", () => {
    it("should show credits section when enable_billing is true", async () => {
      // Arrange
      const getConfigSpy = vi.spyOn(OptionService, "getConfig");
      getConfigSpy.mockResolvedValue(
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

      // Act
      renderManageOrg();
      await screen.findByTestId("manage-org-screen");
      await selectOrganization({ orgIndex: 0 });

      // Assert
      await waitFor(() => {
        expect(screen.getByTestId("available-credits")).toBeInTheDocument();
      });

      getConfigSpy.mockRestore();
    });

    it("should show organization name section when enable_billing is true", async () => {
      // Arrange
      const getConfigSpy = vi.spyOn(OptionService, "getConfig");
      getConfigSpy.mockResolvedValue(
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

      // Act
      renderManageOrg();
      await screen.findByTestId("manage-org-screen");
      await selectOrganization({ orgIndex: 0 });

      // Assert
      await waitFor(() => {
        expect(screen.getByTestId("org-name")).toBeInTheDocument();
      });

      getConfigSpy.mockRestore();
    });

    it("should show Add Credits button when enable_billing is true", async () => {
      // Arrange
      const getConfigSpy = vi.spyOn(OptionService, "getConfig");
      getConfigSpy.mockResolvedValue(
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

      // Act
      renderManageOrg();
      await screen.findByTestId("manage-org-screen");
      await selectOrganization({ orgIndex: 0 });

      // Assert
      await waitFor(() => {
        const addButton = screen.getByText(/add/i);
        expect(addButton).toBeInTheDocument();
      });

      getConfigSpy.mockRestore();
    });

    it("should hide all billing-related elements when enable_billing is false", async () => {
      // Arrange
      const getConfigSpy = vi.spyOn(OptionService, "getConfig");
      getConfigSpy.mockResolvedValue(
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

      // Act
      renderManageOrg();
      await screen.findByTestId("manage-org-screen");
      await selectOrganization({ orgIndex: 0 });

      // Assert
      await waitFor(() => {
        expect(
          screen.queryByTestId("available-credits"),
        ).not.toBeInTheDocument();
        expect(screen.queryByText(/add/i)).not.toBeInTheDocument();
      });

      getConfigSpy.mockRestore();
    });
  });
});
