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
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
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
    useSelectedOrganizationStore.setState({ organizationId: MOCK_TEAM_ORG_ACME.id });
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
    // @ts-expect-error - partial mock for testing
    getConfigSpy.mockResolvedValue({
      app_mode: "saas",
      feature_flags: {
        enable_billing: true, // Enable billing by default so billing UI is shown
        hide_llm_settings: false,
        enable_jira: false,
        enable_jira_dc: false,
        enable_linear: false,
      },
    });

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

  describe("AddCreditsModal", () => {
    const openAddCreditsModal = async () => {
      const user = userEvent.setup();
      renderManageOrg();
      await screen.findByTestId("manage-org-screen");

      await selectOrganization({ orgIndex: 0 }); // user is owner in org 1

      const addCreditsButton = await waitFor(() => screen.getByText(/add/i));
      await user.click(addCreditsButton);

      const addCreditsForm = screen.getByTestId("add-credits-form");
      expect(addCreditsForm).toBeInTheDocument();

      return { user, addCreditsForm };
    };

    describe("Button State Management", () => {
      it("should enable submit button initially when modal opens", async () => {
        await openAddCreditsModal();

        const nextButton = screen.getByRole("button", { name: /next/i });
        expect(nextButton).not.toBeDisabled();
      });

      it("should enable submit button when input contains invalid value", async () => {
        const { user } = await openAddCreditsModal();
        const amountInput = screen.getByTestId("amount-input");
        const nextButton = screen.getByRole("button", { name: /next/i });

        await user.type(amountInput, "-50");

        expect(nextButton).not.toBeDisabled();
      });

      it("should enable submit button when input contains valid value", async () => {
        const { user } = await openAddCreditsModal();
        const amountInput = screen.getByTestId("amount-input");
        const nextButton = screen.getByRole("button", { name: /next/i });

        await user.type(amountInput, "100");

        expect(nextButton).not.toBeDisabled();
      });

      it("should enable submit button after validation error is shown", async () => {
        const { user } = await openAddCreditsModal();
        const amountInput = screen.getByTestId("amount-input");
        const nextButton = screen.getByRole("button", { name: /next/i });

        await user.type(amountInput, "9");
        await user.click(nextButton);

        await waitFor(() => {
          expect(screen.getByTestId("amount-error")).toBeInTheDocument();
        });

        expect(nextButton).not.toBeDisabled();
      });
    });

    describe("Input Attributes & Placeholder", () => {
      it("should have min attribute set to 10", async () => {
        await openAddCreditsModal();

        const amountInput = screen.getByTestId("amount-input");
        expect(amountInput).toHaveAttribute("min", "10");
      });

      it("should have max attribute set to 25000", async () => {
        await openAddCreditsModal();

        const amountInput = screen.getByTestId("amount-input");
        expect(amountInput).toHaveAttribute("max", "25000");
      });

      it("should have step attribute set to 1", async () => {
        await openAddCreditsModal();

        const amountInput = screen.getByTestId("amount-input");
        expect(amountInput).toHaveAttribute("step", "1");
      });

      it("should display correct placeholder text", async () => {
        await openAddCreditsModal();

        const amountInput = screen.getByTestId("amount-input");
        expect(amountInput).toHaveAttribute(
          "placeholder",
          "PAYMENT$SPECIFY_AMOUNT_USD",
        );
      });
    });

    describe("Error Message Display", () => {
      it("should not display error message initially when modal opens", async () => {
        await openAddCreditsModal();

        const errorMessage = screen.queryByTestId("amount-error");
        expect(errorMessage).not.toBeInTheDocument();
      });

      it("should display error message after submitting amount above maximum", async () => {
        const { user } = await openAddCreditsModal();
        const amountInput = screen.getByTestId("amount-input");
        const nextButton = screen.getByRole("button", { name: /next/i });

        await user.type(amountInput, "25001");
        await user.click(nextButton);

        await waitFor(() => {
          const errorMessage = screen.getByTestId("amount-error");
          expect(errorMessage).toHaveTextContent(
            "PAYMENT$ERROR_MAXIMUM_AMOUNT",
          );
        });
      });

      it("should display error message after submitting decimal value", async () => {
        const { user } = await openAddCreditsModal();
        const amountInput = screen.getByTestId("amount-input");
        const nextButton = screen.getByRole("button", { name: /next/i });

        await user.type(amountInput, "50.5");
        await user.click(nextButton);

        await waitFor(() => {
          const errorMessage = screen.getByTestId("amount-error");
          expect(errorMessage).toHaveTextContent(
            "PAYMENT$ERROR_MUST_BE_WHOLE_NUMBER",
          );
        });
      });

      it("should replace error message when submitting different invalid value", async () => {
        const { user } = await openAddCreditsModal();
        const amountInput = screen.getByTestId("amount-input");
        const nextButton = screen.getByRole("button", { name: /next/i });

        await user.type(amountInput, "9");
        await user.click(nextButton);

        await waitFor(() => {
          const errorMessage = screen.getByTestId("amount-error");
          expect(errorMessage).toHaveTextContent(
            "PAYMENT$ERROR_MINIMUM_AMOUNT",
          );
        });

        await user.clear(amountInput);
        await user.type(amountInput, "25001");
        await user.click(nextButton);

        await waitFor(() => {
          const errorMessage = screen.getByTestId("amount-error");
          expect(errorMessage).toHaveTextContent(
            "PAYMENT$ERROR_MAXIMUM_AMOUNT",
          );
        });
      });
    });

    describe("Form Submission Behavior", () => {
      it("should prevent submission when amount is invalid", async () => {
        const createCheckoutSessionSpy = vi.spyOn(
          BillingService,
          "createCheckoutSession",
        );
        const { user } = await openAddCreditsModal();
        const amountInput = screen.getByTestId("amount-input");
        const nextButton = screen.getByRole("button", { name: /next/i });

        await user.type(amountInput, "9");
        await user.click(nextButton);

        expect(createCheckoutSessionSpy).not.toHaveBeenCalled();
        await waitFor(() => {
          const errorMessage = screen.getByTestId("amount-error");
          expect(errorMessage).toHaveTextContent(
            "PAYMENT$ERROR_MINIMUM_AMOUNT",
          );
        });
      });

      it("should call createCheckoutSession with correct amount when valid", async () => {
        const createCheckoutSessionSpy = vi.spyOn(
          BillingService,
          "createCheckoutSession",
        );
        const { user } = await openAddCreditsModal();
        const amountInput = screen.getByTestId("amount-input");
        const nextButton = screen.getByRole("button", { name: /next/i });

        await user.type(amountInput, "1000");
        await user.click(nextButton);

        expect(createCheckoutSessionSpy).toHaveBeenCalledWith(1000);
        const errorMessage = screen.queryByTestId("amount-error");
        expect(errorMessage).not.toBeInTheDocument();
      });

      it("should not call createCheckoutSession when validation fails", async () => {
        const createCheckoutSessionSpy = vi.spyOn(
          BillingService,
          "createCheckoutSession",
        );
        const { user } = await openAddCreditsModal();
        const amountInput = screen.getByTestId("amount-input");
        const nextButton = screen.getByRole("button", { name: /next/i });

        await user.type(amountInput, "-50");
        await user.click(nextButton);

        // Verify mutation was not called
        expect(createCheckoutSessionSpy).not.toHaveBeenCalled();
        await waitFor(() => {
          const errorMessage = screen.getByTestId("amount-error");
          expect(errorMessage).toHaveTextContent(
            "PAYMENT$ERROR_NEGATIVE_AMOUNT",
          );
        });
      });

      it("should close modal on successful submission", async () => {
        const createCheckoutSessionSpy = vi
          .spyOn(BillingService, "createCheckoutSession")
          .mockResolvedValue("https://checkout.stripe.com/test-session");

        const { user } = await openAddCreditsModal();
        const amountInput = screen.getByTestId("amount-input");
        const nextButton = screen.getByRole("button", { name: /next/i });

        await user.type(amountInput, "1000");
        await user.click(nextButton);

        expect(createCheckoutSessionSpy).toHaveBeenCalledWith(1000);

        await waitFor(() => {
          expect(
            screen.queryByTestId("add-credits-form"),
          ).not.toBeInTheDocument();
        });
      });

      it("should allow API call when validation passes and clear any previous errors", async () => {
        const createCheckoutSessionSpy = vi.spyOn(
          BillingService,
          "createCheckoutSession",
        );

        const { user } = await openAddCreditsModal();
        const amountInput = screen.getByTestId("amount-input");
        const nextButton = screen.getByRole("button", { name: /next/i });

        // First submit invalid value
        await user.type(amountInput, "9");
        await user.click(nextButton);

        await waitFor(() => {
          expect(screen.getByTestId("amount-error")).toBeInTheDocument();
        });

        // Then submit valid value
        await user.clear(amountInput);
        await user.type(amountInput, "100");
        await user.click(nextButton);

        expect(createCheckoutSessionSpy).toHaveBeenCalledWith(100);
        const errorMessage = screen.queryByTestId("amount-error");
        expect(errorMessage).not.toBeInTheDocument();
      });
    });

    describe("Edge Cases", () => {
      it("should handle zero value correctly", async () => {
        const { user } = await openAddCreditsModal();
        const amountInput = screen.getByTestId("amount-input");
        const nextButton = screen.getByRole("button", { name: /next/i });

        await user.type(amountInput, "0");
        await user.click(nextButton);

        await waitFor(() => {
          const errorMessage = screen.getByTestId("amount-error");
          expect(errorMessage).toHaveTextContent(
            "PAYMENT$ERROR_MINIMUM_AMOUNT",
          );
        });
      });

      it("should handle whitespace-only input correctly", async () => {
        const createCheckoutSessionSpy = vi.spyOn(
          BillingService,
          "createCheckoutSession",
        );
        const { user } = await openAddCreditsModal();
        const amountInput = screen.getByTestId("amount-input");
        const nextButton = screen.getByRole("button", { name: /next/i });

        // Number inputs typically don't accept spaces, but test the behavior
        await user.type(amountInput, "   ");
        await user.click(nextButton);

        // Should not call API (empty/invalid input)
        expect(createCheckoutSessionSpy).not.toHaveBeenCalled();
      });
    });
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

      // @ts-expect-error - only return the properties we need for this test
      getConfigSpy.mockResolvedValue({
        app_mode: "saas", // required to enable getMe
      });

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
      // @ts-expect-error - only return the properties we need for this test
      getConfigSpy.mockResolvedValue({
        app_mode: "saas", // required to enable getMe
      });

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
      // @ts-expect-error - partial mock for testing
      getConfigSpy.mockResolvedValue({
        app_mode: "saas",
        feature_flags: {
          enable_billing: true,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
        },
      });

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
      // @ts-expect-error - partial mock for testing
      getConfigSpy.mockResolvedValue({
        app_mode: "saas",
        feature_flags: {
          enable_billing: true,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
        },
      });

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
      // @ts-expect-error - partial mock for testing
      getConfigSpy.mockResolvedValue({
        app_mode: "saas",
        feature_flags: {
          enable_billing: true,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
        },
      });

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
      // @ts-expect-error - partial mock for testing
      getConfigSpy.mockResolvedValue({
        app_mode: "saas",
        feature_flags: {
          enable_billing: false,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
        },
      });

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
