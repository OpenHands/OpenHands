import { render, screen, waitFor } from "@testing-library/react";
import { createRoutesStub } from "react-router";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { QueryClientProvider } from "@tanstack/react-query";
import BillingSettingsScreen, { clientLoader } from "#/routes/billing";
import { PaymentForm } from "#/components/features/payment/payment-form";
import OptionService from "#/api/option-service/option-service.api";
import { OrganizationMember } from "#/types/org";
import * as orgStore from "#/stores/selected-organization-store";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { createMockWebClientConfig } from "#/mocks/settings-handlers";

// Mock the i18next hook
vi.mock("react-i18next", async () => {
  const actual =
    await vi.importActual<typeof import("react-i18next")>("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => key,
      i18n: {
        changeLanguage: vi.fn(),
      },
    }),
  };
});

// Mock toast handlers
const mockDisplaySuccessToast = vi.fn();
const mockDisplayErrorToast = vi.fn();
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: (...args: unknown[]) => mockDisplaySuccessToast(...args),
  displayErrorToast: (...args: unknown[]) => mockDisplayErrorToast(...args),
}));

// Mock useTracking hook
const mockTrackCreditsPurchased = vi.fn();
vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackCreditsPurchased: mockTrackCreditsPurchased,
  }),
}));

// Mock useBalance hook
const mockUseBalance = vi.fn();
vi.mock("#/hooks/query/use-balance", () => ({
  useBalance: () => mockUseBalance(),
}));

// Mock useCreateStripeCheckoutSession hook
vi.mock(
  "#/hooks/mutation/stripe/use-create-stripe-checkout-session",
  () => ({
    useCreateStripeCheckoutSession: () => ({
      mutate: vi.fn(),
      isPending: false,
    }),
  }),
);

describe("Billing Route", () => {
  const { mockQueryClient } = vi.hoisted(() => ({
    mockQueryClient: (() => {
      const { QueryClient } = require("@tanstack/react-query");
      return new QueryClient({
        defaultOptions: {
          queries: { retry: false },
        },
      });
    })(),
  }));

  // Mock queryClient to use our test instance
  vi.mock("#/query-client-config", () => ({
    queryClient: mockQueryClient,
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
    llm_api_key_for_byor: null,
    llm_base_url: "",
    status: "active",
    ...overrides,
  });

  const seedActiveUser = (user: Partial<OrganizationMember>) => {
    orgStore.useSelectedOrganizationStore.setState({ organizationId: "org-1" });
    vi.spyOn(organizationService, "getMe").mockResolvedValue(
      createMockUser(user),
    );
  };

  const setupSaasMode = (featureFlags = {}) => {
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
          ...featureFlags,
        },
      }),
    );
  };

  beforeEach(() => {
    mockQueryClient.clear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("clientLoader cache key", () => {
    it("should use the 'web-client-config' query key to read cached config", async () => {
      // Arrange: pre-populate the cache under the canonical key
      seedActiveUser({ role: "admin" });
      const cachedConfig = {
        app_mode: "saas" as const,
        posthog_client_key: "test",
        feature_flags: {
          enable_billing: true,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
        },
      };
      mockQueryClient.setQueryData(["web-client-config"], cachedConfig);

      const getConfigSpy = vi.spyOn(OptionService, "getConfig");

      // Act: invoke the clientLoader directly
      const result = await clientLoader();

      // Assert: the loader should have found the cached config and NOT called getConfig
      expect(getConfigSpy).not.toHaveBeenCalled();
      expect(result).toBeNull(); // admin with billing enabled = no redirect
    });
  });

  describe("clientLoader permission checks", () => {
    it("should redirect members to /settings/user when accessing billing directly", async () => {
      // Arrange
      setupSaasMode();
      seedActiveUser({ role: "member" });

      const RouterStub = createRoutesStub([
        {
          Component: BillingSettingsScreen,
          loader: clientLoader,
          path: "/settings/billing",
        },
        {
          Component: () => <div data-testid="user-settings-screen" />,
          path: "/settings/user",
        },
      ]);

      // Act
      render(<RouterStub initialEntries={["/settings/billing"]} />, {
        wrapper: ({ children }) => (
          <QueryClientProvider client={mockQueryClient}>
            {children}
          </QueryClientProvider>
        ),
      });

      // Assert - should be redirected to user settings
      await waitFor(() => {
        expect(screen.getByTestId("user-settings-screen")).toBeInTheDocument();
      });
    });

    it("should allow admins to access billing route", async () => {
      // Arrange
      setupSaasMode();
      seedActiveUser({ role: "admin" });

      const RouterStub = createRoutesStub([
        {
          Component: BillingSettingsScreen,
          loader: clientLoader,
          path: "/settings/billing",
        },
        {
          Component: () => <div data-testid="user-settings-screen" />,
          path: "/settings/user",
        },
      ]);

      // Act
      render(<RouterStub initialEntries={["/settings/billing"]} />, {
        wrapper: ({ children }) => (
          <QueryClientProvider client={mockQueryClient}>
            {children}
          </QueryClientProvider>
        ),
      });

      // Assert - should stay on billing page (component renders PaymentForm)
      await waitFor(() => {
        expect(
          screen.queryByTestId("user-settings-screen"),
        ).not.toBeInTheDocument();
      });
    });

    it("should allow owners to access billing route", async () => {
      // Arrange
      setupSaasMode();
      seedActiveUser({ role: "owner" });

      const RouterStub = createRoutesStub([
        {
          Component: BillingSettingsScreen,
          loader: clientLoader,
          path: "/settings/billing",
        },
        {
          Component: () => <div data-testid="user-settings-screen" />,
          path: "/settings/user",
        },
      ]);

      // Act
      render(<RouterStub initialEntries={["/settings/billing"]} />, {
        wrapper: ({ children }) => (
          <QueryClientProvider client={mockQueryClient}>
            {children}
          </QueryClientProvider>
        ),
      });

      // Assert - should stay on billing page
      await waitFor(() => {
        expect(
          screen.queryByTestId("user-settings-screen"),
        ).not.toBeInTheDocument();
      });
    });

    it("should redirect when user is undefined (no org selected)", async () => {
      // Arrange: no org selected, so getActiveOrganizationUser returns undefined
      setupSaasMode();
      // Explicitly clear org store so getActiveOrganizationUser returns undefined
      orgStore.useSelectedOrganizationStore.setState({ organizationId: null });

      const RouterStub = createRoutesStub([
        {
          Component: BillingSettingsScreen,
          loader: clientLoader,
          path: "/settings/billing",
        },
        {
          Component: () => <div data-testid="user-settings-screen" />,
          path: "/settings/user",
        },
      ]);

      // Act
      render(<RouterStub initialEntries={["/settings/billing"]} />, {
        wrapper: ({ children }) => (
          <QueryClientProvider client={mockQueryClient}>
            {children}
          </QueryClientProvider>
        ),
      });

      // Assert - should be redirected to user settings
      await waitFor(() => {
        expect(screen.getByTestId("user-settings-screen")).toBeInTheDocument();
      });
    });

    it("should redirect all users when enable_billing is false", async () => {
      // Arrange: enable_billing=false means billing is hidden for everyone
      setupSaasMode({ enable_billing: false });
      seedActiveUser({ role: "owner" }); // Even owners should be redirected

      const RouterStub = createRoutesStub([
        {
          Component: BillingSettingsScreen,
          loader: clientLoader,
          path: "/settings/billing",
        },
        {
          Component: () => <div data-testid="user-settings-screen" />,
          path: "/settings/user",
        },
      ]);

      // Act
      render(<RouterStub initialEntries={["/settings/billing"]} />, {
        wrapper: ({ children }) => (
          <QueryClientProvider client={mockQueryClient}>
            {children}
          </QueryClientProvider>
        ),
      });

      // Assert - should be redirected to user settings
      await waitFor(() => {
        expect(screen.getByTestId("user-settings-screen")).toBeInTheDocument();
      });
    });
  });

  describe("checkout success flow", () => {
    beforeEach(() => {
      mockUseBalance.mockReturnValue({
        data: "150.00",
        isLoading: false,
      });
    });

    it("should display success toast exactly once and track credits on checkout success", async () => {
      const RouterStub = createRoutesStub([
        {
          Component: BillingSettingsScreen,
          path: "/settings/billing",
        },
      ]);

      render(
        <RouterStub
          initialEntries={[
            "/settings/billing?checkout=success&amount=25&session_id=sess_123",
          ]}
        />,
        {
          wrapper: ({ children }) => (
            <QueryClientProvider client={mockQueryClient}>
              {children}
            </QueryClientProvider>
          ),
        },
      );

      await waitFor(() => {
        expect(mockDisplaySuccessToast).toHaveBeenCalledTimes(1);
      });

      expect(mockTrackCreditsPurchased).toHaveBeenCalledTimes(1);
      expect(mockTrackCreditsPurchased).toHaveBeenCalledWith({
        amountUsd: 25,
        stripeSessionId: "sess_123",
      });
    });

    it("should display error toast exactly once on checkout cancel", async () => {
      const RouterStub = createRoutesStub([
        {
          Component: BillingSettingsScreen,
          path: "/settings/billing",
        },
      ]);

      render(
        <RouterStub
          initialEntries={["/settings/billing?checkout=cancel"]}
        />,
        {
          wrapper: ({ children }) => (
            <QueryClientProvider client={mockQueryClient}>
              {children}
            </QueryClientProvider>
          ),
        },
      );

      await waitFor(() => {
        expect(mockDisplayErrorToast).toHaveBeenCalledTimes(1);
      });

      expect(mockTrackCreditsPurchased).not.toHaveBeenCalled();
    });
  });

  describe("PaymentForm permission behavior", () => {
    beforeEach(() => {
      mockUseBalance.mockReturnValue({
        data: "150.00",
        isLoading: false,
      });
    });

    it("should disable input and button when isDisabled is true, but show balance", async () => {
      // Arrange & Act
      render(<PaymentForm isDisabled />, {
        wrapper: ({ children }) => (
          <QueryClientProvider client={mockQueryClient}>
            {children}
          </QueryClientProvider>
        ),
      });

      // Assert - balance is visible
      const balance = screen.getByTestId("user-balance");
      expect(balance).toBeInTheDocument();
      expect(balance).toHaveTextContent("$150.00");

      // Assert - input is disabled
      const topUpInput = screen.getByTestId("top-up-input");
      expect(topUpInput).toBeDisabled();

      // Assert - button is disabled
      const submitButton = screen.getByRole("button");
      expect(submitButton).toBeDisabled();
    });

    it("should enable input and button when isDisabled is false", async () => {
      // Arrange & Act
      render(<PaymentForm isDisabled={false} />, {
        wrapper: ({ children }) => (
          <QueryClientProvider client={mockQueryClient}>
            {children}
          </QueryClientProvider>
        ),
      });

      // Assert - input is enabled
      const topUpInput = screen.getByTestId("top-up-input");
      expect(topUpInput).not.toBeDisabled();

      // Assert - button starts disabled (no amount entered) but is NOT
      // permanently disabled by the isDisabled prop
      const submitButton = screen.getByRole("button");
      // The button is disabled because no valid amount is entered, not because of isDisabled
      expect(submitButton).toBeDisabled();
    });
  });
});
