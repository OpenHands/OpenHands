import { render, screen, waitFor } from "@testing-library/react";
import { createRoutesStub } from "react-router";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { QueryClientProvider } from "@tanstack/react-query";
import BillingSettingsScreen, { clientLoader } from "#/routes/billing";
import OptionService from "#/api/option-service/option-service.api";
import { OrganizationMember } from "#/types/org";
import * as orgStore from "#/stores/selected-organization-store";
import { organizationService } from "#/api/organization-service/organization-service.api";

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

// Mock useTracking hook
vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackCreditsPurchased: vi.fn(),
  }),
}));

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
    // @ts-expect-error - partial mock for testing
    vi.spyOn(OptionService, "getConfig").mockResolvedValue({
      app_mode: "saas",
      posthog_client_key: "test",
      feature_flags: {
        enable_billing: false,
        hide_llm_settings: false,
        enable_jira: false,
        enable_jira_dc: false,
        enable_linear: false,
        ...featureFlags,
      },
    });
  };

  beforeEach(() => {
    mockQueryClient.clear();
  });

  afterEach(() => {
    vi.clearAllMocks();
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
});
