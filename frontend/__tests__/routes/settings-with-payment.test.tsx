import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createRoutesStub } from "react-router";
import SettingsScreen from "#/routes/settings";
import { PaymentForm } from "#/components/features/payment/payment-form";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

let queryClient: QueryClient;

// Mock the useSettings hook
vi.mock("#/hooks/query/use-settings", async () => {
  const actual = await vi.importActual<typeof import("#/hooks/query/use-settings")>(
    "#/hooks/query/use-settings"
  );
  return {
    ...actual,
    useSettings: vi.fn().mockReturnValue({
      data: { EMAIL_VERIFIED: true },
      isLoading: false,
    }),
  };
});

// Mock the i18next hook
vi.mock("react-i18next", async () => {
  const actual =
    await vi.importActual<typeof import("react-i18next")>("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => {
        const translations: Record<string, string> = {
          SETTINGS$NAV_INTEGRATIONS: "Integrations",
          SETTINGS$NAV_APPLICATION: "Application",
          SETTINGS$NAV_CREDITS: "Credits",
          SETTINGS$NAV_BILLING: "Billing",
          SETTINGS$NAV_API_KEYS: "API Keys",
          SETTINGS$NAV_LLM: "LLM",
          SETTINGS$NAV_USER: "User",
          SETTINGS$NAV_SECRETS: "Secrets",
          SETTINGS$NAV_MCP: "MCP",
          SETTINGS$TITLE: "Settings",
        };
        return translations[key] || key;
      },
      i18n: {
        changeLanguage: vi.fn(),
      },
    }),
  };
});

// Mock useConfig hook
const { mockUseConfig, mockUseMe, mockUsePermission } = vi.hoisted(() => ({
  mockUseConfig: vi.fn(),
  mockUseMe: vi.fn(),
  mockUsePermission: vi.fn(),
}));

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: mockUseConfig,
}));

vi.mock("#/hooks/query/use-me", () => ({
  useMe: mockUseMe,
}));

vi.mock("#/hooks/organizations/use-permissions", () => ({
  usePermission: () => ({
    hasPermission: mockUsePermission,
  }),
}));

describe("Settings Billing", () => {
  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    // Set default config to OSS mode with lowercase keys
    mockUseConfig.mockReturnValue({
      data: {
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
      },
      isLoading: false,
    });

    mockUseMe.mockReturnValue({
      data: { role: "admin" },
      isLoading: false,
    });

    mockUsePermission.mockReturnValue(false); // default: no billing access
  });

  const RoutesStub = createRoutesStub([
    {
      Component: SettingsScreen,
      path: "/settings",
      children: [
        {
          Component: () => <PaymentForm />,
          path: "/settings/billing",
        },
        {
          Component: () => <div data-testid="git-settings-screen" />,
          path: "/settings/integrations",
        },
        {
          Component: () => <div data-testid="user-settings-screen" />,
          path: "/settings/user",
        },
      ],
    },
  ]);

  const renderSettingsScreen = () =>
    render(<RoutesStub initialEntries={["/settings"]} />, {
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      ),
    });

  afterEach(() => vi.clearAllMocks());

  it("should not render the billing tab if OSS mode", async () => {
    mockUseConfig.mockReturnValue({
      data: {
        app_mode: "oss",
        feature_flags: {
          enable_billing: true,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
        },
      },
      isLoading: false,
    });

    mockUseMe.mockReturnValue({
      data: { role: "admin" },
      isLoading: false,
    });

    mockUsePermission.mockReturnValue(true);

    renderSettingsScreen();

    const navbar = await screen.findByTestId("settings-navbar");
    const credits = within(navbar).queryByText("Billing");
    expect(credits).not.toBeInTheDocument();
  });

  it("should render the billing tab if: SaaS mode, billing enabled, admin user", async () => {
    mockUseConfig.mockReturnValue({
      data: {
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
      },
      isLoading: false,
    });

    mockUseMe.mockReturnValue({
      data: { role: "admin" },
      isLoading: false,
    });

    mockUsePermission.mockReturnValue(true);

    renderSettingsScreen();

    const navbar = await screen.findByTestId("settings-navbar");
    expect(within(navbar).getByText("Billing")).toBeInTheDocument();
  });

  it("should NOT render the billing tab if: SaaS mode, billing is enabled, and member user", async () => {
    mockUseConfig.mockReturnValue({
      data: {
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
      },
      isLoading: false,
    });

    mockUseMe.mockReturnValue({
      data: { role: "member" },
      isLoading: false,
    });

    mockUsePermission.mockReturnValue(false);

    renderSettingsScreen();

    const navbar = await screen.findByTestId("settings-navbar");
    expect(within(navbar).queryByText("Billing")).not.toBeInTheDocument();
  });

  it("should render the billing settings if clicking the billing item", async () => {
    const user = userEvent.setup();
    // When enable_billing is true, the billing nav item is shown
    mockUseConfig.mockReturnValue({
      data: {
        app_mode: "saas",
        feature_flags: {
          enable_billing: true,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
        },
      },
      isLoading: false,
    });

    mockUseMe.mockReturnValue({
      data: { role: "admin" },
      isLoading: false,
    });

    mockUsePermission.mockReturnValue(true);

    renderSettingsScreen();

    const navbar = await screen.findByTestId("settings-navbar");
    const credits = within(navbar).getByText("Billing");
    await user.click(credits);

    const billingSection = await screen.findByTestId("billing-settings");
    expect(billingSection).toBeInTheDocument();
  });
});
