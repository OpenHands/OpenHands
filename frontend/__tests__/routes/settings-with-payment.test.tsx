import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createRoutesStub } from "react-router";
import { renderWithProviders } from "test-utils";
import SettingsScreen from "#/routes/settings";
import { PaymentForm } from "#/components/features/payment/payment-form";

// Mock the useSettings hook
vi.mock("#/hooks/query/use-settings", async () => {
  const actual = await vi.importActual<
    typeof import("#/hooks/query/use-settings")
  >("#/hooks/query/use-settings");
  return {
    ...actual,
    useSettings: vi.fn().mockReturnValue({
      data: {
        EMAIL_VERIFIED: true, // Mock email as verified to prevent redirection
      },
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
const { mockUseConfig } = vi.hoisted(() => ({
  mockUseConfig: vi.fn(),
}));
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: mockUseConfig,
}));

describe("Settings Billing", () => {
  beforeEach(() => {
    // Set default config to OSS mode
    mockUseConfig.mockReturnValue({
      data: {
        app_mode: "oss",
        github_client_id: "123",
        posthog_client_key: "456",
        feature_flags: {
          enable_billing: false,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
        },
      },
      isLoading: false,
    });
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
    renderWithProviders(<RoutesStub initialEntries={["/settings/billing"]} />);

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should not render the billing tab if OSS mode", async () => {
    // OSS mode is set by default in beforeEach
    renderSettingsScreen();

    const navbar = await screen.findByTestId("settings-navbar");
    const credits = within(navbar).queryByText("Billing");
    expect(credits).not.toBeInTheDocument();
  });

  it("should render the billing tab if SaaS mode and billing is enabled", async () => {
    mockUseConfig.mockReturnValue({
      data: {
        app_mode: "saas",
        github_client_id: "123",
        posthog_client_key: "456",
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

    renderSettingsScreen();

    const navbar = await screen.findByTestId("settings-navbar");
    within(navbar).getByText("Billing");
  });

  it("should render the billing settings if clicking the billing item", async () => {
    const user = userEvent.setup();
    mockUseConfig.mockReturnValue({
      data: {
        app_mode: "saas",
        github_client_id: "123",
        posthog_client_key: "456",
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

    renderSettingsScreen();

    const navbar = await screen.findByTestId("settings-navbar");
    const credits = within(navbar).getByText("Billing");
    await user.click(credits);

    const billingSection = await screen.findByTestId("billing-settings");
    expect(billingSection).toBeInTheDocument();
  });
});
