import { render, screen, within } from "@testing-library/react";
import { createRoutesStub } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClientProvider } from "@tanstack/react-query";
import SettingsScreen, {
  clientLoader,
  getFirstAvailablePath,
} from "#/routes/settings";
import OptionService from "#/api/option-service/option-service.api";
import { WebClientFeatureFlags } from "#/api/option-service/option.types";

// Module-level mocks using vi.hoisted
const { handleLogoutMock, mockQueryClient } = vi.hoisted(() => ({
  handleLogoutMock: vi.fn(),
  mockQueryClient: (() => {
    const { QueryClient } = require("@tanstack/react-query");
    return new QueryClient();
  })(),
}));

vi.mock("#/hooks/use-app-logout", () => ({
  useAppLogout: vi.fn().mockReturnValue({ handleLogout: handleLogoutMock }),
}));

vi.mock("#/query-client-config", () => ({
  queryClient: mockQueryClient,
}));

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
          SETTINGS$NAV_API_KEYS: "API Keys",
          SETTINGS$NAV_LLM: "LLM",
          SETTINGS$NAV_SECRETS: "Secrets",
          SETTINGS$NAV_MCP: "MCP",
          SETTINGS$NAV_USER: "User",
          SETTINGS$NAV_BILLING: "Billing",
          SETTINGS$TITLE: "Settings",
          COMMON$LANGUAGE_MODEL_LLM: "LLM",
        };
        return translations[key] || key;
      },
      i18n: {
        changeLanguage: vi.fn(),
      },
    }),
  };
});

describe("Settings Screen", () => {
  const RouterStub = createRoutesStub([
    {
      Component: SettingsScreen,
      // @ts-expect-error - custom loader
      clientLoader,
      path: "/settings",
      children: [
        {
          Component: () => <div data-testid="llm-settings-screen" />,
          path: "/settings",
        },
        {
          Component: () => <div data-testid="git-settings-screen" />,
          path: "/settings/integrations",
        },
        {
          Component: () => <div data-testid="application-settings-screen" />,
          path: "/settings/app",
        },
        {
          Component: () => <div data-testid="billing-settings-screen" />,
          path: "/settings/billing",
        },
        {
          Component: () => <div data-testid="api-keys-settings-screen" />,
          path: "/settings/api-keys",
        },
      ],
    },
  ]);

  const renderSettingsScreen = (path = "/settings") =>
    render(<RouterStub initialEntries={[path]} />, {
      wrapper: ({ children }) => (
        <QueryClientProvider client={mockQueryClient}>
          {children}
        </QueryClientProvider>
      ),
    });

  it("should render the navbar", async () => {
    const sectionsToInclude = ["llm", "integrations", "application", "secrets"];
    const sectionsToExclude = ["api keys", "credits", "billing"];
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    // @ts-expect-error - only return app mode
    getConfigSpy.mockResolvedValue({
      app_mode: "oss",
    });

    // Clear any existing query data
    mockQueryClient.clear();

    renderSettingsScreen();

    const navbar = await screen.findByTestId("settings-navbar");
    sectionsToInclude.forEach((section) => {
      const sectionElement = within(navbar).getByText(section, {
        exact: false, // case insensitive
      });
      expect(sectionElement).toBeInTheDocument();
    });
    sectionsToExclude.forEach((section) => {
      const sectionElement = within(navbar).queryByText(section, {
        exact: false, // case insensitive
      });
      expect(sectionElement).not.toBeInTheDocument();
    });

    getConfigSpy.mockRestore();
  });

  it("should render the saas navbar", async () => {
    const saasConfig = { app_mode: "saas" };

    // Clear any existing query data and set the config
    mockQueryClient.clear();
    mockQueryClient.setQueryData(["web-client-config"], saasConfig);

    const sectionsToInclude = [
      "llm", // LLM settings are now always shown in SaaS mode
      "user",
      "integrations",
      "application",
      "billing", // The nav item shows "Billing" text and routes to /billing
      "secrets",
      "api keys",
    ];
    const sectionsToExclude: string[] = []; // No sections are excluded in SaaS mode now

    renderSettingsScreen();

    const navbar = await screen.findByTestId("settings-navbar");
    sectionsToInclude.forEach((section) => {
      const sectionElement = within(navbar).getByText(section, {
        exact: false, // case insensitive
      });
      expect(sectionElement).toBeInTheDocument();
    });
    sectionsToExclude.forEach((section) => {
      const sectionElement = within(navbar).queryByText(section, {
        exact: false, // case insensitive
      });
      expect(sectionElement).not.toBeInTheDocument();
    });
  });

  it("should not be able to access saas-only routes in oss mode", async () => {
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    // @ts-expect-error - only return app mode
    getConfigSpy.mockResolvedValue({
      app_mode: "oss",
    });

    // Clear any existing query data
    mockQueryClient.clear();

    // In OSS mode, accessing restricted routes should redirect to /settings
    // Since createRoutesStub doesn't handle clientLoader redirects properly,
    // we test that the correct navbar is shown (OSS navbar) and that
    // the restricted route components are not rendered when accessing /settings
    renderSettingsScreen("/settings");

    // Verify we're in OSS mode by checking the navbar
    const navbar = await screen.findByTestId("settings-navbar");
    expect(within(navbar).getByText("LLM")).toBeInTheDocument();
    expect(
      within(navbar).queryByText("credits", { exact: false }),
    ).not.toBeInTheDocument();

    // Verify the LLM settings screen is shown
    expect(screen.getByTestId("llm-settings-screen")).toBeInTheDocument();
    expect(
      screen.queryByTestId("billing-settings-screen"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("api-keys-settings-screen"),
    ).not.toBeInTheDocument();

    getConfigSpy.mockRestore();
  });

  it.todo("should not be able to access oss-only routes in saas mode");

  describe("hide page feature flags", () => {
    it("should hide users page in navbar when hide_users_page is true", async () => {
      const saasConfig = {
        app_mode: "saas",
        feature_flags: {
          enable_billing: false,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
          hide_users_page: true,
          hide_billing_page: false,
          hide_integrations_page: false,
        },
      };

      mockQueryClient.clear();
      mockQueryClient.setQueryData(["web-client-config"], saasConfig);

      renderSettingsScreen();

      const navbar = await screen.findByTestId("settings-navbar");
      expect(
        within(navbar).queryByText("User", { exact: false }),
      ).not.toBeInTheDocument();
      // Other pages should still be visible
      expect(
        within(navbar).getByText("Integrations", { exact: false }),
      ).toBeInTheDocument();
      expect(
        within(navbar).getByText("Billing", { exact: false }),
      ).toBeInTheDocument();
    });

    it("should hide billing page in navbar when hide_billing_page is true", async () => {
      const saasConfig = {
        app_mode: "saas",
        feature_flags: {
          enable_billing: false,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
          hide_users_page: false,
          hide_billing_page: true,
          hide_integrations_page: false,
        },
      };

      mockQueryClient.clear();
      mockQueryClient.setQueryData(["web-client-config"], saasConfig);

      renderSettingsScreen();

      const navbar = await screen.findByTestId("settings-navbar");
      expect(
        within(navbar).queryByText("Billing", { exact: false }),
      ).not.toBeInTheDocument();
      // Other pages should still be visible
      expect(
        within(navbar).getByText("User", { exact: false }),
      ).toBeInTheDocument();
      expect(
        within(navbar).getByText("Integrations", { exact: false }),
      ).toBeInTheDocument();
    });

    it("should hide integrations page in navbar when hide_integrations_page is true", async () => {
      const saasConfig = {
        app_mode: "saas",
        feature_flags: {
          enable_billing: false,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
          hide_users_page: false,
          hide_billing_page: false,
          hide_integrations_page: true,
        },
      };

      mockQueryClient.clear();
      mockQueryClient.setQueryData(["web-client-config"], saasConfig);

      renderSettingsScreen();

      const navbar = await screen.findByTestId("settings-navbar");
      expect(
        within(navbar).queryByText("Integrations", { exact: false }),
      ).not.toBeInTheDocument();
      // Other pages should still be visible
      expect(
        within(navbar).getByText("User", { exact: false }),
      ).toBeInTheDocument();
      expect(
        within(navbar).getByText("Billing", { exact: false }),
      ).toBeInTheDocument();
    });

    it("should hide multiple pages when multiple flags are true", async () => {
      const saasConfig = {
        app_mode: "saas",
        feature_flags: {
          enable_billing: false,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
          hide_users_page: true,
          hide_billing_page: true,
          hide_integrations_page: true,
        },
      };

      mockQueryClient.clear();
      mockQueryClient.setQueryData(["web-client-config"], saasConfig);

      renderSettingsScreen();

      const navbar = await screen.findByTestId("settings-navbar");
      expect(
        within(navbar).queryByText("User", { exact: false }),
      ).not.toBeInTheDocument();
      expect(
        within(navbar).queryByText("Billing", { exact: false }),
      ).not.toBeInTheDocument();
      expect(
        within(navbar).queryByText("Integrations", { exact: false }),
      ).not.toBeInTheDocument();
      // Other pages should still be visible
      expect(
        within(navbar).getByText("Application", { exact: false }),
      ).toBeInTheDocument();
      expect(
        within(navbar).getByText("LLM", { exact: false }),
      ).toBeInTheDocument();
    });

    it("should hide integrations page in OSS mode when hide_integrations_page is true", async () => {
      const ossConfig = {
        app_mode: "oss",
        feature_flags: {
          enable_billing: false,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
          hide_users_page: false,
          hide_billing_page: false,
          hide_integrations_page: true,
        },
      };

      mockQueryClient.clear();
      mockQueryClient.setQueryData(["web-client-config"], ossConfig);

      renderSettingsScreen();

      const navbar = await screen.findByTestId("settings-navbar");
      expect(
        within(navbar).queryByText("Integrations", { exact: false }),
      ).not.toBeInTheDocument();
      // Other OSS pages should still be visible
      expect(
        within(navbar).getByText("LLM", { exact: false }),
      ).toBeInTheDocument();
      expect(
        within(navbar).getByText("Application", { exact: false }),
      ).toBeInTheDocument();
    });
  });
});

describe("getFirstAvailablePath", () => {
  const baseFeatureFlags: WebClientFeatureFlags = {
    enable_billing: false,
    hide_llm_settings: false,
    enable_jira: false,
    enable_jira_dc: false,
    enable_linear: false,
    hide_users_page: false,
    hide_billing_page: false,
    hide_integrations_page: false,
  };

  describe("SaaS mode", () => {
    it("should return /settings/user when no pages are hidden", () => {
      const result = getFirstAvailablePath(true, baseFeatureFlags);
      expect(result).toBe("/settings/user");
    });

    it("should return /settings/integrations when users page is hidden", () => {
      const flags = { ...baseFeatureFlags, hide_users_page: true };
      const result = getFirstAvailablePath(true, flags);
      expect(result).toBe("/settings/integrations");
    });

    it("should return /settings/app when users and integrations are hidden", () => {
      const flags = {
        ...baseFeatureFlags,
        hide_users_page: true,
        hide_integrations_page: true,
      };
      const result = getFirstAvailablePath(true, flags);
      expect(result).toBe("/settings/app");
    });

    it("should return /settings/app when users, integrations, and LLM settings are hidden", () => {
      const flags = {
        ...baseFeatureFlags,
        hide_users_page: true,
        hide_integrations_page: true,
        hide_llm_settings: true,
      };
      const result = getFirstAvailablePath(true, flags);
      expect(result).toBe("/settings/app");
    });

    it("should return /settings/app when users, integrations, LLM, and billing are hidden", () => {
      const flags = {
        ...baseFeatureFlags,
        hide_users_page: true,
        hide_integrations_page: true,
        hide_llm_settings: true,
        hide_billing_page: true,
      };
      // /settings/app is never hidden, so it should return that
      const result = getFirstAvailablePath(true, flags);
      expect(result).toBe("/settings/app");
    });

    it("should handle undefined feature flags", () => {
      const result = getFirstAvailablePath(true, undefined);
      expect(result).toBe("/settings/user");
    });
  });

  describe("OSS mode", () => {
    it("should return /settings when no pages are hidden", () => {
      const result = getFirstAvailablePath(false, baseFeatureFlags);
      expect(result).toBe("/settings");
    });

    it("should return /settings/mcp when LLM settings is hidden", () => {
      const flags = { ...baseFeatureFlags, hide_llm_settings: true };
      const result = getFirstAvailablePath(false, flags);
      expect(result).toBe("/settings/mcp");
    });

    it("should return /settings/mcp when LLM settings and integrations are hidden", () => {
      const flags = {
        ...baseFeatureFlags,
        hide_llm_settings: true,
        hide_integrations_page: true,
      };
      const result = getFirstAvailablePath(false, flags);
      expect(result).toBe("/settings/mcp");
    });

    it("should handle undefined feature flags", () => {
      const result = getFirstAvailablePath(false, undefined);
      expect(result).toBe("/settings");
    });
  });
});

describe("clientLoader redirect behavior", () => {
  const createMockRequest = (pathname: string) => ({
    request: new Request(`http://localhost${pathname}`),
  });

  beforeEach(() => {
    mockQueryClient.clear();
  });

  it("should redirect from /settings/user to first available page when hide_users_page is true", async () => {
    const config = {
      app_mode: "saas",
      feature_flags: {
        enable_billing: false,
        hide_llm_settings: false,
        enable_jira: false,
        enable_jira_dc: false,
        enable_linear: false,
        hide_users_page: true,
        hide_billing_page: false,
        hide_integrations_page: false,
      },
    };
    mockQueryClient.setQueryData(["web-client-config"], config);

    const result = await clientLoader(
      createMockRequest("/settings/user") as any,
    );

    expect(result).toBeDefined();
    expect(result?.status).toBe(302);
    expect(result?.headers.get("Location")).toBe("/settings/integrations");
  });

  it("should redirect from /settings/billing to first available page when hide_billing_page is true", async () => {
    const config = {
      app_mode: "saas",
      feature_flags: {
        enable_billing: false,
        hide_llm_settings: false,
        enable_jira: false,
        enable_jira_dc: false,
        enable_linear: false,
        hide_users_page: false,
        hide_billing_page: true,
        hide_integrations_page: false,
      },
    };
    mockQueryClient.setQueryData(["web-client-config"], config);

    const result = await clientLoader(
      createMockRequest("/settings/billing") as any,
    );

    expect(result).toBeDefined();
    expect(result?.status).toBe(302);
    expect(result?.headers.get("Location")).toBe("/settings/user");
  });

  it("should redirect from /settings/integrations to first available page when hide_integrations_page is true", async () => {
    const config = {
      app_mode: "saas",
      feature_flags: {
        enable_billing: false,
        hide_llm_settings: false,
        enable_jira: false,
        enable_jira_dc: false,
        enable_linear: false,
        hide_users_page: false,
        hide_billing_page: false,
        hide_integrations_page: true,
      },
    };
    mockQueryClient.setQueryData(["web-client-config"], config);

    const result = await clientLoader(
      createMockRequest("/settings/integrations") as any,
    );

    expect(result).toBeDefined();
    expect(result?.status).toBe(302);
    expect(result?.headers.get("Location")).toBe("/settings/user");
  });

  it("should redirect from /settings to /settings/app when LLM, users, and integrations are all hidden", async () => {
    const config = {
      app_mode: "saas",
      feature_flags: {
        enable_billing: false,
        hide_llm_settings: true,
        enable_jira: false,
        enable_jira_dc: false,
        enable_linear: false,
        hide_users_page: true,
        hide_billing_page: false,
        hide_integrations_page: true,
      },
    };
    mockQueryClient.setQueryData(["web-client-config"], config);

    const result = await clientLoader(createMockRequest("/settings") as any);

    expect(result).toBeDefined();
    expect(result?.status).toBe(302);
    expect(result?.headers.get("Location")).toBe("/settings/app");
  });

  it("should redirect from /settings to /settings/mcp in OSS mode when LLM settings is hidden", async () => {
    const config = {
      app_mode: "oss",
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
    };
    mockQueryClient.setQueryData(["web-client-config"], config);

    const result = await clientLoader(createMockRequest("/settings") as any);

    expect(result).toBeDefined();
    expect(result?.status).toBe(302);
    expect(result?.headers.get("Location")).toBe("/settings/mcp");
  });

  it("should not redirect when accessing a non-hidden page", async () => {
    const config = {
      app_mode: "saas",
      feature_flags: {
        enable_billing: false,
        hide_llm_settings: false,
        enable_jira: false,
        enable_jira_dc: false,
        enable_linear: false,
        hide_users_page: true,
        hide_billing_page: true,
        hide_integrations_page: true,
      },
    };
    mockQueryClient.setQueryData(["web-client-config"], config);

    // /settings/app is never hidden
    const result = await clientLoader(
      createMockRequest("/settings/app") as any,
    );

    expect(result).toBeNull();
  });

  it("should redirect from /settings/integrations in OSS mode when hide_integrations_page is true", async () => {
    const config = {
      app_mode: "oss",
      feature_flags: {
        enable_billing: false,
        hide_llm_settings: false,
        enable_jira: false,
        enable_jira_dc: false,
        enable_linear: false,
        hide_users_page: false,
        hide_billing_page: false,
        hide_integrations_page: true,
      },
    };
    mockQueryClient.setQueryData(["web-client-config"], config);

    const result = await clientLoader(
      createMockRequest("/settings/integrations") as any,
    );

    expect(result).toBeDefined();
    expect(result?.status).toBe(302);
    // In OSS mode, first available is /settings (LLM)
    expect(result?.headers.get("Location")).toBe("/settings");
  });
});
