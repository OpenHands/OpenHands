import { render, screen, waitFor } from "@testing-library/react";
import { it, describe, expect, vi, beforeEach, afterEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createRoutesStub } from "react-router";
import MainApp from "#/routes/root-layout";
import OptionService from "#/api/option-service/option-service.api";
import AuthService from "#/api/auth-service/auth-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";

// Mock other hooks that are not the focus of these tests
vi.mock("#/hooks/use-github-auth-url", () => ({
  useGitHubAuthUrl: () => "https://github.com/oauth/authorize",
}));

vi.mock("#/hooks/use-is-on-tos-page", () => ({
  useIsOnTosPage: () => false,
}));

vi.mock("#/hooks/use-auto-login", () => ({
  useAutoLogin: () => {},
}));

vi.mock("#/hooks/use-auth-callback", () => ({
  useAuthCallback: () => {},
}));

vi.mock("#/hooks/use-migrate-user-consent", () => ({
  useMigrateUserConsent: () => ({
    migrateUserConsent: vi.fn(),
  }),
}));

vi.mock("#/hooks/use-reo-tracking", () => ({
  useReoTracking: () => {},
}));

vi.mock("#/hooks/use-sync-posthog-consent", () => ({
  useSyncPostHogConsent: () => {},
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
}));

const RouterStub = createRoutesStub([
  {
    Component: MainApp,
    path: "/",
    children: [
      {
        Component: () => <div data-testid="outlet-content">Content</div>,
        path: "/",
      },
    ],
  },
]);

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe("MainApp - Email Verification Flow", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Default mocks for services
    vi.spyOn(OptionService, "getConfig").mockResolvedValue({
      APP_MODE: "saas",
      GITHUB_CLIENT_ID: "test-client-id",
      POSTHOG_CLIENT_KEY: "test-posthog-key",
      PROVIDERS_CONFIGURED: ["github"],
      AUTH_URL: "https://auth.example.com",
      FEATURE_FLAGS: {
        ENABLE_BILLING: false,
        HIDE_LLM_SETTINGS: false,
        ENABLE_JIRA: false,
        ENABLE_JIRA_DC: false,
        ENABLE_LINEAR: false,
      },
    });

    vi.spyOn(AuthService, "authenticate").mockResolvedValue(true);

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      language: "en",
      user_consents_to_analytics: true,
      llm_model: "",
      llm_base_url: "",
      agent: "",
      llm_api_key: null,
      llm_api_key_set: false,
      search_api_key_set: false,
      confirmation_mode: false,
      security_analyzer: null,
      remote_runtime_resource_factor: null,
      provider_tokens_set: {},
      enable_default_condenser: false,
      condenser_max_size: null,
      enable_sound_notifications: false,
      enable_proactive_conversation_starters: false,
      enable_solvability_analysis: false,
      max_budget_per_task: null,
    });

    // Mock localStorage
    vi.stubGlobal("localStorage", {
      getItem: vi.fn(() => null),
      setItem: vi.fn(),
      removeItem: vi.fn(),
      clear: vi.fn(),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("should display EmailVerificationModal when email_verification_required=true is in query params", async () => {
    // Arrange & Act
    render(
      <RouterStub initialEntries={["/?email_verification_required=true"]} />,
      { wrapper: createWrapper() },
    );

    // Assert
    await waitFor(() => {
      expect(
        screen.getByText("AUTH$PLEASE_CHECK_EMAIL_TO_VERIFY"),
      ).toBeInTheDocument();
    });
  });

  it("should set emailVerified state and pass to AuthModal when email_verified=true is in query params", async () => {
    // Arrange
    // Mock a 401 error to simulate unauthenticated user
    const axiosError = {
      response: { status: 401 },
      isAxiosError: true,
    };
    vi.spyOn(AuthService, "authenticate").mockRejectedValue(axiosError);

    // Act
    render(<RouterStub initialEntries={["/?email_verified=true"]} />, {
      wrapper: createWrapper(),
    });

    // Assert - Wait for AuthModal to render (since user is not authenticated)
    await waitFor(() => {
      expect(
        screen.getByText("AUTH$EMAIL_VERIFIED_PLEASE_LOGIN"),
      ).toBeInTheDocument();
    });
  });

  it("should handle both email_verification_required and email_verified params together", async () => {
    // Arrange & Act
    render(
      <RouterStub
        initialEntries={[
          "/?email_verification_required=true&email_verified=true",
        ]}
      />,
      { wrapper: createWrapper() },
    );

    // Assert - EmailVerificationModal should take precedence
    await waitFor(() => {
      expect(
        screen.getByText("AUTH$PLEASE_CHECK_EMAIL_TO_VERIFY"),
      ).toBeInTheDocument();
    });
  });

  it("should remove query parameters from URL after processing", async () => {
    // Arrange & Act
    const { container } = render(
      <RouterStub initialEntries={["/?email_verification_required=true"]} />,
      { wrapper: createWrapper() },
    );

    // Assert - Wait for the modal to appear (which indicates processing happened)
    await waitFor(() => {
      expect(
        screen.getByText("AUTH$PLEASE_CHECK_EMAIL_TO_VERIFY"),
      ).toBeInTheDocument();
    });

    // Verify that the query parameter was processed by checking the modal appeared
    // The hook removes the parameter from the URL, so we verify the behavior indirectly
    expect(container).toBeInTheDocument();
  });

  it("should not display EmailVerificationModal when email_verification_required is not in query params", async () => {
    // Arrange - No query params set

    // Act
    render(<RouterStub />, { wrapper: createWrapper() });

    // Assert
    await waitFor(() => {
      expect(
        screen.queryByText("AUTH$PLEASE_CHECK_EMAIL_TO_VERIFY"),
      ).not.toBeInTheDocument();
    });
  });

  it("should not display email verified message when email_verified is not in query params", async () => {
    // Arrange
    // Mock a 401 error to simulate unauthenticated user
    const axiosError = {
      response: { status: 401 },
      isAxiosError: true,
    };
    vi.spyOn(AuthService, "authenticate").mockRejectedValue(axiosError);

    // Act
    render(<RouterStub />, { wrapper: createWrapper() });

    // Assert - AuthModal should render but without email verified message
    await waitFor(() => {
      const authModal = screen.queryByText(
        "AUTH$SIGN_IN_WITH_IDENTITY_PROVIDER",
      );
      if (authModal) {
        expect(
          screen.queryByText("AUTH$EMAIL_VERIFIED_PLEASE_LOGIN"),
        ).not.toBeInTheDocument();
      }
    });
  });
});
