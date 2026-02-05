import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createRoutesStub, useSearchParams } from "react-router";
import MainApp from "#/routes/root-layout";
import OptionService from "#/api/option-service/option-service.api";
import AuthService from "#/api/auth-service/auth-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";

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

function LoginStub() {
  const [searchParams] = useSearchParams();
  const emailVerificationRequired =
    searchParams.get("email_verification_required") === "true";
  const emailVerified = searchParams.get("email_verified") === "true";
  const emailVerificationText = "AUTH$PLEASE_CHECK_EMAIL_TO_VERIFY";
  const returnTo = searchParams.get("returnTo");

  return (
    <div data-testid="login-page">
      <div data-testid="login-content">
        {emailVerified && <div data-testid="email-verified-message" />}
        {emailVerificationRequired && (
          <div data-testid="email-verification-modal">
            {emailVerificationText}
          </div>
        )}
        {returnTo && <div data-testid="return-to-param">{returnTo}</div>}
      </div>
    </div>
  );
}

const RouterStub = createRoutesStub([
  {
    Component: MainApp,
    path: "/",
    children: [
      {
        Component: () => <div data-testid="outlet-content" />,
        path: "/",
      },
    ],
  },
  {
    Component: LoginStub,
    path: "/login",
  },
]);
const RouterStubWithLogin = createRoutesStub([
  {
    Component: MainApp,
    path: "/",
    children: [
      {
        Component: () => <div data-testid="outlet-content" />,
        path: "/",
      },
      {
        Component: () => <div data-testid="settings-page" />,
        path: "/settings",
      },
    ],
  },
  {
    Component: () => <div data-testid="login-page" />,
    path: "/login",
  },
]);

const RouterStubWithDeviceVerify = createRoutesStub([
  {
    Component: MainApp,
    path: "/",
    children: [
      {
        Component: () => <div data-testid="outlet-content" />,
        path: "/",
      },
      {
        Component: () => <div data-testid="device-verify-page" />,
        path: "/oauth/device/verify",
      },
    ],
  },
  {
    Component: LoginStub,
    path: "/login",
  },
]);

const renderMainApp = (initialEntries: string[] = ["/"]) =>
  render(<RouterStub initialEntries={initialEntries} />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({
            defaultOptions: { queries: { retry: false } },
          })
        }
      >
        {children}
      </QueryClientProvider>
    ),
  });

const renderWithLoginStub = (
  RouterStubComponent: ReturnType<typeof createRoutesStub>,
  initialEntries: string[] = ["/"],
) =>
  render(<RouterStubComponent initialEntries={initialEntries} />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({
            defaultOptions: { queries: { retry: false } },
          })
        }
      >
        {children}
      </QueryClientProvider>
    ),
  });

describe("MainApp", () => {
  beforeEach(() => {
    vi.clearAllMocks();

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

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      MOCK_DEFAULT_USER_SETTINGS,
    );

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

  describe("Email Verification", () => {
    it("should redirect to login when email_verification_required=true is in query params", async () => {
      const axiosError = {
        response: { status: 401 },
        isAxiosError: true,
      };
      vi.spyOn(AuthService, "authenticate").mockRejectedValue(axiosError);

      renderMainApp(["/?email_verification_required=true"]);

      await waitFor(
        () => {
          expect(screen.getByTestId("login-page")).toBeInTheDocument();
        },
        { timeout: 2000 },
      );
    });

    it("should redirect to login when email_verified=true is in query params", async () => {
      const axiosError = {
        response: { status: 401 },
        isAxiosError: true,
      };
      vi.spyOn(AuthService, "authenticate").mockRejectedValue(axiosError);

      renderMainApp(["/?email_verified=true"]);

      await waitFor(
        () => {
          expect(screen.getByTestId("login-page")).toBeInTheDocument();
        },
        { timeout: 2000 },
      );
    });

    it("should redirect to login when email_verification_required and email_verified params are in query params together", async () => {
      const axiosError = {
        response: { status: 401 },
        isAxiosError: true,
      };
      vi.spyOn(AuthService, "authenticate").mockRejectedValue(axiosError);

      renderMainApp(["/?email_verification_required=true&email_verified=true"]);

      await waitFor(
        () => {
          expect(screen.getByTestId("login-page")).toBeInTheDocument();
        },
        { timeout: 2000 },
      );
    });

    it("should redirect to login when email_verification_required=true is in query params", async () => {
      const axiosError = {
        response: { status: 401 },
        isAxiosError: true,
      };
      vi.spyOn(AuthService, "authenticate").mockRejectedValue(axiosError);

      renderMainApp(["/?email_verification_required=true"]);

      await waitFor(
        () => {
          expect(screen.getByTestId("login-page")).toBeInTheDocument();
        },
        { timeout: 2000 },
      );
    });

    it("should not display EmailVerificationModal when email_verification_required is not in query params", async () => {
      const axiosError = {
        response: { status: 401 },
        isAxiosError: true,
      };
      vi.spyOn(AuthService, "authenticate").mockRejectedValue(axiosError);

      renderMainApp(["/"]);

      // User will be redirected to login, but modal should not show without query param
      await waitFor(
        () => {
          expect(screen.getByTestId("login-page")).toBeInTheDocument();
          expect(
            screen.queryByTestId("email-verification-modal"),
          ).not.toBeInTheDocument();
        },
        { timeout: 2000 },
      );
    });

    it("should not display email verified message when email_verified is not in query params", async () => {
      const axiosError = {
        response: { status: 401 },
        isAxiosError: true,
      };
      vi.spyOn(AuthService, "authenticate").mockRejectedValue(axiosError);

      renderMainApp(["/login"]);

      await waitFor(
        () => {
          expect(screen.getByTestId("login-page")).toBeInTheDocument();
          expect(
            screen.queryByTestId("email-verified-message"),
          ).not.toBeInTheDocument();
        },
        { timeout: 2000 },
      );
    });
  });

  describe("Unauthenticated redirect", () => {
    beforeEach(() => {
      vi.spyOn(AuthService, "authenticate").mockRejectedValue({
        response: { status: 401 },
        isAxiosError: true,
      });
    });

    it("should redirect unauthenticated SaaS users to /login", async () => {
      renderWithLoginStub(RouterStubWithLogin, ["/"]);

      await waitFor(
        () => {
          expect(screen.getByTestId("login-page")).toBeInTheDocument();
        },
        { timeout: 2000 },
      );
    });

    it("should redirect to /login with returnTo parameter when on a specific page", async () => {
      renderWithLoginStub(RouterStubWithLogin, ["/settings"]);

      await waitFor(
        () => {
          expect(screen.getByTestId("login-page")).toBeInTheDocument();
        },
        { timeout: 2000 },
      );
    });

    it("should preserve query parameters in returnTo when redirecting to login", async () => {
      renderWithLoginStub(RouterStubWithDeviceVerify, [
        "/oauth/device/verify?user_code=F9XN6BKU",
      ]);

      await waitFor(
        () => {
          expect(screen.getByTestId("login-page")).toBeInTheDocument();
          const returnToElement = screen.getByTestId("return-to-param");
          expect(returnToElement).toBeInTheDocument();
          expect(returnToElement.textContent).toBe(
            "/oauth/device/verify?user_code=F9XN6BKU",
          );
        },
        { timeout: 2000 },
      );
    });
  });
});
