import { render, screen, waitFor } from "@testing-library/react";
import { it, describe, expect, vi, beforeEach, afterEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createRoutesStub } from "react-router";
import MainApp from "#/routes/root-layout";
import OptionService from "#/api/option-service/option-service.api";
import AuthService from "#/api/auth-service/auth-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";

// Mock react-router hooks
let mockSearchParams: URLSearchParams;
let mockSetSearchParams: ReturnType<typeof vi.fn>;
const mockNavigate = vi.fn();
const mockLocation = { pathname: "/" };

vi.mock("react-router", async () => {
  const actual = await vi.importActual("react-router");
  return {
    ...actual,
    useSearchParams: () => {
      if (!mockSearchParams) {
        mockSearchParams = new URLSearchParams();
      }
      if (!mockSetSearchParams) {
        mockSetSearchParams = vi.fn((params: URLSearchParams) => {
          // Update mockSearchParams when setSearchParams is called
          mockSearchParams = params;
        });
      }
      return [mockSearchParams, mockSetSearchParams];
    },
    useNavigate: () => mockNavigate,
    useLocation: () => mockLocation,
  };
});

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
    mockSearchParams = new URLSearchParams();
    mockSetSearchParams = vi.fn((params: URLSearchParams) => {
      mockSearchParams = params;
    });

    // Default mocks for services
    vi.spyOn(OptionService, "getConfig").mockResolvedValue({
      APP_MODE: "saas",
      PROVIDERS_CONFIGURED: ["github"],
      AUTH_URL: "https://auth.example.com",
      FEATURE_FLAGS: {
        ENABLE_BILLING: false,
      },
    } as any);

    vi.spyOn(AuthService, "authenticate").mockResolvedValue(true);

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      language: "en",
      user_consents_to_analytics: true,
    } as any);

    // Mock localStorage
    Object.defineProperty(window, "localStorage", {
      value: {
        getItem: vi.fn(() => null),
        setItem: vi.fn(),
        removeItem: vi.fn(),
        clear: vi.fn(),
      },
      writable: true,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should display EmailVerificationModal when email_verification_required=true is in query params", async () => {
    // Arrange
    mockSearchParams.set("email_verification_required", "true");

    // Act
    render(<RouterStub />, { wrapper: createWrapper() });

    // Assert
    await waitFor(() => {
      expect(
        screen.getByText("AUTH$PLEASE_CHECK_EMAIL_TO_VERIFY"),
      ).toBeInTheDocument();
    });

    // Verify URL cleanup was called
    expect(mockSetSearchParams).toHaveBeenCalled();
  });

  it("should set emailVerified state and pass to AuthModal when email_verified=true is in query params", async () => {
    // Arrange
    mockSearchParams.set("email_verified", "true");
    // Mock a 401 error to simulate unauthenticated user
    const axiosError = {
      response: { status: 401 },
      isAxiosError: true,
    };
    vi.spyOn(AuthService, "authenticate").mockRejectedValue(axiosError);

    // Act
    render(<RouterStub />, { wrapper: createWrapper() });

    // Assert - Wait for AuthModal to render (since user is not authenticated)
    await waitFor(() => {
      expect(
        screen.getByText("AUTH$EMAIL_VERIFIED_PLEASE_LOGIN"),
      ).toBeInTheDocument();
    });

    // Verify URL cleanup was called
    expect(mockSetSearchParams).toHaveBeenCalled();
  });

  it("should handle both email_verification_required and email_verified params together", async () => {
    // Arrange
    mockSearchParams.set("email_verification_required", "true");
    mockSearchParams.set("email_verified", "true");

    // Act
    render(<RouterStub />, { wrapper: createWrapper() });

    // Assert - EmailVerificationModal should take precedence
    await waitFor(() => {
      expect(
        screen.getByText("AUTH$PLEASE_CHECK_EMAIL_TO_VERIFY"),
      ).toBeInTheDocument();
    });

    // Verify URL cleanup was called
    expect(mockSetSearchParams).toHaveBeenCalled();
  });

  it("should remove query parameters from URL after processing", async () => {
    // Arrange
    mockSearchParams.set("email_verification_required", "true");

    // Act
    render(<RouterStub />, { wrapper: createWrapper() });

    // Assert
    await waitFor(() => {
      expect(mockSetSearchParams).toHaveBeenCalled();
    });

    // Verify that setSearchParams was called with updated params
    expect(mockSetSearchParams).toHaveBeenCalled();
    const setSearchParamsCall = mockSetSearchParams.mock.calls[0];
    const updatedParams = setSearchParamsCall[0];
    expect(updatedParams.get("email_verification_required")).toBeNull();
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
