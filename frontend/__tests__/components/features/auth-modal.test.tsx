import { render, screen, waitFor } from "@testing-library/react";
import { it, describe, expect, vi, beforeEach, afterEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { UseQueryResult } from "@tanstack/react-query";
import { AuthModal } from "#/components/features/waitlist/auth-modal";
import { useRecaptcha } from "#/hooks/use-recaptcha";
import type { UseRecaptchaReturn } from "#/hooks/use-recaptcha";
import { useConfig } from "#/hooks/query/use-config";
import type { GetConfigResponse } from "#/api/option-service/option.types";
import { useTracking } from "#/hooks/use-tracking";

// Mock the useAuthUrl hook
vi.mock("#/hooks/use-auth-url", () => ({
  useAuthUrl: () => "https://gitlab.com/oauth/authorize",
}));

// Mock the useTracking hook
vi.mock("#/hooks/use-tracking");
// Mock the useConfig hook (needed for reCAPTCHA)
vi.mock("#/hooks/query/use-config");
// Mock the useRecaptcha hook (needed for reCAPTCHA)
vi.mock("#/hooks/use-recaptcha");

// Helper functions to create minimal mocks
const createUseConfigMock = (
  data: Partial<GetConfigResponse> = {},
): UseQueryResult<GetConfigResponse> =>
  ({
    data: data as GetConfigResponse,
    isLoading: false,
    isError: false,
  }) as UseQueryResult<GetConfigResponse>;

const createUseTrackingMock = (
  overrides?: Partial<ReturnType<typeof useTracking>>,
) => ({
  trackLoginButtonClick: vi.fn(),
  ...overrides,
});

describe("AuthModal", () => {
  beforeEach(() => {
    vi.stubGlobal("location", { href: "" });
    // Set up default mocks for all tests
    vi.mocked(useTracking).mockReturnValue(
      createUseTrackingMock() as ReturnType<typeof useTracking>,
    );
    vi.mocked(useConfig).mockReturnValue(createUseConfigMock());
    vi.mocked(useRecaptcha).mockReturnValue({
      isReady: false,
      isLoading: false,
      error: null,
      executeRecaptcha: vi.fn(),
    } as UseRecaptchaReturn);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.resetAllMocks();
  });

  it("should render the GitHub and GitLab buttons", () => {
    render(
      <MemoryRouter>
        <AuthModal
          githubAuthUrl="mock-url"
          appMode="saas"
          providersConfigured={["github", "gitlab"]}
        />
      </MemoryRouter>,
    );

    const githubButton = screen.getByRole("button", {
      name: "GITHUB$CONNECT_TO_GITHUB",
    });
    const gitlabButton = screen.getByRole("button", {
      name: "GITLAB$CONNECT_TO_GITLAB",
    });

    expect(githubButton).toBeInTheDocument();
    expect(gitlabButton).toBeInTheDocument();
  });

  it("should redirect to GitHub auth URL when GitHub button is clicked", async () => {
    const user = userEvent.setup();
    const mockUrl = "https://github.com/login/oauth/authorize";
    render(
      <MemoryRouter>
        <AuthModal
          githubAuthUrl={mockUrl}
          appMode="saas"
          providersConfigured={["github"]}
        />
      </MemoryRouter>,
    );

    const githubButton = screen.getByRole("button", {
      name: "GITHUB$CONNECT_TO_GITHUB",
    });
    await user.click(githubButton);

    expect(window.location.href).toBe(mockUrl);
  });

  it("should render Terms of Service and Privacy Policy text with correct links", () => {
    render(
      <MemoryRouter>
        <AuthModal githubAuthUrl="mock-url" appMode="saas" />
      </MemoryRouter>,
    );

    // Find the terms of service section using data-testid
    const termsSection = screen.getByTestId("terms-and-privacy-notice");
    expect(termsSection).toBeInTheDocument();

    // Check that all text content is present in the paragraph
    expect(termsSection).toHaveTextContent(
      "AUTH$BY_SIGNING_UP_YOU_AGREE_TO_OUR",
    );
    expect(termsSection).toHaveTextContent("COMMON$TERMS_OF_SERVICE");
    expect(termsSection).toHaveTextContent("COMMON$AND");
    expect(termsSection).toHaveTextContent("COMMON$PRIVACY_POLICY");

    // Check Terms of Service link
    const tosLink = screen.getByRole("link", {
      name: "COMMON$TERMS_OF_SERVICE",
    });
    expect(tosLink).toBeInTheDocument();
    expect(tosLink).toHaveAttribute("href", "https://www.all-hands.dev/tos");
    expect(tosLink).toHaveAttribute("target", "_blank");
    expect(tosLink).toHaveClass("underline", "hover:text-primary");

    // Check Privacy Policy link
    const privacyLink = screen.getByRole("link", {
      name: "COMMON$PRIVACY_POLICY",
    });
    expect(privacyLink).toBeInTheDocument();
    expect(privacyLink).toHaveAttribute(
      "href",
      "https://www.all-hands.dev/privacy",
    );
    expect(privacyLink).toHaveAttribute("target", "_blank");
    expect(privacyLink).toHaveClass("underline", "hover:text-primary");

    // Verify that both links are within the terms section
    expect(termsSection).toContainElement(tosLink);
    expect(termsSection).toContainElement(privacyLink);
  });

  it("should display email verified message when emailVerified prop is true", () => {
    render(
      <MemoryRouter>
        <AuthModal
          githubAuthUrl="mock-url"
          appMode="saas"
          emailVerified={true}
        />
      </MemoryRouter>,
    );

    expect(
      screen.getByText("AUTH$EMAIL_VERIFIED_PLEASE_LOGIN"),
    ).toBeInTheDocument();
  });

  it("should not display email verified message when emailVerified prop is false", () => {
    render(
      <MemoryRouter>
        <AuthModal
          githubAuthUrl="mock-url"
          appMode="saas"
          emailVerified={false}
        />
      </MemoryRouter>,
    );

    expect(
      screen.queryByText("AUTH$EMAIL_VERIFIED_PLEASE_LOGIN"),
    ).not.toBeInTheDocument();
  });

  it("should open Terms of Service link in new tab", () => {
    render(
      <MemoryRouter>
        <AuthModal githubAuthUrl="mock-url" appMode="saas" />
      </MemoryRouter>,
    );

    const tosLink = screen.getByRole("link", {
      name: "COMMON$TERMS_OF_SERVICE",
    });
    expect(tosLink).toHaveAttribute("target", "_blank");
  });

  it("should open Privacy Policy link in new tab", () => {
    render(
      <MemoryRouter>
        <AuthModal githubAuthUrl="mock-url" appMode="saas" />
      </MemoryRouter>,
    );

    const privacyLink = screen.getByRole("link", {
      name: "COMMON$PRIVACY_POLICY",
    });
    expect(privacyLink).toHaveAttribute("target", "_blank");
  });

  describe("Duplicate email error message", () => {
    const renderAuthModalWithRouter = (initialEntries: string[]) => {
      const hasDuplicatedEmail = initialEntries.includes(
        "/?duplicated_email=true",
      );

      return render(
        <MemoryRouter initialEntries={initialEntries}>
          <AuthModal
            githubAuthUrl="mock-url"
            appMode="saas"
            providersConfigured={["github"]}
            hasDuplicatedEmail={hasDuplicatedEmail}
          />
        </MemoryRouter>,
      );
    };

    it("should display error message when duplicated_email query parameter is true", () => {
      // Arrange
      const initialEntries = ["/?duplicated_email=true"];

      // Act
      renderAuthModalWithRouter(initialEntries);

      // Assert
      const errorMessage = screen.getByText("AUTH$DUPLICATE_EMAIL_ERROR");
      expect(errorMessage).toBeInTheDocument();
    });

    it("should not display error message when duplicated_email query parameter is missing", () => {
      // Arrange
      const initialEntries = ["/"];

      // Act
      renderAuthModalWithRouter(initialEntries);

      // Assert
      const errorMessage = screen.queryByText("AUTH$DUPLICATE_EMAIL_ERROR");
      expect(errorMessage).not.toBeInTheDocument();
    });
  });

  describe("reCAPTCHA Integration", () => {
    const mockExecuteRecaptcha = vi.fn();
    const mockTrackLoginButtonClick = vi.fn();

    // Mock window.location.href
    const originalLocation = window.location;
    const mockLocationHref = vi.fn();

    const renderAuthModalWithQueryClient = (props = {}) => {
      const queryClient = new QueryClient({
        defaultOptions: {
          queries: { retry: false },
        },
      });

      return render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter>
            <AuthModal
              githubAuthUrl="https://auth.example.com/github"
              appMode="saas"
              providersConfigured={["github"]}
              {...props}
            />
          </MemoryRouter>
        </QueryClientProvider>,
      );
    };

    beforeEach(() => {
      vi.clearAllMocks();
      delete (window as { location?: Location }).location;
      (window as { location: Location }).location = {
        ...originalLocation,
        href: "",
      } as Location;
      Object.defineProperty(window.location, "href", {
        set: mockLocationHref,
        get: () => "",
        configurable: true,
      });
      Object.defineProperty(window.location, "origin", {
        value: "https://example.com",
        writable: true,
      });

      // Override mocks for reCAPTCHA tests
      vi.mocked(useTracking).mockReturnValue(
        createUseTrackingMock({
          trackLoginButtonClick: mockTrackLoginButtonClick,
        }) as ReturnType<typeof useTracking>,
      );

      vi.mocked(useConfig).mockReturnValue(
        createUseConfigMock({ RECAPTCHA_SITE_KEY: "test-site-key" }),
      );

      vi.mocked(useRecaptcha).mockReturnValue({
        isReady: false,
        isLoading: false,
        error: null,
        executeRecaptcha: mockExecuteRecaptcha,
      } as UseRecaptchaReturn);
    });

    afterEach(() => {
      Object.defineProperty(window, "location", {
        value: originalLocation,
        writable: true,
        configurable: true,
      });
    });

    it("should display error message when recaptchaBlocked prop is true", () => {
      // Arrange
      vi.mocked(useRecaptcha).mockReturnValue({
        isReady: false,
        isLoading: false,
        error: null,
        executeRecaptcha: mockExecuteRecaptcha,
      } as UseRecaptchaReturn);

      // Act
      renderAuthModalWithQueryClient({ recaptchaBlocked: true });

      // Assert
      expect(screen.getByText("AUTH$RECAPTCHA_ERROR")).toBeInTheDocument();
    });

    it("should redirect without token when reCAPTCHA is not configured", async () => {
      // Arrange
      const user = userEvent.setup();
      vi.mocked(useConfig).mockReturnValue(createUseConfigMock());
      vi.mocked(useRecaptcha).mockReturnValue({
        isReady: false,
        isLoading: false,
        error: null,
        executeRecaptcha: mockExecuteRecaptcha,
      } as UseRecaptchaReturn);

      // Act
      renderAuthModalWithQueryClient();
      const button = screen.getByText("GITHUB$CONNECT_TO_GITHUB");
      await user.click(button);

      // Assert
      await waitFor(() => {
        expect(mockTrackLoginButtonClick).toHaveBeenCalledWith({
          provider: "github",
        });
        expect(mockLocationHref).toHaveBeenCalledWith(
          "https://auth.example.com/github",
        );
      });
      expect(mockExecuteRecaptcha).not.toHaveBeenCalled();
    });

    it("should generate token and encode in state when reCAPTCHA is configured and ready", async () => {
      // Arrange
      const user = userEvent.setup();
      const mockToken = "test-recaptcha-token-123";
      mockExecuteRecaptcha.mockResolvedValue(mockToken);
      vi.mocked(useRecaptcha).mockReturnValue({
        isReady: true,
        isLoading: false,
        error: null,
        executeRecaptcha: mockExecuteRecaptcha,
      } as UseRecaptchaReturn);

      // Act
      renderAuthModalWithQueryClient();
      const button = screen.getByText("GITHUB$CONNECT_TO_GITHUB");
      await user.click(button);

      // Assert
      await waitFor(() => {
        expect(mockExecuteRecaptcha).toHaveBeenCalledWith("LOGIN");
      });

      await waitFor(() => {
        expect(mockLocationHref).toHaveBeenCalled();
        const callArgs = mockLocationHref.mock.calls[0][0];
        expect(callArgs).toContain("https://auth.example.com/github");
        const url = new URL(callArgs);
        const stateParam = url.searchParams.get("state");
        expect(stateParam).toBeTruthy();
        const decodedState = JSON.parse(atob(stateParam!));
        expect(decodedState.recaptcha_token).toBe(mockToken);
        expect(decodedState.redirect_url).toBe("https://example.com");
      });
    });

    it("should redirect without token when reCAPTCHA token generation fails", async () => {
      // Arrange
      const user = userEvent.setup();
      const consoleErrorSpy = vi
        .spyOn(console, "error")
        .mockImplementation(() => {});
      mockExecuteRecaptcha.mockRejectedValue(
        new Error("Token generation failed"),
      );
      vi.mocked(useRecaptcha).mockReturnValue({
        isReady: true,
        isLoading: false,
        error: null,
        executeRecaptcha: mockExecuteRecaptcha,
      } as UseRecaptchaReturn);

      // Act
      renderAuthModalWithQueryClient();
      const button = screen.getByText("GITHUB$CONNECT_TO_GITHUB");
      await user.click(button);

      // Assert
      await waitFor(() => {
        expect(mockExecuteRecaptcha).toHaveBeenCalledWith("LOGIN");
        expect(consoleErrorSpy).toHaveBeenCalledWith(
          "reCAPTCHA token generation failed:",
          expect.any(Error),
        );
        // Should still redirect even on failure (fail open)
        expect(mockLocationHref).not.toHaveBeenCalled();
      });
      consoleErrorSpy.mockRestore();
    });

    it("should track login click when clicking auth button with reCAPTCHA", async () => {
      // Arrange
      const user = userEvent.setup();
      mockExecuteRecaptcha.mockResolvedValue("test-token");
      vi.mocked(useRecaptcha).mockReturnValue({
        isReady: true,
        isLoading: false,
        error: null,
        executeRecaptcha: mockExecuteRecaptcha,
      } as UseRecaptchaReturn);

      // Act
      renderAuthModalWithQueryClient();
      const button = screen.getByText("GITHUB$CONNECT_TO_GITHUB");
      await user.click(button);

      // Assert
      await waitFor(() => {
        expect(mockTrackLoginButtonClick).toHaveBeenCalledWith({
          provider: "github",
        });
      });
    });

    it("should redirect normally when reCAPTCHA is not ready", async () => {
      // Arrange
      const user = userEvent.setup();
      vi.mocked(useRecaptcha).mockReturnValue({
        isReady: false,
        isLoading: true,
        error: null,
        executeRecaptcha: mockExecuteRecaptcha,
      } as UseRecaptchaReturn);

      // Act
      renderAuthModalWithQueryClient();
      const button = screen.getByText("GITHUB$CONNECT_TO_GITHUB");
      await user.click(button);

      // Assert
      await waitFor(() => {
        expect(mockTrackLoginButtonClick).toHaveBeenCalledWith({
          provider: "github",
        });
        expect(mockLocationHref).toHaveBeenCalledWith(
          "https://auth.example.com/github",
        );
      });
      expect(mockExecuteRecaptcha).not.toHaveBeenCalled();
    });

    it("should preserve existing state parameter when encoding reCAPTCHA token", async () => {
      // Arrange
      const user = userEvent.setup();
      const mockToken = "test-token";
      const existingState = "existing-state-value";
      mockExecuteRecaptcha.mockResolvedValue(mockToken);
      vi.mocked(useRecaptcha).mockReturnValue({
        isReady: true,
        isLoading: false,
        error: null,
        executeRecaptcha: mockExecuteRecaptcha,
      } as UseRecaptchaReturn);

      // Act
      renderAuthModalWithQueryClient({
        githubAuthUrl: `https://auth.example.com/github?state=${existingState}`,
      });
      const button = screen.getByText("GITHUB$CONNECT_TO_GITHUB");
      await user.click(button);

      // Assert
      await waitFor(() => {
        expect(mockLocationHref).toHaveBeenCalled();
        const callArgs = mockLocationHref.mock.calls[0][0];
        const url = new URL(callArgs);
        const stateParam = url.searchParams.get("state");
        const decodedState = JSON.parse(atob(stateParam!));
        expect(decodedState.redirect_url).toBe(existingState);
        expect(decodedState.recaptcha_token).toBe(mockToken);
      });
    });
  });
});
