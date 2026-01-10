import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createRoutesStub } from "react-router";
import LoginPage from "#/routes/login";
import OptionService from "#/api/option-service/option-service.api";
import AuthService from "#/api/auth-service/auth-service.api";

const { useEmailVerificationMock } = vi.hoisted(() => ({
  useEmailVerificationMock: vi.fn(() => ({
    emailVerified: false,
    hasDuplicatedEmail: false,
    emailVerificationModalOpen: false,
    setEmailVerificationModalOpen: vi.fn(),
  })),
}));

vi.mock("#/hooks/use-github-auth-url", () => ({
  useGitHubAuthUrl: () => "https://github.com/login/oauth/authorize",
}));

vi.mock("#/hooks/use-email-verification", () => ({
  useEmailVerification: () => useEmailVerificationMock(),
}));

const { useAuthUrlMock } = vi.hoisted(() => ({
  useAuthUrlMock: vi.fn(
    (config: { identityProvider: string; appMode: string | null }) => {
      const urls: Record<string, string> = {
        gitlab: "https://gitlab.com/oauth/authorize",
        bitbucket: "https://bitbucket.org/site/oauth2/authorize",
      };
      if (config.appMode === "saas") {
        return (
          urls[config.identityProvider] || "https://gitlab.com/oauth/authorize"
        );
      }
      return null;
    },
  ),
}));

vi.mock("#/hooks/use-auth-url", () => ({
  useAuthUrl: (config: { identityProvider: string; appMode: string | null }) =>
    useAuthUrlMock(config),
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackLoginButtonClick: vi.fn(),
  }),
}));

const RouterStub = createRoutesStub([
  {
    Component: LoginPage,
    path: "/login",
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

  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  }

  return Wrapper;
};

describe("LoginPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal("location", { href: "" });

    vi.spyOn(OptionService, "getConfig").mockResolvedValue({
      APP_MODE: "saas",
      GITHUB_CLIENT_ID: "test-client-id",
      POSTHOG_CLIENT_KEY: "test-posthog-key",
      PROVIDERS_CONFIGURED: ["github", "gitlab", "bitbucket"],
      AUTH_URL: "https://auth.example.com",
      FEATURE_FLAGS: {
        ENABLE_BILLING: false,
        HIDE_LLM_SETTINGS: false,
        ENABLE_JIRA: false,
        ENABLE_JIRA_DC: false,
        ENABLE_LINEAR: false,
      },
    });

    vi.spyOn(AuthService, "authenticate").mockRejectedValue({
      response: { status: 401 },
      isAxiosError: true,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  describe("Rendering", () => {
    it("should render login page with heading", async () => {
      render(<RouterStub initialEntries={["/login"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(screen.getByTestId("login-page")).toBeInTheDocument();
      });

      expect(screen.getByText("AUTH$LETS_GET_STARTED")).toBeInTheDocument();
    });

    it("should display all configured provider buttons", async () => {
      render(<RouterStub initialEntries={["/login"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(screen.getByTestId("login-content")).toBeInTheDocument();
      });

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "GITHUB$CONNECT_TO_GITHUB" }),
        ).toBeInTheDocument();
        expect(
          screen.getByRole("button", { name: "GITLAB$CONNECT_TO_GITLAB" }),
        ).toBeInTheDocument();
        expect(
          screen.getByRole("button", {
            name: /BITBUCKET\$CONNECT_TO_BITBUCKET/i,
          }),
        ).toBeInTheDocument();
      });
    });

    it("should only display configured providers", async () => {
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

      render(<RouterStub initialEntries={["/login"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "GITHUB$CONNECT_TO_GITHUB" }),
        ).toBeInTheDocument();
      });

      expect(
        screen.queryByRole("button", { name: "GITLAB$CONNECT_TO_GITLAB" }),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByRole("button", {
          name: "BITBUCKET$CONNECT_TO_BITBUCKET",
        }),
      ).not.toBeInTheDocument();
    });

    it("should display message when no providers are configured", async () => {
      vi.spyOn(OptionService, "getConfig").mockResolvedValue({
        APP_MODE: "saas",
        GITHUB_CLIENT_ID: "test-client-id",
        POSTHOG_CLIENT_KEY: "test-posthog-key",
        PROVIDERS_CONFIGURED: [],
        AUTH_URL: "https://auth.example.com",
        FEATURE_FLAGS: {
          ENABLE_BILLING: false,
          HIDE_LLM_SETTINGS: false,
          ENABLE_JIRA: false,
          ENABLE_JIRA_DC: false,
          ENABLE_LINEAR: false,
        },
      });

      render(<RouterStub initialEntries={["/login"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(
          screen.getByText("AUTH$NO_PROVIDERS_CONFIGURED"),
        ).toBeInTheDocument();
      });
    });
  });

  describe("OAuth Flow", () => {
    it("should redirect to GitHub auth URL when GitHub button is clicked", async () => {
      const user = userEvent.setup();
      const mockUrl = "https://github.com/login/oauth/authorize";

      render(<RouterStub initialEntries={["/login"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "GITHUB$CONNECT_TO_GITHUB" }),
        ).toBeInTheDocument();
      });

      const githubButton = screen.getByRole("button", {
        name: "GITHUB$CONNECT_TO_GITHUB",
      });
      await user.click(githubButton);

      expect(window.location.href).toBe(mockUrl);
    });

    it("should redirect to GitLab auth URL when GitLab button is clicked", async () => {
      const user = userEvent.setup();

      render(<RouterStub initialEntries={["/login"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "GITLAB$CONNECT_TO_GITLAB" }),
        ).toBeInTheDocument();
      });

      const gitlabButton = screen.getByRole("button", {
        name: "GITLAB$CONNECT_TO_GITLAB",
      });
      await user.click(gitlabButton);

      expect(window.location.href).toBe("https://gitlab.com/oauth/authorize");
    });

    it("should redirect to Bitbucket auth URL when Bitbucket button is clicked", async () => {
      const user = userEvent.setup();

      render(<RouterStub initialEntries={["/login"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(screen.getByTestId("login-content")).toBeInTheDocument();
      });

      await waitFor(() => {
        expect(
          screen.getByRole("button", {
            name: /BITBUCKET\$CONNECT_TO_BITBUCKET/i,
          }),
        ).toBeInTheDocument();
      });

      const bitbucketButton = screen.getByRole("button", {
        name: /BITBUCKET\$CONNECT_TO_BITBUCKET/i,
      });
      await user.click(bitbucketButton);

      expect(window.location.href).toBe(
        "https://bitbucket.org/site/oauth2/authorize",
      );
    });
  });

  describe("Redirects", () => {
    it("should redirect authenticated users to home", async () => {
      vi.spyOn(AuthService, "authenticate").mockResolvedValue(true);

      render(<RouterStub initialEntries={["/login"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(
        () => {
          expect(screen.queryByTestId("login-page")).not.toBeInTheDocument();
        },
        { timeout: 2000 },
      );
    });

    it("should redirect authenticated users to returnTo destination", async () => {
      vi.spyOn(AuthService, "authenticate").mockResolvedValue(true);

      render(<RouterStub initialEntries={["/login?returnTo=/settings"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(
        () => {
          expect(screen.queryByTestId("login-page")).not.toBeInTheDocument();
        },
        { timeout: 2000 },
      );
    });

    it("should redirect OSS mode users to home", async () => {
      vi.spyOn(OptionService, "getConfig").mockResolvedValue({
        APP_MODE: "oss",
        GITHUB_CLIENT_ID: "test-client-id",
        POSTHOG_CLIENT_KEY: "test-posthog-key",
        FEATURE_FLAGS: {
          ENABLE_BILLING: false,
          HIDE_LLM_SETTINGS: false,
          ENABLE_JIRA: false,
          ENABLE_JIRA_DC: false,
          ENABLE_LINEAR: false,
        },
      });

      render(<RouterStub initialEntries={["/login"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(
        () => {
          expect(screen.queryByTestId("login-page")).not.toBeInTheDocument();
        },
        { timeout: 2000 },
      );
    });
  });

  describe("Email Verification", () => {
    it("should display email verified message when emailVerified is true", async () => {
      useEmailVerificationMock.mockReturnValue({
        emailVerified: true,
        hasDuplicatedEmail: false,
        emailVerificationModalOpen: false,
        setEmailVerificationModalOpen: vi.fn(),
      });

      render(<RouterStub initialEntries={["/login"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(
          screen.getByText("AUTH$EMAIL_VERIFIED_PLEASE_LOGIN"),
        ).toBeInTheDocument();
      });
    });

    it("should display duplicate email error when hasDuplicatedEmail is true", async () => {
      useEmailVerificationMock.mockReturnValue({
        emailVerified: false,
        hasDuplicatedEmail: true,
        emailVerificationModalOpen: false,
        setEmailVerificationModalOpen: vi.fn(),
      });

      render(<RouterStub initialEntries={["/login"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(
          screen.getByText("AUTH$DUPLICATE_EMAIL_ERROR"),
        ).toBeInTheDocument();
      });
    });
  });

  describe("Loading States", () => {
    it("should show loading spinner while checking auth", async () => {
      vi.spyOn(AuthService, "authenticate").mockImplementation(
        () => new Promise(() => {}),
      );

      render(<RouterStub initialEntries={["/login"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        const spinner = document.querySelector(".animate-spin");
        expect(spinner).toBeInTheDocument();
      });
    });

    it("should show loading spinner while loading config", async () => {
      vi.spyOn(OptionService, "getConfig").mockImplementation(
        () => new Promise(() => {}),
      );

      render(<RouterStub initialEntries={["/login"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        const spinner = document.querySelector(".animate-spin");
        expect(spinner).toBeInTheDocument();
      });
    });
  });

  describe("Terms and Privacy", () => {
    it("should display Terms and Privacy notice", async () => {
      render(<RouterStub initialEntries={["/login"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(
          screen.getByTestId("terms-and-privacy-notice"),
        ).toBeInTheDocument();
      });
    });
  });
});
