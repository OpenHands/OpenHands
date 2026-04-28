import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { LoginContent } from "#/components/features/auth/login-content";

vi.mock("#/hooks/use-auth-url", () => ({
  useAuthUrl: (config: {
    identityProvider: string;
    appMode: string | null;
    authUrl?: string;
  }) => {
    const urls: Record<string, string> = {
      gitlab: "https://gitlab.com/oauth/authorize",
      bitbucket: "https://bitbucket.org/site/oauth2/authorize",
      bitbucket_data_center:
        "https://bitbucket-dc.example.com/site/oauth2/authorize",
      enterprise_sso: "https://auth.example.com/realms/test/protocol/openid-connect/auth",
    };
    if (config.appMode === "saas") {
      return urls[config.identityProvider] || null;
    }
    return null;
  },
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackLoginButtonClick: vi.fn(),
  }),
}));

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({
    data: undefined,
  }),
}));

vi.mock("#/hooks/use-recaptcha", () => ({
  useRecaptcha: () => ({
    isReady: false,
    isLoading: false,
    error: null,
    executeRecaptcha: vi.fn().mockResolvedValue(null),
  }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

const mockUseAppMode = vi.fn(() => ({
  isOss: false,
  isSaas: true,
  isCloud: true,
  isSelfHosted: false,
  isEnterpriseSelfHosted: false,
  isEnterpriseCloud: true,
  appMode: "saas" as string | undefined,
  deploymentMode: "cloud" as string | undefined,
}));
vi.mock("#/hooks/use-app-mode", () => ({
  useAppMode: () => mockUseAppMode(),
}));

describe("LoginContent", () => {
  beforeEach(() => {
    vi.stubGlobal("location", { href: "" });
    // Reset mock to return SaaS Cloud (CTA enabled) by default
    mockUseAppMode.mockReturnValue({
      isOss: false,
      isSaas: true,
      isCloud: true,
      isSelfHosted: false,
      isEnterpriseSelfHosted: false,
      isEnterpriseCloud: true,
      appMode: "saas",
      deploymentMode: "cloud",
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("should render login content with heading", () => {
    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          providersConfigured={["github", "gitlab", "bitbucket"]}
        />
      </MemoryRouter>,
    );

    expect(screen.getByTestId("login-content")).toBeInTheDocument();
    expect(screen.getByText("AUTH$LETS_GET_STARTED")).toBeInTheDocument();
  });

  it("should display all configured provider buttons", () => {
    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          authUrl="https://auth.example.com"
          providersConfigured={["github", "gitlab", "bitbucket"]}
        />
      </MemoryRouter>,
    );

    expect(
      screen.getByRole("button", { name: "GITHUB$CONNECT_TO_GITHUB" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "GITLAB$CONNECT_TO_GITLAB" }),
    ).toBeInTheDocument();

    const bitbucketButton = screen.getByRole("button", {
      name: /BITBUCKET\$CONNECT_TO_BITBUCKET/i,
    });
    expect(bitbucketButton).toBeInTheDocument();
  });

  it("should only display configured providers", () => {
    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          providersConfigured={["github"]}
        />
      </MemoryRouter>,
    );

    expect(
      screen.getByRole("button", { name: "GITHUB$CONNECT_TO_GITHUB" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "GITLAB$CONNECT_TO_GITLAB" }),
    ).not.toBeInTheDocument();
  });

  it("should display Enterprise SSO button when configured", () => {
    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          authUrl="https://auth.example.com"
          providersConfigured={["enterprise_sso"]}
        />
      </MemoryRouter>,
    );

    expect(
      screen.getByRole("button", { name: /ENTERPRISE_SSO\$CONNECT_TO_ENTERPRISE_SSO/i }),
    ).toBeInTheDocument();
  });

  it("should display Enterprise SSO alongside other providers when all configured", () => {
    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          authUrl="https://auth.example.com"
          providersConfigured={["github", "gitlab", "bitbucket", "enterprise_sso"]}
        />
      </MemoryRouter>,
    );

    expect(
      screen.getByRole("button", { name: "GITHUB$CONNECT_TO_GITHUB" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "GITLAB$CONNECT_TO_GITLAB" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /BITBUCKET\$CONNECT_TO_BITBUCKET/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /ENTERPRISE_SSO\$CONNECT_TO_ENTERPRISE_SSO/i }),
    ).toBeInTheDocument();
  });

  it("should redirect to Enterprise SSO auth URL when Enterprise SSO button is clicked", async () => {
    const user = userEvent.setup();
    const mockUrl = "https://auth.example.com/realms/test/protocol/openid-connect/auth";

    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          authUrl="https://auth.example.com"
          providersConfigured={["enterprise_sso"]}
        />
      </MemoryRouter>,
    );

    const enterpriseSsoButton = screen.getByRole("button", {
      name: /ENTERPRISE_SSO\$CONNECT_TO_ENTERPRISE_SSO/i,
    });
    await user.click(enterpriseSsoButton);

    await waitFor(() => {
      expect(window.location.href).toContain(mockUrl);
    });
  });

  it("should display message when no providers are configured", () => {
    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          providersConfigured={[]}
        />
      </MemoryRouter>,
    );

    expect(
      screen.getByText("AUTH$NO_PROVIDERS_CONFIGURED"),
    ).toBeInTheDocument();
  });

  it("should redirect to GitHub auth URL when GitHub button is clicked", async () => {
    const user = userEvent.setup();
    const mockUrl = "https://github.com/login/oauth/authorize";

    render(
      <MemoryRouter>
        <LoginContent
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

    // Wait for async handleAuthRedirect to complete
    // The URL includes state parameter added by handleAuthRedirect
    await waitFor(() => {
      expect(window.location.href).toContain(mockUrl);
    });
  });

  it("should display email verified message when emailVerified is true", () => {
    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          providersConfigured={["github"]}
          emailVerified
        />
      </MemoryRouter>,
    );

    expect(
      screen.getByText("AUTH$EMAIL_VERIFIED_PLEASE_LOGIN"),
    ).toBeInTheDocument();
  });

  it("should display duplicate email error when hasDuplicatedEmail is true", () => {
    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          providersConfigured={["github"]}
          hasDuplicatedEmail
        />
      </MemoryRouter>,
    );

    expect(screen.getByText("AUTH$DUPLICATE_EMAIL_ERROR")).toBeInTheDocument();
  });

  it("should display Terms and Privacy notice", () => {
    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          providersConfigured={["github"]}
        />
      </MemoryRouter>,
    );

    expect(screen.getByTestId("terms-and-privacy-notice")).toBeInTheDocument();
  });

  it("should display the enterprise LoginCTA component when in SaaS Cloud mode", () => {
    mockUseAppMode.mockReturnValue({
      isOss: false,
      isSaas: true,
      isCloud: true,
      isSelfHosted: false,
      isEnterpriseSelfHosted: false,
      isEnterpriseCloud: true,
      appMode: "saas",
      deploymentMode: "cloud",
    });

    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          providersConfigured={["github"]}
        />
      </MemoryRouter>,
    );

    expect(screen.getByTestId("login-cta")).toBeInTheDocument();
  });

  it("should not display the enterprise LoginCTA component when in OSS mode", () => {
    mockUseAppMode.mockReturnValue({
      isOss: true,
      isSaas: false,
      isCloud: false,
      isSelfHosted: false,
      isEnterpriseSelfHosted: false,
      isEnterpriseCloud: false,
      appMode: "oss",
      deploymentMode: undefined,
    });

    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="oss"
          providersConfigured={["github"]}
        />
      </MemoryRouter>,
    );

    expect(screen.queryByTestId("login-cta")).not.toBeInTheDocument();
  });

  it("should not display the enterprise LoginCTA component when in SaaS Self-hosted mode", () => {
    mockUseAppMode.mockReturnValue({
      isOss: false,
      isSaas: true,
      isCloud: false,
      isSelfHosted: true,
      isEnterpriseSelfHosted: true,
      isEnterpriseCloud: false,
      appMode: "saas",
      deploymentMode: "self_hosted",
    });

    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          providersConfigured={["github"]}
        />
      </MemoryRouter>,
    );

    expect(screen.queryByTestId("login-cta")).not.toBeInTheDocument();
  });

  it("should display invitation pending message when hasInvitation is true", () => {
    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          providersConfigured={["github"]}
          hasInvitation
        />
      </MemoryRouter>,
    );

    expect(screen.getByText("AUTH$INVITATION_PENDING")).toBeInTheDocument();
  });

  it("should not display invitation pending message when hasInvitation is false", () => {
    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          providersConfigured={["github"]}
          hasInvitation={false}
        />
      </MemoryRouter>,
    );

    expect(
      screen.queryByText("AUTH$INVITATION_PENDING"),
    ).not.toBeInTheDocument();
  });

  it("should display Bitbucket signup disabled message when Bitbucket is configured", () => {
    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          providersConfigured={["github", "bitbucket"]}
        />
      </MemoryRouter>,
    );

    expect(
      screen.getByText("AUTH$BITBUCKET_SIGNUP_DISABLED"),
    ).toBeInTheDocument();
  });

  it("should not display Bitbucket signup disabled message when Bitbucket is not configured", () => {
    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/oauth/authorize"
          appMode="saas"
          providersConfigured={["github"]}
        />
      </MemoryRouter>,
    );

    expect(
      screen.queryByText("AUTH$BITBUCKET_SIGNUP_DISABLED"),
    ).not.toBeInTheDocument();
  });

  it("should call buildOAuthStateData when clicking auth button", async () => {
    const user = userEvent.setup();
    const mockBuildOAuthStateData = vi.fn((baseState) => ({
      ...baseState,
      invitation_token: "inv-test-token-12345",
    }));

    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/login/oauth/authorize"
          appMode="saas"
          providersConfigured={["github"]}
          buildOAuthStateData={mockBuildOAuthStateData}
        />
      </MemoryRouter>,
    );

    const githubButton = screen.getByRole("button", {
      name: "GITHUB$CONNECT_TO_GITHUB",
    });
    await user.click(githubButton);

    await waitFor(() => {
      expect(mockBuildOAuthStateData).toHaveBeenCalled();
      const callArg = mockBuildOAuthStateData.mock.calls[0][0];
      expect(callArg).toHaveProperty("redirect_url");
    });
  });

  it("should display Bitbucket Data Center button when configured", () => {
    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl={null}
          appMode="saas"
          providersConfigured={["bitbucket_data_center"]}
        />
      </MemoryRouter>,
    );

    expect(
      screen.getByRole("button", {
        name: /BITBUCKET_DATA_CENTER\$CONNECT_TO_BITBUCKET_DATA_CENTER/i,
      }),
    ).toBeInTheDocument();
  });

  it("should encode state with invitation token when buildOAuthStateData provides token", async () => {
    const user = userEvent.setup();
    const mockBuildOAuthStateData = vi.fn((baseState) => ({
      ...baseState,
      invitation_token: "inv-test-token-12345",
    }));

    render(
      <MemoryRouter>
        <LoginContent
          githubAuthUrl="https://github.com/login/oauth/authorize"
          appMode="saas"
          providersConfigured={["github"]}
          buildOAuthStateData={mockBuildOAuthStateData}
        />
      </MemoryRouter>,
    );

    const githubButton = screen.getByRole("button", {
      name: "GITHUB$CONNECT_TO_GITHUB",
    });
    await user.click(githubButton);

    await waitFor(() => {
      const redirectUrl = window.location.href;
      // The URL should contain an encoded state parameter
      expect(redirectUrl).toContain("state=");
      // Decode and verify the state contains invitation_token
      const url = new URL(redirectUrl);
      const state = url.searchParams.get("state");
      if (state) {
        const decodedState = JSON.parse(atob(state));
        expect(decodedState.invitation_token).toBe("inv-test-token-12345");
      }
    });
  });
});
