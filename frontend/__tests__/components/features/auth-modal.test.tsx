import { render, screen } from "@testing-library/react";
import { it, describe, expect, vi, beforeEach, afterEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { AuthModal } from "#/components/features/waitlist/auth-modal";

// Mock the useAuthUrl hook
vi.mock("#/hooks/use-auth-url", () => ({
  useAuthUrl: () => "https://gitlab.com/oauth/authorize",
}));

// Mock the useTracking hook
vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackLoginButtonClick: vi.fn(),
  }),
}));

describe("AuthModal", () => {
  beforeEach(() => {
    vi.stubGlobal("location", { href: "" });
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
});
