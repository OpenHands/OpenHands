import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { createRoutesStub } from "react-router";
import { LoginCTA } from "#/components/features/auth/login-cta";

vi.mock("#/hooks/use-client-analytics", () => ({
  useClientAnalytics: () => ({
    trackEnterpriseCTAClicked: vi.fn(),
    trackEnterpriseLeadFormSubmitted: vi.fn(),
  }),
}));

describe("LoginCTA", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  const renderWithRouter = (source?: "login_page" | "device_verify") => {
    const Stub = createRoutesStub([
      {
        path: "/",
        Component: () => <LoginCTA source={source} />,
      },
      {
        path: "/information-request",
        Component: () => <div data-testid="information-request-page" />,
      },
    ]);

    return render(<Stub initialEntries={["/"]} />);
  };

  it("should render enterprise CTA with title and description", () => {
    renderWithRouter();

    expect(screen.getByTestId("login-cta")).toBeInTheDocument();
    expect(screen.getByText("CTA$ENTERPRISE")).toBeInTheDocument();
    expect(screen.getByText("CTA$ENTERPRISE_DEPLOY")).toBeInTheDocument();
  });

  it("should render all enterprise feature list items", () => {
    renderWithRouter();

    expect(screen.getByText("CTA$FEATURE_ON_PREMISES")).toBeInTheDocument();
    expect(screen.getByText("CTA$FEATURE_DATA_CONTROL")).toBeInTheDocument();
    expect(screen.getByText("CTA$FEATURE_COMPLIANCE")).toBeInTheDocument();
    expect(screen.getByText("CTA$FEATURE_SUPPORT")).toBeInTheDocument();
  });

  it("should navigate to information request page when Learn More is clicked", async () => {
    const user = userEvent.setup();
    renderWithRouter();

    const learnMoreLink = screen.getByRole("link", {
      name: "CTA$LEARN_MORE",
    });
    await user.click(learnMoreLink);

    expect(screen.getByTestId("information-request-page")).toBeInTheDocument();
  });

  it("should render Learn More as a link for Open in New Tab support", () => {
    renderWithRouter();

    const learnMoreLink = screen.getByRole("link", {
      name: "CTA$LEARN_MORE",
    });
    expect(learnMoreLink).toHaveAttribute(
      "href",
      "/information-request",
    );
  });

  it("should render external enterprise URL in device verify mode", () => {
    renderWithRouter("device_verify");

    const learnMoreLink = screen.getByRole("link", {
      name: "CTA$LEARN_MORE",
    });
    expect(learnMoreLink).toHaveAttribute(
      "href",
      "https://openhands.dev/enterprise",
    );
    expect(learnMoreLink).toHaveAttribute("target", "_blank");
    expect(learnMoreLink).toHaveAttribute("rel", "noopener noreferrer");
  });

});
