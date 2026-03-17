import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { LoginCTA } from "#/components/features/auth/login-cta";

// Mock useTracking hook
const mockTrackSaasSelfhostedInquiry = vi.fn();
vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackSaasSelfhostedInquiry: mockTrackSaasSelfhostedInquiry,
  }),
}));

describe("LoginCTA", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render enterprise CTA with title and description", () => {
    render(<LoginCTA />);

    expect(screen.getByTestId("login-cta")).toBeInTheDocument();
    expect(screen.getByText("CTA$ENTERPRISE")).toBeInTheDocument();
    expect(screen.getByText("CTA$ENTERPRISE_DEPLOY")).toBeInTheDocument();
  });

  it("should render all enterprise feature list items", () => {
    render(<LoginCTA />);

    expect(screen.getByText("CTA$FEATURE_ON_PREMISES")).toBeInTheDocument();
    expect(screen.getByText("CTA$FEATURE_DATA_CONTROL")).toBeInTheDocument();
    expect(screen.getByText("CTA$FEATURE_COMPLIANCE")).toBeInTheDocument();
    expect(screen.getByText("CTA$FEATURE_SUPPORT")).toBeInTheDocument();
  });

  it("should render Learn More as a link with correct href and target", () => {
    render(<LoginCTA />);

    const learnMoreLink = screen.getByRole("link", {
      name: "CTA$LEARN_MORE",
    });
    expect(learnMoreLink).toHaveAttribute(
      "href",
      "https://openhands.dev/enterprise/",
    );
    expect(learnMoreLink).toHaveAttribute("target", "_blank");
    expect(learnMoreLink).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("should call trackSaasSelfhostedInquiry with location 'login_page' when Learn More is clicked", async () => {
    const user = userEvent.setup();
    render(<LoginCTA />);

    const learnMoreLink = screen.getByRole("link", {
      name: "CTA$LEARN_MORE",
    });
    await user.click(learnMoreLink);

    expect(mockTrackSaasSelfhostedInquiry).toHaveBeenCalledWith({
      location: "login_page",
    });
  });
});
