import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ContextMenuCTA } from "#/components/features/context-menu/context-menu-cta";

vi.mock("#/hooks/use-client-analytics", () => ({
  useClientAnalytics: () => ({
    trackSaasSelfhostedInquiry: vi.fn(),
    trackEnterpriseLeadFormSubmitted: vi.fn(),
  }),
}));

describe("ContextMenuCTA", () => {
  it("should render the CTA component", () => {
    render(<ContextMenuCTA />);

    expect(screen.getByText("CTA$ENTERPRISE_TITLE")).toBeInTheDocument();
    expect(screen.getByText("CTA$ENTERPRISE_DESCRIPTION")).toBeInTheDocument();
    expect(screen.getByText("CTA$LEARN_MORE")).toBeInTheDocument();
  });

  it("should render Learn More as a link with correct href and target", () => {
    render(<ContextMenuCTA />);

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

  it("should render the stacked icon", () => {
    render(<ContextMenuCTA />);

    const contentContainer = screen.getByTestId("context-menu-cta-content");
    const icon = contentContainer.querySelector("svg");
    expect(icon).toBeInTheDocument();
    expect(icon).toHaveAttribute("width", "40");
    expect(icon).toHaveAttribute("height", "40");
  });
});
