import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { ContextMenuCTA } from "#/components/features/context-menu/context-menu-cta";

// Mock useTracking hook
const mockTrackSaasSelfhostedInquiry = vi.fn();
vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackSaasSelfhostedInquiry: mockTrackSaasSelfhostedInquiry,
  }),
}));

describe("ContextMenuCTA", () => {
  it("should render the CTA component", () => {
    render(<ContextMenuCTA />);

    expect(screen.getByText("CTA$ENTERPRISE_TITLE")).toBeInTheDocument();
    expect(screen.getByText("CTA$ENTERPRISE_DESCRIPTION")).toBeInTheDocument();
    expect(screen.getByText("CTA$LEARN_MORE")).toBeInTheDocument();
  });

  it("should call trackSaasSelfhostedInquiry with location 'context_menu' when Learn More is clicked", async () => {
    const user = userEvent.setup();
    render(<ContextMenuCTA />);

    const learnMoreButton = screen.getByRole("button", {
      name: "CTA$LEARN_MORE",
    });
    await user.click(learnMoreButton);

    expect(mockTrackSaasSelfhostedInquiry).toHaveBeenCalledWith({
      location: "context_menu",
    });
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
