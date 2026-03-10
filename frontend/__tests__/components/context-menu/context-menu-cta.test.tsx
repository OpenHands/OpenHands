import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { ContextMenuCTA } from "#/components/features/context-menu/context-menu-cta";
import { renderWithProviders } from "../../../test-utils";

describe("ContextMenuCTA", () => {
  it("should render the CTA component", () => {
    renderWithProviders(<ContextMenuCTA />);

    expect(screen.getByText("CTA$ENTERPRISE_TITLE")).toBeInTheDocument();
    expect(screen.getByText("CTA$ENTERPRISE_DESCRIPTION")).toBeInTheDocument();
    expect(screen.getByText("CTA$LEARN_MORE")).toBeInTheDocument();
  });

  it("should render the Learn more button with correct link", () => {
    renderWithProviders(<ContextMenuCTA />);

    const learnMoreButton = screen.getByRole("button", {
      name: "CTA$LEARN_MORE",
    });
    expect(learnMoreButton).toBeInTheDocument();

    const link = learnMoreButton.closest("a");
    expect(link).toHaveAttribute("href", "https://openhands.dev/enterprise/");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("should have correct container styling", () => {
    renderWithProviders(<ContextMenuCTA />);

    const ctaContainer = screen.getByTestId("context-menu-cta");
    expect(ctaContainer).toHaveClass("w-[286px]");
    expect(ctaContainer).toHaveClass("h-[449px]");
    expect(ctaContainer).toHaveClass("rounded-[6px]");
    expect(ctaContainer).toHaveClass("cta-card-gradient");
  });

  it("should have correct inner content container", () => {
    renderWithProviders(<ContextMenuCTA />);

    const contentContainer = screen.getByTestId("context-menu-cta-content");
    expect(contentContainer).toBeInTheDocument();
    expect(contentContainer).toHaveClass("flex");
    expect(contentContainer).toHaveClass("flex-col");
    expect(contentContainer).toHaveClass("gap-[11px]");
    expect(contentContainer).toHaveClass("p-[25px]");
  });

  it("should render the stacked icon", () => {
    renderWithProviders(<ContextMenuCTA />);

    const contentContainer = screen.getByTestId("context-menu-cta-content");
    const icon = contentContainer.querySelector("svg");
    expect(icon).toBeInTheDocument();
    expect(icon).toHaveAttribute("width", "40");
    expect(icon).toHaveAttribute("height", "40");
  });

  it("should have Learn more button with correct styling", () => {
    renderWithProviders(<ContextMenuCTA />);

    const learnMoreButton = screen.getByRole("button", {
      name: "CTA$LEARN_MORE",
    });
    expect(learnMoreButton).toHaveClass("h-[40px]");
    expect(learnMoreButton).toHaveClass("rounded-[4px]");
  });
});
