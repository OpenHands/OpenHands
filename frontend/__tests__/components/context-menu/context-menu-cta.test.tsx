import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ContextMenuCTA } from "#/components/features/context-menu/context-menu-cta";
import { renderWithProviders } from "../../../test-utils";

describe("ContextMenuCTA", () => {
  it("should render the CTA component", () => {
    renderWithProviders(<ContextMenuCTA />);

    expect(screen.getByText("CTA$ENTERPRISE_TITLE")).toBeInTheDocument();
    expect(screen.getByText("CTA$ENTERPRISE_DESCRIPTION")).toBeInTheDocument();
    expect(screen.getByText("CTA$LEARN_MORE")).toBeInTheDocument();
  });

  it("should render the Learn more button", () => {
    renderWithProviders(<ContextMenuCTA />);

    const learnMoreButton = screen.getByRole("button", { name: "CTA$LEARN_MORE" });
    expect(learnMoreButton).toBeInTheDocument();
  });

  it("should have correct container dimensions via className", () => {
    const { container } = renderWithProviders(<ContextMenuCTA />);

    const ctaContainer = container.firstChild as HTMLElement;
    expect(ctaContainer).toHaveClass("w-[286px]");
    expect(ctaContainer).toHaveClass("h-[449px]");
    expect(ctaContainer).toHaveClass("rounded-[6px]");
  });

  it("should have correct inner container dimensions", () => {
    const { container } = renderWithProviders(<ContextMenuCTA />);

    const innerContainer = container.querySelector(".w-\\[236px\\].h-\\[246px\\]");
    expect(innerContainer).toBeInTheDocument();
    expect(innerContainer).toHaveClass("gap-[11px]");
    expect(innerContainer).toHaveClass("mt-[175px]");
    expect(innerContainer).toHaveClass("ml-[25px]");
  });

  it("should have an icon placeholder with correct dimensions", () => {
    const { container } = renderWithProviders(<ContextMenuCTA />);

    const iconPlaceholder = container.querySelector(".w-\\[40px\\].h-\\[40px\\]");
    expect(iconPlaceholder).toBeInTheDocument();
  });

  it("should have Learn more button with correct dimensions", () => {
    renderWithProviders(<ContextMenuCTA />);

    const learnMoreButton = screen.getByRole("button", { name: "CTA$LEARN_MORE" });
    expect(learnMoreButton).toHaveClass("w-[111px]");
    expect(learnMoreButton).toHaveClass("h-[40px]");
    expect(learnMoreButton).toHaveClass("rounded-[4px]");
  });

  it("should have button container with correct styling", () => {
    const { container } = renderWithProviders(<ContextMenuCTA />);

    const buttonContainer = container.querySelector(".w-\\[236px\\].h-\\[40px\\].flex.justify-start");
    expect(buttonContainer).toBeInTheDocument();
  });

  it("should apply gradient background style", () => {
    const { container } = renderWithProviders(<ContextMenuCTA />);

    const ctaContainer = container.firstChild as HTMLElement;
    expect(ctaContainer.style.background).toContain("linear-gradient");
    expect(ctaContainer.style.background).toContain("radial-gradient");
  });
});
