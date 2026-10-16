import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { BaseModalDescription } from "#/components/shared/modals/confirmation-modals/base-modal";

describe("BaseModalDescription", () => {
  it("renders description text with the theme-aware text-modal-description class", () => {
    render(
      <BaseModalDescription description="Are you sure you want to delete this conversation?" />,
    );

    const description = screen.getByText(
      "Are you sure you want to delete this conversation?",
    );
    expect(description).toBeInTheDocument();
    expect(description).toHaveClass("text-sm", "text-modal-description");
  });

  it("renders children when provided", () => {
    render(
      <BaseModalDescription>
        <span>Custom description content</span>
      </BaseModalDescription>,
    );

    const child = screen.getByText("Custom description content");
    expect(child).toBeInTheDocument();
    expect(child.parentElement).toHaveClass("text-modal-description");
  });
});
