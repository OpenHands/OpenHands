import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { NotFoundState } from "#/components/features/automations/detail/not-found-state";

describe("NotFoundState", () => {
  it("renders not found state with link back to automations list", () => {
    const { container } = render(<NotFoundState />);

    // Check for the main container
    expect(container.querySelector("div")).toBeInTheDocument();
    // Check for the link to go back
    const link = screen.getByRole("link");
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "/automations");
  });
});
