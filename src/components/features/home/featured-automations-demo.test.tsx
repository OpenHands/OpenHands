import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FeaturedAutomationsDemo } from "./featured-automations-demo";

describe("FeaturedAutomationsDemo", () => {
  it("adds a selected automation to the featured dashboard", () => {
    render(<FeaturedAutomationsDemo />);

    fireEvent.click(screen.getByRole("button", { name: "PR reviewer" }));

    expect(
      screen.getByText("3 suggestions posted · 1 security check passed"),
    ).toBeInTheDocument();
  });
});
