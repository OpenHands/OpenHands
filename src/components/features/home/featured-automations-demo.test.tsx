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
    expect(
      screen.getByText(
        "I reviewed the pull request and posted three suggestions. The security check completed without findings.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Open most recent conversation" }),
    ).toHaveAttribute("href", "/conversations/automation-pr-review-16182");
    expect(screen.getByRole("button", { name: "PR reviewer" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("shows the latest error when a failed automation is featured", () => {
    render(<FeaturedAutomationsDemo />);

    fireEvent.click(screen.getByRole("button", { name: "Issue triage" }));

    expect(
      screen.getByText(
        "Model provider rejected the repository lookup request: rate limit exceeded.",
      ),
    ).toBeInTheDocument();
  });

  it("links the final add control to Automations", () => {
    render(<FeaturedAutomationsDemo />);

    expect(
      screen.getByRole("link", { name: "Add or manage automations" }),
    ).toHaveAttribute("href", "/automations");
  });
});
