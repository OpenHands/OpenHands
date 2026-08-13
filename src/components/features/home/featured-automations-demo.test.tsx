import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router";
import { FeaturedAutomationsDemo } from "./featured-automations-demo";

describe("FeaturedAutomationsDemo", () => {
  it("adds a selected automation to the featured dashboard", () => {
    render(
      <MemoryRouter>
        <FeaturedAutomationsDemo />
      </MemoryRouter>,
    );

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

  it("removes an automation from the featured dashboard on a second click", () => {
    render(
      <MemoryRouter>
        <FeaturedAutomationsDemo />
      </MemoryRouter>,
    );

    const button = screen.getByRole("button", { name: "PR reviewer" });
    fireEvent.click(button);
    expect(button).toHaveAttribute("aria-pressed", "true");
    expect(
      screen.getByText("3 suggestions posted · 1 security check passed"),
    ).toBeInTheDocument();

    fireEvent.click(button);
    expect(button).toHaveAttribute("aria-pressed", "false");
    expect(
      screen.queryByText("3 suggestions posted · 1 security check passed"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText(
        "I reviewed the pull request and posted three suggestions. The security check completed without findings.",
      ),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: "Open most recent conversation" }),
    ).not.toBeInTheDocument();
  });

  it("shows the latest error when a failed automation is featured", () => {
    render(
      <MemoryRouter>
        <FeaturedAutomationsDemo />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Issue triage" }));

    expect(
      screen.getByText(
        "Model provider rejected the repository lookup request: rate limit exceeded.",
      ),
    ).toBeInTheDocument();
  });

  it("links the final add control to Automations", () => {
    render(
      <MemoryRouter>
        <FeaturedAutomationsDemo />
      </MemoryRouter>,
    );

    expect(
      screen.getByRole("link", { name: "Add or manage automations" }),
    ).toHaveAttribute("href", "/automations");
  });
});
