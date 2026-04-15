import { createEvent, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AutomationsButton } from "#/components/shared/buttons/automations-button";

describe("AutomationsButton", () => {
  it("should render a link to /automations", () => {
    render(<AutomationsButton />);

    const link = screen.getByTestId("automations-button");
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "/automations");
  });

  it("should be focusable and accessible when enabled", () => {
    render(<AutomationsButton />);

    const link = screen.getByTestId("automations-button");
    expect(link).toHaveAttribute("tabIndex", "0");
    expect(link).toHaveAttribute("aria-label", "SIDEBAR$AUTOMATIONS");
  });

  it("should prevent navigation and remove from tab order when disabled", () => {
    render(<AutomationsButton disabled />);

    const link = screen.getByTestId("automations-button");
    expect(link).toHaveAttribute("tabIndex", "-1");

    const clickEvent = createEvent.click(link);
    fireEvent(link, clickEvent);
    expect(clickEvent.defaultPrevented).toBe(true);
  });
});
