import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AutomationViewToggle } from "#/components/features/automations/automation-view-toggle";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

describe("AutomationViewToggle", () => {
  it("opens a menu from the icon trigger and switches to list view", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(<AutomationViewToggle view="grid" onChange={onChange} />);

    const trigger = screen.getByTestId("automations-view-toggle");
    expect(trigger).toHaveClass("size-9");
    expect(trigger).toHaveAttribute("aria-haspopup", "menu");

    await user.click(trigger);
    await user.click(screen.getByTestId("automations-view-toggle-list"));

    expect(onChange).toHaveBeenCalledWith("list");
  });

  it("does not open the menu or fire onChange when disabled", async () => {
    // Arrange
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <AutomationViewToggle view="grid" onChange={onChange} disabled />,
    );
    const trigger = screen.getByTestId("automations-view-toggle");

    // Act — try to open the menu
    await user.click(trigger);

    // Assert — menu items never render and onChange stays untouched
    expect(trigger).toBeDisabled();
    expect(
      screen.queryByTestId("automations-view-toggle-list"),
    ).not.toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });
});
