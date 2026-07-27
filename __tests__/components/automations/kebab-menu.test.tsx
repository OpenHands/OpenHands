import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { KebabMenu } from "#/components/features/automations/kebab-menu";

describe("KebabMenu", () => {
  it("opens the menu and invokes an item's onClick when selected", async () => {
    const onClick = vi.fn();
    render(
      <KebabMenu items={[{ label: "Delete", icon: <span />, onClick }]} />,
    );

    // Open the menu, then select the item (rendered via KebabMenuItemContent).
    fireEvent.click(
      screen.getByRole("button", { name: "AUTOMATIONS$ACTIONS_MENU" }),
    );
    fireEvent.click(await screen.findByText("Delete"));

    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
