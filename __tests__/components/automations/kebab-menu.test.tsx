import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { KebabMenu } from "#/components/features/automations/kebab-menu";

const ITEMS = ["Run now", "View", "Export", "Edit", "Turn on", "Delete"].map(
  (label) => ({ label, icon: <span />, onClick: vi.fn() }),
);

/** Place the trigger at a fixed viewport position before the menu opens. */
function openMenuWithTriggerAt(top: number, bottom: number) {
  const trigger = screen.getByRole("button", {
    name: "AUTOMATIONS$ACTIONS_MENU",
  });
  vi.spyOn(trigger, "getBoundingClientRect").mockReturnValue({
    top,
    bottom,
    left: 900,
    right: 1000,
    width: 100,
    height: bottom - top,
    x: 900,
    y: top,
    toJSON: () => ({}),
  } as DOMRect);
  fireEvent.click(trigger);
  return document.querySelector<HTMLElement>('div[style*="position: fixed"]');
}

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

  it("anchors below the trigger when the menu fits in the viewport", () => {
    render(<KebabMenu items={ITEMS} />);

    const portal = openMenuWithTriggerAt(68, 100);

    expect(portal?.style.top).toBe("104px");
    expect(portal?.style.bottom).toBe("");
  });

  it("flips above the trigger when the menu would clip at the viewport bottom", () => {
    render(<KebabMenu items={ITEMS} />);

    // window.innerHeight is 768 in jsdom; 700 + 4 + (6 * 36) overflows it.
    const portal = openMenuWithTriggerAt(668, 700);

    expect(portal?.style.bottom).toBe("104px");
    expect(portal?.style.top).toBe("");
  });
});
