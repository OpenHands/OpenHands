import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, describe, it, expect } from "vitest";
import { ToggleSwitch } from "#/components/features/automations/toggle-switch";

describe("ToggleSwitch", () => {
  it("calls onToggle when clicked", async () => {
    const onToggle = vi.fn();
    const user = userEvent.setup();
    render(
      <ToggleSwitch enabled={false} label="Toggle test" onToggle={onToggle} />,
    );

    await user.click(screen.getByRole("switch"));

    expect(onToggle).toHaveBeenCalledOnce();
  });

  it("reflects enabled state via aria-checked", () => {
    render(<ToggleSwitch enabled label="Toggle test" onToggle={vi.fn()} />);

    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "true");
  });

  it("ignores pointer and keyboard activation when disabled", async () => {
    const onToggle = vi.fn();
    const user = userEvent.setup();
    render(
      <ToggleSwitch
        enabled={false}
        label="Toggle test"
        onToggle={onToggle}
        disabled
      />,
    );

    const toggle = screen.getByRole("switch");
    await user.click(toggle);
    toggle.focus();
    await user.keyboard(" ");

    expect(toggle).toBeDisabled();
    expect(toggle).not.toHaveFocus();
    expect(onToggle).not.toHaveBeenCalled();
  });

  it("stops event propagation on click", async () => {
    const parentClick = vi.fn();
    const user = userEvent.setup();
    render(
      <div onClick={parentClick} onKeyDown={parentClick} role="presentation">
        <ToggleSwitch enabled={false} label="Toggle test" onToggle={vi.fn()} />
      </div>,
    );

    await user.click(screen.getByRole("switch"));

    expect(parentClick).not.toHaveBeenCalled();
  });
});
