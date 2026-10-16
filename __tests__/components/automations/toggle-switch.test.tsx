import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, describe, it, expect } from "vitest";
import { ToggleSwitch } from "#/components/features/automations/toggle-switch";
import { ToggleSwitchVisual } from "#/ui/toggle-switch";

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

  describe("ToggleSwitchVisual RTL adaptation", () => {
    it("renders thumb with mirrored translation classes for md size in checked and unchecked states", () => {
      const { container: uncheckedContainer } = render(
        <ToggleSwitchVisual enabled={false} size="md" />,
      );
      const uncheckedThumb = uncheckedContainer.querySelector("span > span");
      expect(uncheckedThumb).toHaveClass("translate-x-0.5");
      expect(uncheckedThumb).toHaveClass("rtl:-translate-x-0.5");

      const { container: checkedContainer } = render(
        <ToggleSwitchVisual enabled size="md" />,
      );
      const checkedThumb = checkedContainer.querySelector("span > span");
      expect(checkedThumb).toHaveClass("translate-x-5.25");
      expect(checkedThumb).toHaveClass("rtl:-translate-x-5.25");
    });

    it("renders thumb with mirrored translation classes for sm (compact) size in checked and unchecked states", () => {
      const { container: uncheckedContainer } = render(
        <ToggleSwitchVisual enabled={false} size="sm" />,
      );
      const uncheckedThumb = uncheckedContainer.querySelector("span > span");
      expect(uncheckedThumb).toHaveClass("translate-x-0");
      expect(uncheckedThumb).toHaveClass("rtl:-translate-x-0");

      const { container: checkedContainer } = render(
        <ToggleSwitchVisual enabled size="sm" />,
      );
      const checkedThumb = checkedContainer.querySelector("span > span");
      expect(checkedThumb).toHaveClass("translate-x-2.5");
      expect(checkedThumb).toHaveClass("rtl:-translate-x-2.5");
    });

    it("supports dir='rtl' wrapper and preserves track and thumb styling", () => {
      const { container } = render(
        <div dir="rtl">
          <ToggleSwitchVisual enabled size="md" />
        </div>,
      );
      const thumb = container.querySelector("span > span");
      expect(thumb).toHaveClass("translate-x-5.25");
      expect(thumb).toHaveClass("rtl:-translate-x-5.25");
    });
  });
});
