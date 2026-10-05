import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";

describe("SettingsSwitch", () => {
  it.each(["left", "right"] as const)(
    "exposes the %s switch to keyboard users and label activation",
    async (togglePosition) => {
      const user = userEvent.setup();
      const onToggle = vi.fn();
      render(
        <SettingsSwitch onToggle={onToggle} togglePosition={togglePosition}>
          Keyboard setting
        </SettingsSwitch>,
      );

      const checkbox = screen.getByRole("checkbox", {
        name: "Keyboard setting",
      });
      await user.tab();
      expect(checkbox).toHaveFocus();

      await user.keyboard(" ");
      expect(checkbox).toBeChecked();
      expect(onToggle).toHaveBeenNthCalledWith(1, true);

      await user.keyboard(" ");
      expect(checkbox).not.toBeChecked();
      expect(onToggle).toHaveBeenNthCalledWith(2, false);

      await user.click(screen.getByText("Keyboard setting"));
      expect(checkbox).toBeChecked();
      expect(onToggle).toHaveBeenNthCalledWith(3, true);
      expect(onToggle).toHaveBeenCalledTimes(3);
    },
  );

  it("preserves a controlled value until its owner updates it", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    const { rerender } = render(
      <SettingsSwitch isToggled={false} onToggle={onToggle}>
        Controlled setting
      </SettingsSwitch>,
    );
    const checkbox = screen.getByRole("checkbox", {
      name: "Controlled setting",
    });

    await user.tab();
    await user.keyboard(" ");
    expect(onToggle).toHaveBeenCalledWith(true);
    expect(checkbox).not.toBeChecked();

    rerender(
      <SettingsSwitch isToggled onToggle={onToggle}>
        Controlled setting
      </SettingsSwitch>,
    );
    expect(checkbox).toBeChecked();
    await user.keyboard(" ");
    expect(onToggle).toHaveBeenLastCalledWith(false);
    expect(checkbox).toBeChecked();
  });

  it("keeps disabled switches exposed but skips them during keyboard navigation", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    render(
      <>
        <SettingsSwitch isDisabled defaultIsToggled onToggle={onToggle}>
          Disabled setting
        </SettingsSwitch>
        <SettingsSwitch>Available setting</SettingsSwitch>
      </>,
    );
    const disabled = screen.getByRole("checkbox", {
      name: "Disabled setting",
    });
    expect(disabled).toBeDisabled();
    await user.tab();
    expect(
      screen.getByRole("checkbox", { name: "Available setting" }),
    ).toHaveFocus();
    await user.keyboard(" ");
    await user.click(screen.getByText("Disabled setting"));
    expect(disabled).toBeChecked();
    expect(onToggle).not.toHaveBeenCalled();
  });

  it("should call the onChange handler when the input is clicked", async () => {
    const user = userEvent.setup();
    const onToggleMock = vi.fn();
    render(
      <SettingsSwitch testId="test-switch" onToggle={onToggleMock}>
        Test Switch
      </SettingsSwitch>,
    );

    const switchInput = screen.getByTestId("test-switch");

    await user.click(switchInput);
    expect(onToggleMock).toHaveBeenCalledWith(true);

    await user.click(switchInput);
    expect(onToggleMock).toHaveBeenCalledWith(false);
  });

  it("should render a beta tag if isBeta is true", () => {
    const { rerender } = render(
      <SettingsSwitch testId="test-switch" onToggle={vi.fn()} isBeta={false}>
        Test Switch
      </SettingsSwitch>,
    );

    expect(screen.queryByText(/beta/i)).not.toBeInTheDocument();

    rerender(
      <SettingsSwitch testId="test-switch" onToggle={vi.fn()} isBeta>
        Test Switch
      </SettingsSwitch>,
    );

    expect(screen.getByText(/beta/i)).toBeInTheDocument();
  });

  it("should be able to set a default toggle state", async () => {
    const user = userEvent.setup();
    const onToggleMock = vi.fn();
    render(
      <SettingsSwitch
        testId="test-switch"
        onToggle={onToggleMock}
        defaultIsToggled
      >
        Test Switch
      </SettingsSwitch>,
    );

    expect(screen.getByTestId("test-switch")).toBeChecked();

    const switchInput = screen.getByTestId("test-switch");
    await user.click(switchInput);
    expect(onToggleMock).toHaveBeenCalledWith(false);

    expect(screen.getByTestId("test-switch")).not.toBeChecked();
  });
});
