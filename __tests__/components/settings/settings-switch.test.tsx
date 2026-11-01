import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";

describe("SettingsSwitch", () => {
  // @spec SSW-001 — Expose checked state and native keyboard interaction.
  it("exposes its label and checked state and toggles with the keyboard", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    render(
      <SettingsSwitch onToggle={onToggle}>Sound Notifications</SettingsSwitch>,
    );
    const checkbox = screen.getByRole("checkbox", {
      name: "Sound Notifications",
    });

    await user.tab();
    expect(checkbox).toHaveFocus();
    await user.keyboard(" ");

    expect(checkbox).toBeChecked();
    expect(onToggle).toHaveBeenLastCalledWith(true);
    await user.keyboard(" ");
    expect(checkbox).not.toBeChecked();
    expect(onToggle).toHaveBeenLastCalledWith(false);
  });

  // @spec SSW-001 — Disabled controls skip Tab and retain controlled state.
  it("skips a disabled switch and keeps its controlled state", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    render(
      <>
        <SettingsSwitch isDisabled isToggled onToggle={onToggle}>
          Disabled Switch
        </SettingsSwitch>
        <SettingsSwitch togglePosition="right">Enabled Switch</SettingsSwitch>
      </>,
    );

    await user.tab();
    expect(
      screen.getByRole("checkbox", { name: "Enabled Switch" }),
    ).toHaveFocus();
    await user.click(screen.getByText("Disabled Switch"));

    expect(
      screen.getByRole("checkbox", { name: "Disabled Switch" }),
    ).toBeChecked();
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
