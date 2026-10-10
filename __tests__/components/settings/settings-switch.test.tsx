import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";

const SWITCH_LABEL = "Test Switch";

describe("SettingsSwitch", () => {
  it("should call the onChange handler when the input is clicked", async () => {
    const user = userEvent.setup();
    const onToggleMock = vi.fn();
    render(
      <SettingsSwitch testId="test-switch" onToggle={onToggleMock}>
        {SWITCH_LABEL}
      </SettingsSwitch>,
    );

    const switchInput = screen.getByTestId("test-switch");

    await user.click(switchInput);
    expect(onToggleMock).toHaveBeenCalledWith(true);

    await user.click(switchInput);
    expect(onToggleMock).toHaveBeenCalledWith(false);
  });

  it("is keyboard-focusable and toggles with Space", async () => {
    // Arrange
    const user = userEvent.setup();
    const onToggleMock = vi.fn();
    render(
      <SettingsSwitch testId="test-switch" onToggle={onToggleMock}>
        {SWITCH_LABEL}
      </SettingsSwitch>,
    );

    // Act
    await user.tab();
    const switchInput = screen.getByRole("checkbox", { name: SWITCH_LABEL });
    await user.keyboard("[Space]");

    // Assert
    expect(switchInput).toHaveFocus();
    expect(switchInput).toBeChecked();
    expect(onToggleMock).toHaveBeenCalledWith(true);
  });

  it("keeps disabled switches out of the tab order", async () => {
    // Arrange
    const user = userEvent.setup();
    const onToggleMock = vi.fn();
    render(
      <SettingsSwitch testId="test-switch" onToggle={onToggleMock} isDisabled>
        {SWITCH_LABEL}
      </SettingsSwitch>,
    );

    // Act
    await user.tab();
    await user.keyboard("[Space]");

    // Assert
    expect(screen.getByTestId("test-switch")).not.toHaveFocus();
    expect(screen.getByTestId("test-switch")).not.toBeChecked();
    expect(onToggleMock).not.toHaveBeenCalled();
  });

  it("should render a beta tag if isBeta is true", () => {
    const { rerender } = render(
      <SettingsSwitch testId="test-switch" onToggle={vi.fn()} isBeta={false}>
        {SWITCH_LABEL}
      </SettingsSwitch>,
    );

    expect(screen.queryByText(/beta/i)).not.toBeInTheDocument();

    rerender(
      <SettingsSwitch testId="test-switch" onToggle={vi.fn()} isBeta>
        {SWITCH_LABEL}
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
        {SWITCH_LABEL}
      </SettingsSwitch>,
    );

    expect(screen.getByTestId("test-switch")).toBeChecked();

    const switchInput = screen.getByTestId("test-switch");
    await user.click(switchInput);
    expect(onToggleMock).toHaveBeenCalledWith(false);

    expect(screen.getByTestId("test-switch")).not.toBeChecked();
  });
});
