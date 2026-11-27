import React from "react";
import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetEnvironmentSwitchOverlayForTests,
  dismissEnvironmentSwitch,
  ENVIRONMENT_SWITCH_DURATION_MS,
  EnvironmentSwitchOverlay,
  triggerEnvironmentSwitch,
} from "#/components/features/backends/environment-switch-overlay";

describe("EnvironmentSwitchOverlay", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    __resetEnvironmentSwitchOverlayForTests();
    vi.useRealTimers();
  });

  it("renders the overlay with the requested target on trigger and tears it down after the configured duration", () => {
    // Arrange
    render(<EnvironmentSwitchOverlay />);

    // Act — request a switch to a specific environment
    act(() => {
      triggerEnvironmentSwitch("Acme Cloud");
    });

    // Assert — overlay is mounted and carries the requested target
    expect(screen.getByTestId("environment-switch-overlay")).toHaveAttribute(
      "data-target",
      "Acme Cloud",
    );

    // Act — wait out the auto-hide window
    act(() => {
      vi.advanceTimersByTime(ENVIRONMENT_SWITCH_DURATION_MS);
    });

    // Assert — overlay tears itself down without further user action
    expect(
      screen.queryByTestId("environment-switch-overlay"),
    ).not.toBeInTheDocument();
  });

  it("hides immediately when dismissed before the auto-hide timeout", () => {
    render(<EnvironmentSwitchOverlay />);

    act(() => {
      triggerEnvironmentSwitch("Acme Cloud");
    });
    expect(
      screen.getByTestId("environment-switch-overlay"),
    ).toBeInTheDocument();

    act(() => {
      dismissEnvironmentSwitch();
    });

    expect(
      screen.queryByTestId("environment-switch-overlay"),
    ).not.toBeInTheDocument();
  });
});
