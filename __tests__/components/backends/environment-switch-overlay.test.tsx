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
import { type MarsSession } from "#/api/mars/mars-tunnel-backend";
import { ManagedAgentsSessionRow } from "#/components/features/backends/managed-agents/managed-agents-session-row";

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

describe("managed-agents-session-row", () => {
  function renderRow(status: string, isConnected = false) {
    const session = {
      session_id: "sess_1",
      name: "agent-abc123",
      status,
    } as MarsSession;
    render(
      <ul>
        <ManagedAgentsSessionRow
          session={session}
          isConnected={isConnected}
          connectStage={null}
          isBusy={false}
          error={null}
          onConnect={vi.fn()}
          onOpen={vi.fn()}
          onDisconnect={vi.fn()}
          onPause={vi.fn()}
          onStop={vi.fn()}
        />
      </ul>,
    );
  }

  describe("ManagedAgentsSessionRow", () => {
    it.each(["SESSION_STATUS_READY", "SESSION_STATUS_PAUSED"])(
      "puts Stop directly left of Connect for a %s session",
      (status) => {
        renderRow(status);

        const stop = screen.getByTestId("managed-agents-stop-sess_1");
        const connect = screen.getByTestId("managed-agents-connect-sess_1");
        expect(stop.nextElementSibling).toBe(connect);
      },
    );

    it("keeps Open rightmost for a connected session, with Stop just left of it", () => {
      renderRow("SESSION_STATUS_READY", true);

      const stop = screen.getByTestId("managed-agents-stop-sess_1");
      const open = screen.getByTestId("managed-agents-open-sess_1");
      expect(stop.nextElementSibling).toBe(open);
      expect(open.nextElementSibling).toBeNull();
      expect(
        screen.queryByTestId("managed-agents-connect-sess_1"),
      ).not.toBeInTheDocument();
    });
  });
});
