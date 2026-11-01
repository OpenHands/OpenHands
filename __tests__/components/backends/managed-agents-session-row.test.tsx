import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { MarsSession } from "#/api/mars/mars-tunnel-backend";
import { ManagedAgentsSessionRow } from "#/components/features/backends/managed-agents/managed-agents-session-row";

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
