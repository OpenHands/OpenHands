/**
 * Integration tests for GenericEventMessageWrapper — specifically the
 * EventElapsedTime (titleTrailing) integration added for issue #16267.
 *
 * These tests verify:
 *  - A running ActionEvent shows a live elapsed-time counter.
 *  - A completed ObservationEvent + correspondingAction shows a static duration.
 *  - Missing correspondingAction → no timing rendered for the observation.
 *  - SkillReady synthetic events do not show timing.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { GenericEventMessageWrapper } from "#/components/conversation-events/chat/event-message-components/generic-event-message-wrapper";
import {
  ActionEvent,
  ObservationEvent,
  SecurityRisk,
} from "#/types/agent-server/core";
import { ExecuteBashAction } from "#/types/agent-server/core/base/action";
import { ExecuteBashObservation } from "#/types/agent-server/core/base/observation";
import {
  SkillReadyEvent,
} from "#/components/conversation-events/chat/event-content-helpers/create-skill-ready-event";

// ── Fixtures ────────────────────────────────────────────────────────────────

const makeBashAction = (
  id: string,
  command: string,
  timestamp: string,
): ActionEvent<ExecuteBashAction> => ({
  id,
  timestamp,
  source: "agent",
  thought: [],
  thinking_blocks: [],
  action: {
    kind: "ExecuteBashAction",
    command,
    is_input: false,
    timeout: null,
    reset: false,
  },
  tool_name: "execute_bash",
  tool_call_id: `call_${id}`,
  tool_call: {
    id: `call_${id}`,
    type: "function",
    function: {
      name: "execute_bash",
      arguments: JSON.stringify({ command }),
    },
  },
  llm_response_id: `response_${id}`,
  security_risk: SecurityRisk.UNKNOWN,
});

const makeBashObservation = (
  id: string,
  actionId: string,
  command: string,
  timestamp: string,
  exitCode = 0,
): ObservationEvent<ExecuteBashObservation> => ({
  id,
  timestamp,
  source: "environment",
  tool_name: "execute_bash",
  tool_call_id: `call_${actionId}`,
  action_id: actionId,
  observation: {
    kind: "ExecuteBashObservation",
    content: [{ type: "text", text: "ok" }],
    command,
    exit_code: exitCode,
    error: false,
    timeout: false,
    metadata: {} as never,
  },
});

// ── Tests ────────────────────────────────────────────────────────────────────

afterEach(() => {
  vi.useRealTimers();
});

describe("GenericEventMessageWrapper — elapsed time (issue #16267)", () => {
  it("shows an elapsed-time counter for a running ActionEvent", () => {
    vi.useFakeTimers();
    const now = new Date("2026-01-01T10:00:00.000Z");
    vi.setSystemTime(now);

    // Action started 7 seconds ago.
    const start = new Date(now.getTime() - 7_000).toISOString();
    const action = makeBashAction("a1", "sleep 60", start);

    renderWithProviders(
      <GenericEventMessageWrapper event={action} isLastMessage={false} />,
    );

    expect(screen.getByTestId("event-elapsed-time")).toHaveTextContent("7s");
  });

  it("updates the live counter each second for a running ActionEvent", async () => {
    vi.useFakeTimers();
    const now = new Date("2026-01-01T10:00:00.000Z");
    vi.setSystemTime(now);

    const start = now.toISOString();
    const action = makeBashAction("a1", "npm run build", start);

    renderWithProviders(
      <GenericEventMessageWrapper event={action} isLastMessage={false} />,
    );

    expect(screen.getByTestId("event-elapsed-time")).toHaveTextContent("0s");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3_000);
    });

    expect(screen.getByTestId("event-elapsed-time")).toHaveTextContent("3s");
  });

  it("shows a static duration for a completed ObservationEvent", () => {
    vi.useFakeTimers();
    const now = new Date("2026-01-01T10:00:10.000Z");
    vi.setSystemTime(now);

    const actionStart = "2026-01-01T10:00:00.000Z"; // 10 s before observation
    const obsEnd = "2026-01-01T10:00:10.000Z";

    const action = makeBashAction("a1", "ls -la", actionStart);
    const observation = makeBashObservation("o1", "a1", "ls -la", obsEnd);

    renderWithProviders(
      <GenericEventMessageWrapper
        event={observation}
        isLastMessage={false}
        correspondingAction={action}
      />,
    );

    expect(screen.getByTestId("event-elapsed-time")).toHaveTextContent("10s");
  });

  it("does not tick after the ObservationEvent is supplied", async () => {
    vi.useFakeTimers();
    const now = new Date("2026-01-01T10:00:05.000Z");
    vi.setSystemTime(now);

    const actionStart = "2026-01-01T10:00:00.000Z";
    const obsEnd = "2026-01-01T10:00:05.000Z";

    const action = makeBashAction("a1", "echo hi", actionStart);
    const observation = makeBashObservation("o1", "a1", "echo hi", obsEnd);

    renderWithProviders(
      <GenericEventMessageWrapper
        event={observation}
        isLastMessage={false}
        correspondingAction={action}
      />,
    );

    const textAtRender =
      screen.getByTestId("event-elapsed-time").textContent;

    // Advance 5 more seconds — no timer should update the static value.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });

    expect(screen.getByTestId("event-elapsed-time").textContent).toBe(
      textAtRender,
    );
  });

  it("renders no timing for a completed ObservationEvent without correspondingAction", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-01-01T10:00:10.000Z"));

    const observation = makeBashObservation(
      "o1",
      "a1",
      "ls",
      "2026-01-01T10:00:10.000Z",
    );

    renderWithProviders(
      // No correspondingAction prop — timing cannot be derived.
      <GenericEventMessageWrapper event={observation} isLastMessage={false} />,
    );

    expect(screen.queryByTestId("event-elapsed-time")).not.toBeInTheDocument();
  });

  it("renders no timing for a SkillReady synthetic event", () => {
    // Build a SkillReadyEvent directly (plain object) so we don't have to
    // supply real skill data just to test the timing guard.
    const skillReadyEvent: SkillReadyEvent = {
      id: "skill-ready-1",
      timestamp: "2026-01-01T10:00:00.000Z",
      source: "agent",
      _isSkillReadyEvent: true,
      _skillReadyContent: "some skill content",
      _skillReadyItems: [],
    };

    renderWithProviders(
      <GenericEventMessageWrapper
        event={skillReadyEvent}
        isLastMessage={false}
      />,
    );

    expect(screen.queryByTestId("event-elapsed-time")).not.toBeInTheDocument();
  });

  it("renders a soft-timeout observation (exit_code -1) with the action→observation elapsed time", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-01-01T10:00:30.000Z"));

    const actionStart = "2026-01-01T10:00:00.000Z";
    const obsEnd = "2026-01-01T10:00:30.000Z"; // 30 s partial result

    const action = makeBashAction("a1", "long-running.sh", actionStart);
    // exit_code -1 = soft timeout (still running in sandbox, partial output)
    const observation = makeBashObservation(
      "o1",
      "a1",
      "long-running.sh",
      obsEnd,
      -1,
    );

    renderWithProviders(
      <GenericEventMessageWrapper
        event={observation}
        isLastMessage={false}
        correspondingAction={action}
      />,
    );

    // Shows the elapsed time to the partial observation, not a live counter.
    expect(screen.getByTestId("event-elapsed-time")).toHaveTextContent("30s");
  });
});
