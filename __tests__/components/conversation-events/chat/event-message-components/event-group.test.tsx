import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { EventGroup } from "#/components/conversation-events/chat/event-message-components/event-group";
import {
  ActionEvent,
  ObservationEvent,
  SecurityRisk,
} from "#/types/agent-server/core";
import { ExecuteBashAction } from "#/types/agent-server/core/base/action";
import { ExecuteBashObservation } from "#/types/agent-server/core/base/observation";

const makeBashAction = (
  id: string,
  command: string,
): ActionEvent<ExecuteBashAction> => ({
  id,
  timestamp: new Date().toISOString(),
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
): ObservationEvent<ExecuteBashObservation> => ({
  id,
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "execute_bash",
  tool_call_id: `call_${actionId}`,
  action_id: actionId,
  observation: {
    kind: "ExecuteBashObservation",
    content: [{ type: "text", text: "ok" }],
    command,
    exit_code: 0,
    error: false,
    timeout: false,
    metadata: {} as never,
  },
});

describe("EventGroup", () => {
  it("returns null for an empty events array", () => {
    const { container } = renderWithProviders(
      <EventGroup events={[]}>
        <div>child</div>
      </EventGroup>,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("renders a 'completed' summary when all events are observations", () => {
    const events = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashObservation("o2", "a2", "pwd"),
      makeBashObservation("o3", "a3", "whoami"),
    ];

    renderWithProviders(
      <EventGroup events={events}>
        <div data-testid="child">child content</div>
      </EventGroup>,
    );

    expect(
      screen.getByText("EVENT_GROUP$ACTIONS_COMPLETED"),
    ).toBeInTheDocument();
    // Children should not be visible in the collapsed state.
    expect(screen.queryByTestId("child")).not.toBeInTheDocument();
  });

  it("renders a progress summary and the running action title while in flight", () => {
    const events = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashObservation("o2", "a2", "pwd"),
      // Last action has NOT been replaced by an observation -> still running.
      makeBashAction("a3", "echo hello"),
    ];

    renderWithProviders(
      <EventGroup events={events}>
        <div>child</div>
      </EventGroup>,
    );

    expect(
      screen.getByText("EVENT_GROUP$ACTIONS_PROGRESS"),
    ).toBeInTheDocument();
    // The running action's title is rendered next to the summary. Translations
    // aren't loaded in tests, so we just verify the action's translation key
    // shows up (the localized version would interpolate the command).
    expect(screen.getByText(/ACTION_MESSAGE\$RUN/)).toBeInTheDocument();
  });

  it("keeps showing the latest completed action's title while the group is still the live tail", () => {
    // All observations -> nothing in flight, but the group has not been
    // "moved past" yet, so we expect the latest observation's title to keep
    // showing as the prominent summary alongside the completed count.
    const events = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashObservation("o2", "a2", "pwd"),
      makeBashObservation("o3", "a3", "whoami"),
    ];

    renderWithProviders(
      <EventGroup events={events}>
        <div>child</div>
      </EventGroup>,
    );

    // Latest observation's title is still in the summary line.
    expect(screen.getByText(/OBSERVATION_MESSAGE\$RUN/)).toBeInTheDocument();
    // ...next to the completed count.
    expect(
      screen.getByText("EVENT_GROUP$ACTIONS_COMPLETED"),
    ).toBeInTheDocument();
  });

  it("hides the latest action title once the group is finalized", () => {
    const events = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashObservation("o2", "a2", "pwd"),
      makeBashObservation("o3", "a3", "whoami"),
    ];

    renderWithProviders(
      <EventGroup events={events} isFinalized>
        <div>child</div>
      </EventGroup>,
    );

    expect(
      screen.getByText("EVENT_GROUP$ACTIONS_COMPLETED"),
    ).toBeInTheDocument();
    // Once moved past, we collapse to just the count — the per-action title
    // and the success check both go away.
    expect(
      screen.queryByText(/OBSERVATION_MESSAGE\$RUN/),
    ).not.toBeInTheDocument();
    expect(screen.queryByTestId("status-icon")).not.toBeInTheDocument();
  });

  it("shows a spinner while running and no status icon when done", () => {
    const running = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashAction("a2", "pwd"),
    ];

    const { rerender } = renderWithProviders(
      <EventGroup events={running}>
        <div>child</div>
      </EventGroup>,
    );
    expect(screen.getByTestId("spinner-icon")).toBeInTheDocument();
    expect(screen.queryByTestId("status-icon")).not.toBeInTheDocument();

    const done = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashObservation("o2", "a2", "pwd"),
    ];
    rerender(
      <EventGroup events={done}>
        <div>child</div>
      </EventGroup>,
    );
    expect(screen.queryByTestId("spinner-icon")).not.toBeInTheDocument();
    expect(screen.queryByTestId("status-icon")).not.toBeInTheDocument();
  });

  it("updates accessibility state while toggling the group", async () => {
    const events = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashObservation("o2", "a2", "pwd"),
      makeBashObservation("o3", "a3", "whoami"),
    ];
    const user = userEvent.setup();

    renderWithProviders(
      <EventGroup events={events}>
        <div data-testid="child">child content</div>
      </EventGroup>,
    );

    const toggle = screen.getByTestId("event-group-toggle");
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(toggle).toHaveAttribute("aria-label", "EVENT_GROUP$EXPAND");
    expect(screen.queryByRole("region")).not.toBeInTheDocument();
    expect(screen.queryByTestId("child")).not.toBeInTheDocument();

    await user.click(toggle);

    const content = screen.getByRole("region");
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(toggle).toHaveAttribute("aria-label", "EVENT_GROUP$COLLAPSE");
    expect(toggle).toHaveAttribute("aria-controls", content.id);
    expect(content).toHaveAttribute("aria-labelledby", toggle.id);
    expect(screen.getByTestId("child")).toBeInTheDocument();

    await user.click(toggle);

    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(toggle).toHaveAttribute("aria-label", "EVENT_GROUP$EXPAND");
    expect(screen.queryByRole("region")).not.toBeInTheDocument();
    expect(screen.queryByTestId("child")).not.toBeInTheDocument();
  });
});
