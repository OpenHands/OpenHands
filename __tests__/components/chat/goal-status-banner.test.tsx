import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { GoalStatusBanner } from "#/components/features/chat/goal-status-banner";
import { useGoalStore } from "#/stores/goal-store";
import { GoalStatus } from "#/types/agent-server/core/events/conversation-state-event";

const CONV = "conv-1";

const makeStatus = (overrides: Partial<GoalStatus> = {}): GoalStatus => ({
  active: true,
  status: "running",
  iteration: 1,
  max_iterations: 10,
  objective: "make pytest pass",
  verdict: null,
  ...overrides,
});

describe("<GoalStatusBanner />", () => {
  beforeEach(() => {
    useGoalStore.setState({ statusByConversation: {} });
  });

  it("renders nothing when there is no goal status", () => {
    const { container } = render(<GoalStatusBanner conversationId={CONV} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the live banner while the loop is active", () => {
    useGoalStore.getState().setStatus(CONV, makeStatus());
    render(<GoalStatusBanner conversationId={CONV} />);
    expect(screen.getByTestId("goal-status")).toBeInTheDocument();
    expect(screen.getByTestId("goal-spinner")).toBeInTheDocument();
  });

  it("renders nothing once the loop ends (the terminal status renders inline)", () => {
    useGoalStore.getState().setStatus(
      CONV,
      makeStatus({
        active: false,
        status: "complete",
        verdict: { score: 1, complete: true, missing: "" },
      }),
    );
    const { container } = render(<GoalStatusBanner conversationId={CONV} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("does not render a status from another conversation", () => {
    useGoalStore.getState().setStatus("other", makeStatus());
    const { container } = render(<GoalStatusBanner conversationId={CONV} />);
    expect(container).toBeEmptyDOMElement();
  });
});
