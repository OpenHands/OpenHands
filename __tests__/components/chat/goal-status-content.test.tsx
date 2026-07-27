import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GoalStatusContent } from "#/components/features/chat/goal-status-content";
import { I18nKey } from "#/i18n/declaration";
import { GoalStatus } from "#/types/agent-server/core/events/conversation-state-event";

const makeStatus = (overrides: Partial<GoalStatus> = {}): GoalStatus => ({
  active: true,
  status: "running",
  iteration: 1,
  max_iterations: 10,
  objective: "make pytest pass",
  verdict: null,
  ...overrides,
});

describe("<GoalStatusContent />", () => {
  it("shows the objective and a spinner while running", () => {
    render(<GoalStatusContent status={makeStatus()} />);
    expect(screen.getByText("make pytest pass")).toBeInTheDocument();
    expect(screen.getByTestId("goal-spinner")).toBeInTheDocument();
    expect(screen.queryByTestId("goal-done")).toBeNull();
    expect(screen.queryByTestId("goal-ended")).toBeNull();
  });

  it("shows a green check when complete", () => {
    render(
      <GoalStatusContent
        status={makeStatus({
          active: false,
          status: "complete",
          verdict: { score: 1, complete: true, missing: "" },
        })}
      />,
    );
    expect(screen.getByTestId("goal-done")).toBeInTheDocument();
    expect(screen.queryByTestId("goal-spinner")).toBeNull();
    expect(screen.queryByTestId("goal-ended")).toBeNull();
  });

  it("shows a muted cross, not a check, when it ends without completing", () => {
    render(
      <GoalStatusContent
        status={makeStatus({
          active: false,
          status: "capped",
          verdict: { score: 0.7, complete: false, missing: "needs more tests" },
        })}
      />,
    );
    expect(screen.getByTestId("goal-ended")).toBeInTheDocument();
    expect(screen.queryByTestId("goal-done")).toBeNull();
  });

  it("expands the judge's missing note for a terminal status", () => {
    render(
      <GoalStatusContent
        status={makeStatus({
          active: false,
          status: "capped",
          verdict: { score: 0.7, complete: false, missing: "needs more tests" },
        })}
      />,
    );
    expect(screen.getByText(I18nKey.GOAL$MISSING)).toBeInTheDocument();
  });

  it("keeps the missing note collapsed while running", () => {
    render(
      <GoalStatusContent
        status={makeStatus({
          verdict: { score: 0.5, complete: false, missing: "needs tests" },
        })}
      />,
    );
    expect(screen.queryByText(I18nKey.GOAL$MISSING)).toBeNull();
  });
});
