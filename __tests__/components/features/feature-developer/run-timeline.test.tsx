import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RunTimeline } from "#/components/features/feature-developer/run-timeline";
import type { FeatureDevRun } from "#/api/feature-developer-service/feature-developer-types";
import { I18nKey } from "#/i18n/declaration";

function run(overrides: Partial<FeatureDevRun> = {}): FeatureDevRun {
  return {
    id: "run-1",
    project_id: "proj-1",
    spec_text: "Build login",
    board_id: "board-1",
    status: "passed",
    current_ticket_index: 2,
    total_estimate_usd: 0.4,
    total_actual_usd: 0.5,
    max_concurrent_agents: 1,
    continue_on_failure: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    tickets: [
      {
        id: "t1",
        run_id: "run-1",
        card_id: "c1",
        title: "Add login form",
        status: "passed",
        branch_name: "feat/login-form",
        session_id: "s1",
        estimate_usd: 0.2,
        actual_usd: 0.25,
        error: null,
        started_at: "2026-01-01T00:00:00Z",
        finished_at: "2026-01-01T00:01:00Z",
        position: 0,
      },
      {
        id: "t2",
        run_id: "run-1",
        card_id: "c2",
        title: "Add session cookie",
        status: "passed",
        branch_name: "feat/session-cookie",
        session_id: "s2",
        estimate_usd: 0.2,
        actual_usd: 0.25,
        error: null,
        started_at: "2026-01-01T00:01:00Z",
        finished_at: "2026-01-01T00:02:00Z",
        position: 1,
      },
    ],
    ...overrides,
  };
}

describe("RunTimeline", () => {
  it("renders passed tickets with cost and branch", () => {
    render(<RunTimeline run={run()} />);

    expect(screen.getByTestId("feature-dev-run-status")).toHaveTextContent(
      I18nKey.FEATURE_DEV$STATUS_PASSED,
    );
    expect(
      screen.getByTestId("feature-dev-ticket-status-t1"),
    ).toHaveTextContent(I18nKey.FEATURE_DEV$STATUS_PASSED);
    expect(
      screen.getByTestId("feature-dev-ticket-branch-t1"),
    ).toHaveTextContent("feat/login-form");
  });

  it("renders failed ticket errors in red", () => {
    render(
      <RunTimeline
        run={run({
          status: "failed",
          tickets: [
            run().tickets[0],
            {
              ...run().tickets[1],
              status: "failed",
              error: "build failed",
            },
          ],
        })}
      />,
    );

    expect(screen.getByTestId("feature-dev-run-status")).toHaveTextContent(
      I18nKey.FEATURE_DEV$STATUS_FAILED,
    );
    expect(screen.getByTestId("feature-dev-ticket-error-t2")).toHaveTextContent(
      "build failed",
    );
    expect(screen.getByTestId("feature-dev-ticket-error-t2")).toHaveClass(
      "text-red-400",
    );
  });

  it("renders a partial run status chip", () => {
    render(
      <RunTimeline
        run={run({
          status: "partial",
          tickets: [
            run().tickets[0],
            { ...run().tickets[1], status: "failed", error: "loop failed" },
          ],
        })}
      />,
    );

    expect(screen.getByTestId("feature-dev-run-status")).toHaveTextContent(
      I18nKey.FEATURE_DEV$STATUS_PARTIAL,
    );
  });
});
