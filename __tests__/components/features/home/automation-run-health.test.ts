import { describe, expect, it } from "vitest";
import { deriveRunHealth } from "#/components/features/home/featured-automations/automation-run-health";
import type { LatestAutomationRunState } from "#/hooks/query/use-latest-automation-runs";
import {
  AutomationRunStatus,
  type AutomationFinishToolResponse,
  type AutomationRun,
} from "#/types/automation";

function makeRun(
  finishToolResponse: AutomationFinishToolResponse | string | null,
): AutomationRun {
  return {
    id: "run-1",
    status: AutomationRunStatus.COMPLETED,
    conversation_id: "conv-1",
    bash_command_id: "cmd-1",
    error_detail: null,
    started_at: "2026-01-01T10:00:00Z",
    completed_at: "2026-01-01T10:02:00Z",
    run_metadata: { finish_tool_response: finishToolResponse },
  };
}

function makeState(run: AutomationRun): LatestAutomationRunState {
  return {
    latestRun: run,
    recentRuns: [run],
    isLoading: false,
    isError: false,
  };
}

describe("deriveRunHealth", () => {
  it("treats a completed summary-only response as successful", () => {
    const run = makeRun({
      outcome_summary: "Created and published the weekly report.",
    });

    expect(deriveRunHealth(makeState(run))).toBe("success");
  });

  it.each(["blocked", "partial_success", "unexpected_status"])(
    "treats an explicitly reported %s outcome as needing review",
    (status) => {
      expect(deriveRunHealth(makeState(makeRun({ status })))).toBe("warning");
    },
  );
});
