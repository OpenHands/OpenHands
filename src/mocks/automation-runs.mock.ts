import { AutomationRunStatus } from "#/types/automation";
import type { AutomationRun } from "#/types/automation";

const daysAgo = (days: number, hour = 9) => {
  const d = new Date(Date.now() - days * 86_400_000);
  d.setHours(hour, 0, 0, 0);
  return d.toISOString();
};

function makeRun(
  automationId: string,
  id: string,
  status: AutomationRunStatus,
  startedDaysAgo: number,
  hour = 9,
  hasConversation = true,
): AutomationRun {
  const started = daysAgo(startedDaysAgo, hour);
  const isInFlight =
    status === AutomationRunStatus.PENDING ||
    status === AutomationRunStatus.RUNNING;
  return {
    id,
    automation_id: automationId,
    status,
    conversation_id: hasConversation ? `conv-${id}` : null,
    // Runs that have a conversation also have a bash command; runs that
    // failed before sandbox creation have neither.
    bash_command_id: hasConversation ? `cmd-${id}` : null,
    sandbox_id: hasConversation ? `sandbox-${id}` : null,
    error_detail:
      status === AutomationRunStatus.FAILED
        ? "Process exited with code 1"
        : null,
    created_at: new Date(new Date(started).getTime() - 5_000).toISOString(),
    started_at: started,
    completed_at: isInFlight
      ? null
      : new Date(new Date(started).getTime() + 120_000).toISOString(),
    timeout_at: isInFlight
      ? new Date(new Date(started).getTime() + 600_000).toISOString()
      : null,
  };
}

export const MOCK_AUTOMATION_RUNS: Record<string, AutomationRun[]> = {
  "a1000000-0000-0000-0000-000000000001": [
    makeRun(
      "a1000000-0000-0000-0000-000000000001",
      "r1-00",
      AutomationRunStatus.RUNNING,
      0,
      0,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000001",
      "r1-01",
      AutomationRunStatus.COMPLETED,
      0,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000001",
      "r1-02",
      AutomationRunStatus.COMPLETED,
      1,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000001",
      "r1-03",
      AutomationRunStatus.FAILED,
      2,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000001",
      "r1-04",
      AutomationRunStatus.COMPLETED,
      3,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000001",
      "r1-05",
      AutomationRunStatus.COMPLETED,
      4,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000001",
      "r1-06",
      AutomationRunStatus.COMPLETED,
      7,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000001",
      "r1-07",
      AutomationRunStatus.FAILED,
      8,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000001",
      "r1-08",
      AutomationRunStatus.COMPLETED,
      9,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000001",
      "r1-09",
      AutomationRunStatus.COMPLETED,
      10,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000001",
      "r1-10",
      AutomationRunStatus.COMPLETED,
      11,
    ),
  ],
  "a1000000-0000-0000-0000-000000000002": [
    makeRun(
      "a1000000-0000-0000-0000-000000000002",
      "r2-00",
      AutomationRunStatus.PENDING,
      0,
      0,
      false,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000002",
      "r2-01",
      AutomationRunStatus.COMPLETED,
      0,
      1,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000002",
      "r2-02",
      AutomationRunStatus.COMPLETED,
      1,
      1,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000002",
      "r2-03",
      AutomationRunStatus.COMPLETED,
      2,
      1,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000002",
      "r2-04",
      AutomationRunStatus.FAILED,
      3,
      1,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000002",
      "r2-05",
      AutomationRunStatus.COMPLETED,
      4,
      1,
    ),
  ],
  "a1000000-0000-0000-0000-000000000003": [
    makeRun(
      "a1000000-0000-0000-0000-000000000003",
      "r3-01",
      AutomationRunStatus.COMPLETED,
      1,
    ),
    // Terminal statuses the backend emits besides COMPLETED/FAILED.
    makeRun(
      "a1000000-0000-0000-0000-000000000003",
      "r3-02",
      AutomationRunStatus.CANCELLED,
      2,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000003",
      "r3-03",
      AutomationRunStatus.SKIPPED,
      3,
      9,
      false,
    ),
  ],
  "a1000000-0000-0000-0000-000000000004": [
    makeRun(
      "a1000000-0000-0000-0000-000000000004",
      "r4-01",
      AutomationRunStatus.FAILED,
      14,
      11,
      false,
    ), // Failed before sandbox creation
    makeRun(
      "a1000000-0000-0000-0000-000000000004",
      "r4-02",
      AutomationRunStatus.COMPLETED,
      21,
      11,
    ),
  ],
  "a1000000-0000-0000-0000-000000000005": [],
  "a1000000-0000-0000-0000-000000000006": [
    makeRun(
      "a1000000-0000-0000-0000-000000000006",
      "r6-01",
      AutomationRunStatus.COMPLETED,
      0,
      14,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000006",
      "r6-02",
      AutomationRunStatus.COMPLETED,
      0,
      11,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000006",
      "r6-03",
      AutomationRunStatus.FAILED,
      1,
      16,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000006",
      "r6-04",
      AutomationRunStatus.COMPLETED,
      2,
      10,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000006",
      "r6-05",
      AutomationRunStatus.COMPLETED,
      3,
      9,
    ),
  ],
  "a1000000-0000-0000-0000-000000000007": [
    makeRun(
      "a1000000-0000-0000-0000-000000000007",
      "r7-01",
      AutomationRunStatus.COMPLETED,
      3,
      15,
    ),
    makeRun(
      "a1000000-0000-0000-0000-000000000007",
      "r7-02",
      AutomationRunStatus.COMPLETED,
      10,
      12,
    ),
  ],
};
