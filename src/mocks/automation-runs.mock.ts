import { AutomationRunStatus } from "#/types/automation";
import type { AutomationRun } from "#/types/automation";

const daysAgo = (days: number, hour = 9) => {
  const d = new Date(Date.now() - days * 86_400_000);
  d.setHours(hour, 0, 0, 0);
  return d.toISOString();
};

function makeRun(
  id: string,
  status: AutomationRunStatus,
  startedDaysAgo: number,
  hour = 9,
  hasConversation = true,
): AutomationRun {
  const started = daysAgo(startedDaysAgo, hour);
  return {
    id,
    status,
    conversation_id: hasConversation ? `conv-${id}` : null,
    // Runs that have a conversation also have a bash command; runs that
    // failed before sandbox creation have neither.
    bash_command_id: hasConversation ? `cmd-${id}` : null,
    error_detail:
      status === AutomationRunStatus.FAILED
        ? "Process exited with code 1"
        : null,
    started_at: started,
    completed_at: new Date(new Date(started).getTime() + 120_000).toISOString(),
  };
}

export const MOCK_AUTOMATION_RUNS: Record<string, AutomationRun[]> = {
  "a1000000-0000-0000-0000-000000000001": [
    makeRun("r1-01", AutomationRunStatus.COMPLETED, 0),
    makeRun("r1-02", AutomationRunStatus.COMPLETED, 1),
    makeRun("r1-03", AutomationRunStatus.FAILED, 2),
    makeRun("r1-04", AutomationRunStatus.COMPLETED, 3),
    makeRun("r1-05", AutomationRunStatus.COMPLETED, 4),
    makeRun("r1-06", AutomationRunStatus.COMPLETED, 7),
    makeRun("r1-07", AutomationRunStatus.FAILED, 8),
    makeRun("r1-08", AutomationRunStatus.COMPLETED, 9),
    makeRun("r1-09", AutomationRunStatus.COMPLETED, 10),
    makeRun("r1-10", AutomationRunStatus.COMPLETED, 11),
  ],
  "a1000000-0000-0000-0000-000000000002": [
    makeRun("r2-01", AutomationRunStatus.COMPLETED, 0, 1),
    makeRun("r2-02", AutomationRunStatus.COMPLETED, 1, 1),
    makeRun("r2-03", AutomationRunStatus.COMPLETED, 2, 1),
    makeRun("r2-04", AutomationRunStatus.FAILED, 3, 1),
    makeRun("r2-05", AutomationRunStatus.COMPLETED, 4, 1),
  ],
  "a1000000-0000-0000-0000-000000000003": [
    makeRun("r3-01", AutomationRunStatus.COMPLETED, 1),
    makeRun("r3-02", AutomationRunStatus.COMPLETED, 2),
    makeRun("r3-03", AutomationRunStatus.COMPLETED, 3),
  ],
  "a1000000-0000-0000-0000-000000000004": [
    makeRun("r4-01", AutomationRunStatus.FAILED, 14, 11, false), // Failed before sandbox creation
    makeRun("r4-02", AutomationRunStatus.COMPLETED, 21, 11),
  ],
  "a1000000-0000-0000-0000-000000000005": [],
  "a1000000-0000-0000-0000-000000000006": [
    makeRun("r6-01", AutomationRunStatus.COMPLETED, 0, 14),
    makeRun("r6-02", AutomationRunStatus.COMPLETED, 0, 11),
    makeRun("r6-03", AutomationRunStatus.FAILED, 1, 16),
    makeRun("r6-04", AutomationRunStatus.COMPLETED, 2, 10),
    makeRun("r6-05", AutomationRunStatus.COMPLETED, 3, 9),
  ],
  "a1000000-0000-0000-0000-000000000007": [
    makeRun("r7-01", AutomationRunStatus.COMPLETED, 3, 15),
    makeRun("r7-02", AutomationRunStatus.COMPLETED, 10, 12),
  ],
};
