import { AutomationRunStatus } from "#/types/automation";
import type { AutomationRun } from "#/types/automation";

const daysAgo = (days: number, hour = 9) => {
  const d = new Date(Date.now() - days * 86_400_000);
  d.setHours(hour, 0, 0, 0);
  return d.toISOString();
};

const minutesAgo = (minutes: number) =>
  new Date(Date.now() - minutes * 60_000).toISOString();

/** Omitting this entirely is a case too: an older service sends no phase. */
interface PhaseFixture {
  code: string | null;
  label: string | null;
  /** Minutes since the phase was last written; defaults to the run's start. */
  ageMinutes?: number;
}

function phaseFields(phase: PhaseFixture | undefined, fallbackAt: string) {
  if (!phase) return {};
  return {
    phase_code: phase.code,
    phase_label: phase.label,
    phase_updated_at:
      phase.ageMinutes === undefined
        ? fallbackAt
        : minutesAgo(phase.ageMinutes),
  };
}

function makeRun(
  id: string,
  status: AutomationRunStatus,
  startedDaysAgo: number,
  hour = 9,
  hasConversation = true,
  phase?: PhaseFixture,
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
    ...phaseFields(phase, started),
    started_at: started,
    completed_at: new Date(new Date(started).getTime() + 120_000).toISOString(),
  };
}

/**
 * A run that has not reached a terminal status, started relative to page load
 * so its elapsed time keeps growing while the dashboard stays open. These are
 * the rows the phase exists for: without one they render as a bare
 * "Running"/"Pending" pill for however long the run takes.
 */
function makeInFlightRun(
  id: string,
  status: AutomationRunStatus.RUNNING | AutomationRunStatus.PENDING,
  startedMinutesAgo: number,
  hasConversation = true,
  phase?: PhaseFixture,
): AutomationRun {
  const startedAt = minutesAgo(startedMinutesAgo);
  return {
    id,
    status,
    conversation_id: hasConversation ? `conv-${id}` : null,
    bash_command_id: hasConversation ? `cmd-${id}` : null,
    error_detail: null,
    ...phaseFields(phase, startedAt),
    started_at: startedAt,
    completed_at: null,
  };
}

// Newest run first: the cards and home rows read runs[0] as the latest, so
// the in-flight rows below are the ones they show. Only three automations get
// one: `latestRun` also drives health, "Last run" and the status badge, so a
// never-terminal fixture pins all three to "running".
export const MOCK_AUTOMATION_RUNS: Record<string, AutomationRun[]> = {
  "a1000000-0000-0000-0000-000000000001": [
    makeRun("r1-01", AutomationRunStatus.COMPLETED, 0),
    makeRun("r1-02", AutomationRunStatus.COMPLETED, 1),
    // Failed, keeping the phase it stopped at: died while the sandbox came up.
    makeRun("r1-03", AutomationRunStatus.FAILED, 2, 9, true, {
      code: "sandbox_provisioning",
      label: null,
    }),
    // A phase on record that no screen shows, but the export still carries.
    makeRun("r1-04", AutomationRunStatus.COMPLETED, 3, 9, true, {
      code: "running_agent",
      label: null,
    }),
    makeRun("r1-05", AutomationRunStatus.COMPLETED, 4),
    makeRun("r1-06", AutomationRunStatus.COMPLETED, 7),
    makeRun("r1-07", AutomationRunStatus.FAILED, 8),
    makeRun("r1-08", AutomationRunStatus.COMPLETED, 9),
    makeRun("r1-09", AutomationRunStatus.COMPLETED, 10),
    makeRun("r1-10", AutomationRunStatus.COMPLETED, 11),
  ],
  "a1000000-0000-0000-0000-000000000002": [
    // Queued: no sandbox or conversation yet, and only the phase says so.
    makeInFlightRun("r2-00", AutomationRunStatus.PENDING, 2, false, {
      code: "queued",
      label: null,
    }),
    makeRun("r2-01", AutomationRunStatus.COMPLETED, 0, 1),
    makeRun("r2-02", AutomationRunStatus.COMPLETED, 1, 1),
    makeRun("r2-03", AutomationRunStatus.COMPLETED, 2, 1),
    makeRun("r2-04", AutomationRunStatus.FAILED, 3, 1),
    makeRun("r2-05", AutomationRunStatus.COMPLETED, 4, 1),
  ],
  "a1000000-0000-0000-0000-000000000003": [
    // The case the issue is about: 52 minutes into a job that usually takes
    // two, stuck on one step for 41 of them. The code is the automation's
    // own, so its label is shown verbatim — and is long enough to be clipped.
    makeInFlightRun("r3-00", AutomationRunStatus.RUNNING, 52, true, {
      code: "diffing_docs_tree",
      label: "Diffing 340 changed files against main",
      ageMinutes: 41,
    }),
    makeRun("r3-01", AutomationRunStatus.COMPLETED, 1),
    // Terminal statuses the backend emits besides COMPLETED/FAILED.
    makeRun("r3-02", AutomationRunStatus.CANCELLED, 2),
    makeRun("r3-03", AutomationRunStatus.SKIPPED, 3, 9, false),
  ],
  "a1000000-0000-0000-0000-000000000004": [
    makeRun("r4-01", AutomationRunStatus.FAILED, 14, 11, false), // Failed before sandbox creation
    makeRun("r4-02", AutomationRunStatus.COMPLETED, 21, 11),
  ],
  "a1000000-0000-0000-0000-000000000005": [],
  "a1000000-0000-0000-0000-000000000006": [
    // A label with no code: the service accepts that, so the UI must show it.
    makeInFlightRun("r6-00", AutomationRunStatus.RUNNING, 13, true, {
      code: null,
      label: "Running QA checks on PR #4821",
      ageMinutes: 4,
    }),
    // A code with no label: also accepted, and shown as the raw code.
    makeInFlightRun("r6-06", AutomationRunStatus.RUNNING, 6, true, {
      code: "checking_out",
      label: null,
      ageMinutes: 2,
    }),
    // No phase fields at all — an older service: a status pill, no empty
    // slot. Below the head, so it costs no card its health.
    makeInFlightRun("r6-07", AutomationRunStatus.RUNNING, 21),
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
