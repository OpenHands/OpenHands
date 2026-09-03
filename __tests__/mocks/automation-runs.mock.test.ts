import { describe, expect, it } from "vitest";

import { MOCK_AUTOMATION_RUNS } from "#/mocks/automation-runs.mock";
import {
  resolveRunPhaseText,
  shouldShowRunPhase,
} from "#/components/features/automations/detail/run-phase";
import { AutomationRunStatus, type AutomationRun } from "#/types/automation";

// `VITE_MOCK_API=true` is how the automations UI is developed and demoed
// without a local automation service, so a phase branch that no fixture
// reaches is a branch nobody sees before it ships. These assertions are on
// the fixtures themselves: they fail when a case is dropped, which is the
// only way a rendering branch quietly stops being exercised.
const ALL_RUNS: AutomationRun[] = Object.values(MOCK_AUTOMATION_RUNS).flat();
const IN_FLIGHT = ALL_RUNS.filter(
  (run) =>
    run.status === AutomationRunStatus.RUNNING ||
    run.status === AutomationRunStatus.PENDING,
);

describe("mock automation runs — the phase branches mock mode has to reach", () => {
  it("has in-flight runs at all, or no surface ever shows a phase", () => {
    expect(IN_FLIGHT.length).toBeGreaterThan(0);
  });

  it("covers the shipped current_phase string", () => {
    const shipped = IN_FLIGHT.filter((run) => !!run.current_phase);

    expect(shipped.length).toBeGreaterThan(0);
  });

  it("covers an in-flight run with no phase fields at all — an older service", () => {
    const noPhase = IN_FLIGHT.filter((run) => run.current_phase === undefined);

    expect(noPhase.length).toBeGreaterThan(0);
  });

  it("covers a failed run that kept the phase it stopped at", () => {
    const failedWithPhase = ALL_RUNS.filter(
      (run) =>
        run.status === AutomationRunStatus.FAILED &&
        resolveRunPhaseText(run.current_phase) != null,
    );

    expect(failedWithPhase.length).toBeGreaterThan(0);
  });

  it("covers a finished run that has a phase on record but never shows it", () => {
    const hiddenPhase = ALL_RUNS.filter(
      (run) =>
        !shouldShowRunPhase(run.status) &&
        resolveRunPhaseText(run.current_phase) != null,
    );

    expect(hiddenPhase.length).toBeGreaterThan(0);
  });

  it("keeps every fixture within the service's own limit on phase text", () => {
    for (const run of ALL_RUNS) {
      if (run.current_phase)
        expect(run.current_phase.length).toBeLessThanOrEqual(200);
    }
  });
});
