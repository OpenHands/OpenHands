import { describe, expect, it } from "vitest";

import { MOCK_AUTOMATION_RUNS } from "#/mocks/automation-runs.mock";
import {
  resolveRunPhaseText,
  shouldShowRunPhase,
} from "#/components/features/automations/detail/run-phase";
import { AutomationRunStatus, type AutomationRun } from "#/types/automation";
import { I18nKey } from "#/i18n/declaration";

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

// Stands in for `t()`: a known code must resolve to something other than the
// code itself, so the assertions below cannot pass on a raw code.
const t = (key: I18nKey) => `translated:${key}`;

describe("mock automation runs — the phase branches mock mode has to reach", () => {
  it("has in-flight runs at all, or no surface ever shows a phase", () => {
    expect(IN_FLIGHT.length).toBeGreaterThan(0);
  });

  it("covers a phase code the frontend translates", () => {
    const translated = IN_FLIGHT.filter(
      (run) =>
        resolveRunPhaseText(t, run.phase_code, run.phase_label)?.startsWith(
          "translated:",
        ) ?? false,
    );

    expect(translated.length).toBeGreaterThan(0);
  });

  it("covers a custom automation's own code, falling back to its free-form label", () => {
    const custom = IN_FLIGHT.filter(
      (run) =>
        run.phase_code != null &&
        run.phase_label != null &&
        resolveRunPhaseText(t, run.phase_code, run.phase_label) ===
          run.phase_label,
    );

    expect(custom.length).toBeGreaterThan(0);
  });

  it("covers a phase carrying only a label, which the service also accepts", () => {
    const labelOnly = IN_FLIGHT.filter(
      (run) => run.phase_code == null && !!run.phase_label,
    );

    expect(labelOnly.length).toBeGreaterThan(0);
  });

  it("covers a phase carrying only a code, which the service also accepts", () => {
    const codeOnly = IN_FLIGHT.filter(
      (run) =>
        !run.phase_label &&
        !!run.phase_code &&
        resolveRunPhaseText(t, run.phase_code, run.phase_label) ===
          run.phase_code,
    );

    expect(codeOnly.length).toBeGreaterThan(0);
  });

  it("covers an in-flight run with no phase fields at all — an older service", () => {
    const noPhase = IN_FLIGHT.filter(
      (run) => run.phase_code === undefined && run.phase_label === undefined,
    );

    expect(noPhase.length).toBeGreaterThan(0);
  });

  it("covers a failed run that kept the phase it stopped at", () => {
    const failedWithPhase = ALL_RUNS.filter(
      (run) =>
        run.status === AutomationRunStatus.FAILED &&
        resolveRunPhaseText(t, run.phase_code, run.phase_label) != null,
    );

    expect(failedWithPhase.length).toBeGreaterThan(0);
  });

  it("covers a finished run that has a phase on record but never shows it", () => {
    const hiddenPhase = ALL_RUNS.filter(
      (run) =>
        !shouldShowRunPhase(run.status) &&
        resolveRunPhaseText(t, run.phase_code, run.phase_label) != null,
    );

    expect(hiddenPhase.length).toBeGreaterThan(0);
  });

  it("gives every phase-bearing run a usable timestamp, or the age would read 'Invalid Date'", () => {
    const withPhase = ALL_RUNS.filter(
      (run) => run.phase_code || run.phase_label,
    );
    expect(withPhase.length).toBeGreaterThan(0);

    for (const run of withPhase) {
      if (run.phase_updated_at == null) continue;
      expect(Number.isNaN(new Date(run.phase_updated_at).getTime())).toBe(
        false,
      );
    }
  });

  it("keeps every fixture within the service's own limits on code and label", () => {
    for (const run of ALL_RUNS) {
      if (run.phase_code)
        expect(run.phase_code.length).toBeLessThanOrEqual(128);
      if (run.phase_label)
        expect(run.phase_label.length).toBeLessThanOrEqual(200);
    }
  });
});
