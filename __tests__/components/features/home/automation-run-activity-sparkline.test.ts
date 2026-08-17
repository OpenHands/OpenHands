import { describe, expect, it } from "vitest";
import {
  AUTOMATION_RUN_ACTIVITY_MAX_BAR_PX,
  AUTOMATION_RUN_ACTIVITY_MIN_BAR_PX,
  AUTOMATION_RUN_ACTIVITY_UNKNOWN_BAR_PX,
  barColorClassForStatus,
  durationMsToSparklineBarHeightPx,
  formatDurationForTitle,
  getAutomationRunDurationMs,
} from "#/components/features/home/featured-automations/automation-run-activity-metrics";
import { AutomationRunStatus, type AutomationRun } from "#/types/automation";

function makeRun(overrides: Partial<AutomationRun> = {}): AutomationRun {
  return {
    id: "run-1",
    status: AutomationRunStatus.COMPLETED,
    conversation_id: null,
    bash_command_id: null,
    error_detail: null,
    started_at: "2026-08-01T10:00:00.000Z",
    completed_at: "2026-08-01T10:05:00.000Z",
    ...overrides,
  };
}

describe("getAutomationRunDurationMs", () => {
  it("returns completed_at - started_at for finished runs", () => {
    expect(
      getAutomationRunDurationMs(
        makeRun({
          started_at: "2026-08-01T10:00:00.000Z",
          completed_at: "2026-08-01T10:02:30.000Z",
        }),
      ),
    ).toBe(150_000);
  });

  it("uses elapsed time for in-flight runs with a usable start", () => {
    const now = Date.parse("2026-08-01T10:03:00.000Z");
    expect(
      getAutomationRunDurationMs(
        makeRun({
          status: AutomationRunStatus.RUNNING,
          started_at: "2026-08-01T10:00:00.000Z",
          completed_at: null,
        }),
        now,
      ),
    ).toBe(180_000);
  });

  it("returns null when timestamps are missing or invalid", () => {
    expect(
      getAutomationRunDurationMs(
        makeRun({
          status: AutomationRunStatus.FAILED,
          started_at: "2026-08-01T10:00:00.000Z",
          completed_at: null,
        }),
      ),
    ).toBeNull();
  });
});

describe("durationMsToSparklineBarHeightPx", () => {
  it("maps unknown duration to the mid fallback height", () => {
    expect(durationMsToSparklineBarHeightPx(null)).toBe(
      AUTOMATION_RUN_ACTIVITY_UNKNOWN_BAR_PX,
    );
  });

  it("grows with duration between the min and max bar heights", () => {
    const shortBar = durationMsToSparklineBarHeightPx(1_000);
    const mediumBar = durationMsToSparklineBarHeightPx(60_000);
    const longBar = durationMsToSparklineBarHeightPx(30 * 60_000);

    expect(shortBar).toBe(AUTOMATION_RUN_ACTIVITY_MIN_BAR_PX);
    expect(longBar).toBe(AUTOMATION_RUN_ACTIVITY_MAX_BAR_PX);
    expect(mediumBar).toBeGreaterThan(shortBar);
    expect(mediumBar).toBeLessThan(longBar);
  });
});

describe("formatDurationForTitle", () => {
  it("formats seconds, minutes, and hours", () => {
    expect(formatDurationForTitle(12_000)).toBe("12s");
    expect(formatDurationForTitle(180_000)).toBe("3m");
    expect(formatDurationForTitle(7_200_000)).toBe("2.0h");
  });
});

describe("barColorClassForStatus", () => {
  it("uses a pulsing success green for running bars", () => {
    const classes = barColorClassForStatus(AutomationRunStatus.RUNNING);
    expect(classes).toContain("bg-[var(--oh-status-success)]");
    expect(classes).toContain("animate-pulse");
  });

  it("uses a grey diagonal hatch for pending bars", () => {
    const classes = barColorClassForStatus(AutomationRunStatus.PENDING);
    expect(classes).toContain("repeating-linear-gradient");
    expect(classes).toContain("var(--oh-muted)");
    expect(classes).toContain("var(--oh-border)");
  });
});
