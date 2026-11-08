import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { compile } from "@tailwindcss/node";
import { createElement } from "react";
import { act, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../test-utils";
import { AutomationRunActivitySparkline } from "#/components/features/home/featured-automations/automation-run-activity-sparkline";
import {
  AUTOMATION_RUN_ACTIVITY_TRACK_PX,
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

function renderedHeightPx(
  compiler: Awaited<ReturnType<typeof compile>>,
  element: HTMLElement,
): number {
  const css = compiler.build([...element.classList]);
  const declarations = [...css.matchAll(/\.((?:\\.|[\w-])+)\s*\{([^}]+)\}/g)]
    .filter(([, selector]) =>
      element.classList.contains(selector.replace(/\\(.)/g, "$1")),
    )
    .map(([, , body]) => body)
    .join("\n");
  const height = /(?:^|\n)\s*height:\s*var\((--[\w-]+)\);/.exec(declarations);
  expect(height).not.toBeNull();
  const value = element.style.getPropertyValue(height![1]);
  expect(value).toMatch(/^\d+(?:\.\d+)?px$/);
  return Number.parseFloat(value);
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
    expect(classes).toContain("bg-status-success");
    expect(classes).toContain("animate-pulse");
  });

  it("uses a grey diagonal hatch for pending bars", () => {
    const classes = barColorClassForStatus(AutomationRunStatus.PENDING);
    expect(classes).toContain("repeating-linear-gradient");
    expect(classes).toContain("var(--oh-muted)");
    expect(classes).toContain("var(--oh-border)");
  });

  it("uses warning color for blocked task outcomes", () => {
    expect(barColorClassForStatus("blocked")).toContain("bg-warning");
  });

  it("uses warning color for partial and unknown task outcomes", () => {
    expect(barColorClassForStatus("partial_success")).toContain("bg-warning");
    expect(barColorClassForStatus("unknown")).toContain("bg-warning");
  });
});

describe("AutomationRunActivitySparkline", () => {
  it("builds short, long and unknown bar heights from their rendered CSS properties", async () => {
    // Arrange: different durations use the shared metrics without prescribing
    // the utility spelling or repeating the property's name.
    const runs = [
      makeRun({ id: "short", completed_at: "2026-08-01T10:00:01.000Z" }),
      makeRun({ id: "long", completed_at: "2026-08-01T12:00:00.000Z" }),
      makeRun({ id: "unknown", started_at: undefined, completed_at: null }),
    ];
    const compiler = await compile(
      await readFile(resolve("src/tailwind.css"), "utf8"),
      { base: resolve("src"), onDependency: () => {} },
    );
    renderWithProviders(
      createElement(AutomationRunActivitySparkline, {
        automationId: "automation-1",
        runs,
      }),
    );

    // Act + Assert: each native link's rendered bar supplies the value consumed
    // by its compiled height rule, including the unknown-duration fallback.
    for (const run of runs) {
      const link = screen
        .getAllByRole("link")
        .find((element) =>
          element.getAttribute("href")?.endsWith(`?run=${run.id}`),
        );
      const bar = link!.firstElementChild as HTMLElement;
      expect(renderedHeightPx(compiler, bar)).toBe(
        durationMsToSparklineBarHeightPx(getAutomationRunDurationMs(run)),
      );
    }
  });

  it("updates a running bar's CSS height on the one-second elapsed-time tick", async () => {
    // Arrange: compile before fake timers so stylesheet loading stays real.
    const compiler = await compile(
      await readFile(resolve("src/tailwind.css"), "utf8"),
      { base: resolve("src"), onDependency: () => {} },
    );
    const run = makeRun({
      status: AutomationRunStatus.RUNNING,
      completed_at: null,
    });
    vi.useFakeTimers();
    try {
      vi.setSystemTime(Date.parse(run.started_at!) + 1_000);
      const { unmount } = renderWithProviders(
        createElement(AutomationRunActivitySparkline, {
          automationId: "automation-1",
          runs: [run],
        }),
      );
      const bar = screen.getByRole("link").firstElementChild as HTMLElement;
      const initialHeight = renderedHeightPx(compiler, bar);

      // Act + Assert: no earlier render, then the live timer refreshes the
      // property's duration-derived height without a new run-list response.
      act(() => vi.advanceTimersByTime(999));
      expect(renderedHeightPx(compiler, bar)).toBe(initialHeight);
      act(() => vi.advanceTimersByTime(1));
      const updatedHeight = renderedHeightPx(compiler, bar);
      expect(updatedHeight).toBe(
        durationMsToSparklineBarHeightPx(
          getAutomationRunDurationMs(run, Date.now()),
        ),
      );
      expect(updatedHeight).toBeGreaterThan(initialHeight);
      unmount();
    } finally {
      vi.useRealTimers();
    }
  });

  it("builds the track height from the custom property supplied by the rendered component", async () => {
    // Arrange: use the real component and production stylesheet, without
    // prescribing a utility spelling or repeating its custom-property name.
    renderWithProviders(
      createElement(AutomationRunActivitySparkline, {
        automationId: "automation-1",
        runs: [makeRun()],
        testId: "run-activity-track",
      }),
    );
    const track = screen.getByTestId("run-activity-track");
    const compiler = await compile(
      await readFile(resolve("src/tailwind.css"), "utf8"),
      { base: resolve("src"), onDependency: () => {} },
    );

    // Act + Assert: the emitted height rule consumes the property's real value.
    // Removing the class or misspelling either side breaks this contract.
    expect(renderedHeightPx(compiler, track)).toBe(
      AUTOMATION_RUN_ACTIVITY_TRACK_PX,
    );
  });
});
