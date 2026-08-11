import { describe, expect, it } from "vitest";
import {
  applyDashboardView,
  computeOverviewTile,
  deriveAutomationHealth,
  formatAutomationFailureKind,
  getAutomationHealthDetails,
  formatCompactDuration,
  matchesAutomationSearch,
  summarizeAutomationRuns,
  type AutomationHealth,
  type AutomationRunSummary,
  type RunSummaryState,
} from "#/manifests/automation-insights";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";

function createAutomation(overrides: Partial<Automation> = {}): Automation {
  return {
    id: "widget-1",
    name: "Widget monitor",
    trigger: { type: "cron", schedule: "0 9 * * *" },
    enabled: true,
    prompt: "Watch the widgets",
    repository: "acme/widgets",
    model: "widget-profile",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function createRun(overrides: Partial<AutomationRun> = {}): AutomationRun {
  return {
    id: "run-1",
    status: AutomationRunStatus.COMPLETED,
    conversation_id: null,
    bash_command_id: null,
    error_detail: null,
    started_at: "2026-01-02T00:00:00Z",
    completed_at: "2026-01-02T00:01:00Z",
    ...overrides,
  };
}

function settled(summary: Partial<AutomationRunSummary>): RunSummaryState {
  return {
    summary: {
      total: 0,
      latestRun: null,
      recentSuccessRate: null,
      averageDurationMs: null,
      ...summary,
    },
    isLoading: false,
    isError: false,
  };
}

describe("deriveAutomationHealth", () => {
  it.each<[string, Automation, RunSummaryState | undefined, AutomationHealth]>([
    [
      "disabled wins over a failed latest run",
      createAutomation({ enabled: false }),
      settled({ latestRun: createRun({ status: AutomationRunStatus.FAILED }) }),
      "disabled",
    ],
    [
      "an unsettled summary claims nothing",
      createAutomation(),
      { summary: null, isLoading: true, isError: false },
      "unknown",
    ],
    [
      "a summary that failed to load claims nothing",
      createAutomation(),
      { summary: null, isLoading: false, isError: true },
      "unknown",
    ],
    [
      "no runs yet means never-run",
      createAutomation(),
      settled({ latestRun: null }),
      "never-run",
    ],
    [
      "a failed latest run means failing",
      createAutomation(),
      settled({ latestRun: createRun({ status: AutomationRunStatus.FAILED }) }),
      "failing",
    ],
    [
      "a pending latest run means running",
      createAutomation(),
      settled({
        latestRun: createRun({ status: AutomationRunStatus.PENDING }),
      }),
      "running",
    ],
    [
      "a completed latest run means healthy",
      createAutomation(),
      settled({ latestRun: createRun() }),
      "healthy",
    ],
    [
      // Statuses the backend added after the dashboard's reference design:
      // neither a success nor a failure, so not "failing".
      "a cancelled latest run does not read as failing",
      createAutomation(),
      settled({
        latestRun: createRun({ status: AutomationRunStatus.CANCELLED }),
      }),
      "healthy",
    ],
  ])("%s", (_case, automation, state, expected) => {
    expect(deriveAutomationHealth(automation, state)).toBe(expected);
  });
});

describe("getAutomationHealthDetails", () => {
  it("surfaces the persisted reason for an auto-disabled automation", () => {
    const details = getAutomationHealthDetails(
      createAutomation({
        enabled: false,
        disabled_failure_kind: "config",
        disabled_reason: "Model profile is no longer available",
      }),
      settled({ latestRun: createRun({ status: AutomationRunStatus.FAILED }) }),
    );

    expect(details).toEqual({
      issue: "disabled",
      failureKind: "config",
      reason: "Model profile is no longer available",
    });
  });

  it("falls back to the latest run metadata when the disabled fields are absent", () => {
    const details = getAutomationHealthDetails(
      createAutomation({ enabled: false }),
      settled({
        latestRun: createRun({
          status: AutomationRunStatus.FAILED,
          failure_kind: "agent_action",
          blocking_reason: "The GitHub integration is not connected",
          error_detail: "The run could not continue",
        }),
      }),
    );

    expect(details).toEqual({
      issue: "disabled",
      failureKind: "agent_action",
      reason: "The GitHub integration is not connected",
    });
  });

  it("surfaces a blocking reason when a run reports completed", () => {
    const details = getAutomationHealthDetails(
      createAutomation(),
      settled({
        latestRun: createRun({
          status: AutomationRunStatus.COMPLETED,
          failure_kind: "agent_action",
          blocking_reason: "The GitHub integration is not connected",
        }),
      }),
    );

    expect(details).toEqual({
      issue: "blocked",
      failureKind: "agent_action",
      reason: "The GitHub integration is not connected",
    });
  });

  it("leaves legacy failures to the existing health badge", () => {
    const details = getAutomationHealthDetails(
      createAutomation(),
      settled({
        latestRun: createRun({
          status: AutomationRunStatus.FAILED,
          error_detail: "The provider returned an error",
        }),
      }),
    );

    expect(details).toEqual({
      issue: null,
      failureKind: null,
      reason: null,
    });
  });

  it.each([
    ["config", "blocked"],
    ["auth", "blocked"],
    ["quota", "blocked"],
    ["agent_action", "blocked"],
    ["transient", "transient"],
    ["rate_limit", "transient"],
  ] as const)("classifies %s as a %s failure", (failureKind, issue) => {
    const details = getAutomationHealthDetails(
      createAutomation(),
      settled({
        latestRun: createRun({
          status: AutomationRunStatus.FAILED,
          failure_kind: failureKind,
          blocking_reason:
            failureKind === "agent_action"
              ? "The integration is missing"
              : null,
          error_detail: "The provider returned an error",
        }),
      }),
    );

    expect(details.issue).toBe(issue);
  });

  it("does not invent an issue for old or non-failed run responses", () => {
    expect(
      getAutomationHealthDetails(
        createAutomation(),
        settled({ latestRun: createRun() }),
      ),
    ).toEqual({ issue: null, failureKind: null, reason: null });
    expect(getAutomationHealthDetails(createAutomation(), undefined)).toEqual({
      issue: null,
      failureKind: null,
      reason: null,
    });
  });
});

describe("formatAutomationFailureKind", () => {
  it("keeps unknown backend kinds readable", () => {
    expect(formatAutomationFailureKind("provider_timeout")).toBe(
      "Provider Timeout",
    );
  });
});

describe("formatCompactDuration", () => {
  it.each<[number | null, string]>([
    [null, "—"],
    [12_000, "12s"],
    [120_000, "2m"],
    [5_400_000, "1.5h"],
  ])("renders %s as %s", (ms, expected) => {
    expect(formatCompactDuration(ms)).toBe(expected);
  });
});

describe("summarizeAutomationRuns", () => {
  it("summarizes the sample and keeps the lifetime total", () => {
    // Arrange — newest first, mixing terminal and non-terminal statuses.
    const runs = [
      createRun({
        id: "run-latest",
        status: AutomationRunStatus.FAILED,
        started_at: "2026-01-05T00:00:00Z",
        completed_at: "2026-01-05T00:00:30Z",
      }),
      createRun({ status: AutomationRunStatus.CANCELLED, completed_at: null }),
      createRun({ status: AutomationRunStatus.SKIPPED, completed_at: null }),
      createRun({ status: AutomationRunStatus.RUNNING, completed_at: null }),
      createRun({
        started_at: "2026-01-01T00:00:00Z",
        completed_at: "2026-01-01T00:01:30Z",
      }),
    ];

    // Act
    const summary = summarizeAutomationRuns({ runs, total: 40 });

    // Assert — success rate and durations consider COMPLETED and FAILED only;
    // total is the response's lifetime count, not the sample's length.
    expect(summary).toEqual({
      total: 40,
      latestRun: runs[0],
      recentSuccessRate: 0.5,
      averageDurationMs: (30_000 + 90_000) / 2,
    });
  });

  it("reports no rate and no duration without terminal runs", () => {
    // Arrange
    const runs = [
      createRun({ status: AutomationRunStatus.RUNNING, completed_at: null }),
    ];

    // Act
    const summary = summarizeAutomationRuns({ runs, total: 1 });

    // Assert
    expect({
      rate: summary.recentSuccessRate,
      duration: summary.averageDurationMs,
    }).toEqual({ rate: null, duration: null });
  });
});

describe("matchesAutomationSearch", () => {
  it.each<[string, string]>([
    ["name", "widget MONITOR"],
    ["prompt", "watch the"],
    ["repository", "acme/"],
    ["model", "widget-profile"],
  ])("matches on %s, case-insensitively", (_field, query) => {
    expect(matchesAutomationSearch(createAutomation(), query)).toBe(true);
  });

  it("rejects an automation matching no field", () => {
    expect(matchesAutomationSearch(createAutomation(), "gadget")).toBe(false);
  });
});

describe("applyDashboardView", () => {
  const failing = createAutomation({ id: "failing", name: "B failing" });
  const healthy = createAutomation({ id: "healthy", name: "A healthy" });
  const evented = createAutomation({
    id: "evented",
    name: "C evented",
    trigger: { type: "event", source: "github" },
  });
  const all = [failing, healthy, evented];
  const byId = new Map<string, RunSummaryState>([
    [
      "failing",
      settled({
        total: 9,
        latestRun: createRun({
          status: AutomationRunStatus.FAILED,
          started_at: "2026-01-03T00:00:00Z",
        }),
      }),
    ],
    [
      "healthy",
      settled({
        total: 2,
        latestRun: createRun({ started_at: "2026-01-05T00:00:00Z" }),
      }),
    ],
    ["evented", settled({ total: 5, latestRun: null })],
  ]);
  const neutral = {
    search: "",
    status: "all",
    trigger: "all",
    sort: "name",
  } as const;

  it("keeps only latest-run failures under the failing status", () => {
    // Act
    const visible = applyDashboardView(
      all,
      { ...neutral, status: "failing" },
      byId,
    );

    // Assert
    expect(visible.map((a) => a.id)).toEqual(["failing"]);
  });

  it("treats every non-event trigger as a schedule", () => {
    // Act
    const scheduled = applyDashboardView(
      all,
      { ...neutral, trigger: "schedule" },
      byId,
    );
    const events = applyDashboardView(
      all,
      { ...neutral, trigger: "event" },
      byId,
    );

    // Assert
    expect({
      scheduled: scheduled.map((a) => a.id),
      events: events.map((a) => a.id),
    }).toEqual({ scheduled: ["healthy", "failing"], events: ["evented"] });
  });

  it("orders by lifetime run count under the runs sort", () => {
    // Act
    const visible = applyDashboardView(all, { ...neutral, sort: "runs" }, byId);

    // Assert
    expect(visible.map((a) => a.id)).toEqual(["failing", "evented", "healthy"]);
  });

  it("orders by newest activity under the last-run sort, falling back to last_triggered_at", () => {
    // Arrange — no summary for the third automation, only its own timestamp.
    const triggered = createAutomation({
      id: "triggered",
      name: "D triggered",
      last_triggered_at: "2026-01-04T00:00:00Z",
    });

    // Act
    const visible = applyDashboardView(
      [failing, healthy, triggered],
      { ...neutral, sort: "last-run" },
      byId,
    );

    // Assert
    expect(visible.map((a) => a.id)).toEqual([
      "healthy",
      "triggered",
      "failing",
    ]);
  });
});

describe("computeOverviewTile", () => {
  const enabled = createAutomation({ id: "on" });
  const disabled = createAutomation({ id: "off", enabled: false });
  const byId = new Map<string, RunSummaryState>([
    [
      "on",
      settled({
        total: 7,
        latestRun: createRun({ status: AutomationRunStatus.FAILED }),
        averageDurationMs: 30_000,
      }),
    ],
    [
      "off",
      settled({
        total: 3,
        latestRun: createRun({ status: AutomationRunStatus.FAILED }),
        averageDurationMs: 90_000,
      }),
    ],
  ]);

  it("counts automations and exposes the active count to tile copy", () => {
    // Act
    const value = computeOverviewTile("automations", [enabled, disabled], byId);

    // Assert
    expect(value).toEqual({
      display: "2",
      isZero: false,
      placeholderValues: { active: 1 },
    });
  });

  it("counts only failing automations for needs-attention, flagging zero", () => {
    // Act — the disabled automation's failed history must not count.
    const attention = computeOverviewTile(
      "needs-attention",
      [enabled, disabled],
      byId,
    );
    const calm = computeOverviewTile("needs-attention", [disabled], byId);

    // Assert
    expect({
      display: attention.display,
      calmIsZero: calm.isZero,
    }).toEqual({ display: "1", calmIsZero: true });
  });

  it("sums lifetime runs across loaded summaries", () => {
    // Act
    const value = computeOverviewTile("total-runs", [enabled, disabled], byId);

    // Assert
    expect(value.display).toBe("10");
  });

  it("withholds the run total until a summary has loaded", () => {
    // Act
    const value = computeOverviewTile("total-runs", [enabled], new Map());

    // Assert
    expect(value.display).toBe("—");
  });

  it("averages per-automation durations into a compact figure", () => {
    // Act
    const value = computeOverviewTile(
      "average-duration",
      [enabled, disabled],
      byId,
    );

    // Assert — mean of 30s and 90s.
    expect(value.display).toBe("1m");
  });
});
