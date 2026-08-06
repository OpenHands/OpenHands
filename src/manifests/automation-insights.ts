/**
 * The dashboard's automation knowledge, and the only module beside
 * `automation-setup.ts` that has any.
 *
 * The interface manifest's dashboard surface names metrics, filter predicates,
 * and sort comparators from closed sets; the one implementation of each name
 * is here, over the automation service's list and runs responses, so the rest
 * of `src/manifests/` stays about composition rather than about automations.
 *
 * Summaries are derived from the newest page of runs (the sample the runs
 * hook fetches); `total` alone is the response's lifetime count. Success rate
 * and durations consider only COMPLETED and FAILED runs — a cancelled or
 * skipped run says nothing about either.
 */

import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
  type AutomationRunsResponse,
} from "#/types/automation";
import type {
  DashboardSortValue,
  DashboardStatusValue,
  DashboardTriggerValue,
  InterfaceListInsights,
  OverviewMetric,
} from "./types";

export type AutomationHealth =
  | "healthy"
  | "failing"
  | "running"
  | "disabled"
  | "never-run"
  | "unknown";

/** Which caption of the manifest's `insights.health` block names each state. */
export const HEALTH_LABEL_KEYS: Record<
  AutomationHealth,
  keyof InterfaceListInsights["health"]
> = {
  healthy: "healthy",
  failing: "failing",
  running: "running",
  disabled: "disabled",
  "never-run": "neverRun",
  unknown: "checking",
};

export interface AutomationRunSummary {
  /** Lifetime run count, from the response — not the sample's length. */
  total: number;
  latestRun: AutomationRun | null;
  /** COMPLETED over COMPLETED+FAILED in the sample. Null with no terminal runs. */
  recentSuccessRate: number | null;
  /** Mean completed_at − started_at over the sample's terminal runs. */
  averageDurationMs: number | null;
}

/** One automation's summary alongside its query state. */
export interface RunSummaryState {
  summary: AutomationRunSummary | null;
  isLoading: boolean;
  isError: boolean;
}

type Summaries = ReadonlyMap<string, RunSummaryState>;

const TERMINAL_STATUSES = new Set<AutomationRunStatus>([
  AutomationRunStatus.COMPLETED,
  AutomationRunStatus.FAILED,
]);

export function summarizeAutomationRuns(
  response: AutomationRunsResponse,
): AutomationRunSummary {
  const terminal = response.runs.filter((run) =>
    TERMINAL_STATUSES.has(run.status),
  );
  const completed = terminal.filter(
    (run) => run.status === AutomationRunStatus.COMPLETED,
  ).length;

  const durations = terminal.flatMap((run) => {
    if (!run.completed_at) return [];
    const ms =
      new Date(run.completed_at).getTime() - new Date(run.started_at).getTime();
    return Number.isFinite(ms) && ms >= 0 ? [ms] : [];
  });

  return {
    total: response.total,
    latestRun: response.runs[0] ?? null,
    recentSuccessRate:
      terminal.length === 0 ? null : completed / terminal.length,
    averageDurationMs:
      durations.length === 0
        ? null
        : durations.reduce((sum, ms) => sum + ms, 0) / durations.length,
  };
}

/**
 * A disabled automation is disabled no matter its history; while the summary
 * is unsettled nothing is claimed; after that the latest run speaks.
 */
export function deriveAutomationHealth(
  automation: Automation,
  state: RunSummaryState | undefined,
): AutomationHealth {
  if (!automation.enabled) return "disabled";
  if (!state || state.isError || state.isLoading || !state.summary) {
    return "unknown";
  }
  const latest = state.summary.latestRun;
  if (!latest) return "never-run";
  if (latest.status === AutomationRunStatus.FAILED) return "failing";
  if (
    latest.status === AutomationRunStatus.PENDING ||
    latest.status === AutomationRunStatus.RUNNING
  ) {
    return "running";
  }
  return "healthy";
}

/** "—" unknown, seconds under a minute, minutes under an hour, else "1.5h". */
export function formatCompactDuration(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) return "—";
  if (ms < 60_000) return `${Math.max(1, Math.round(ms / 1000))}s`;
  if (ms < 3_600_000) return `${Math.max(1, Math.round(ms / 60_000))}m`;
  return `${(ms / 3_600_000).toFixed(1)}h`;
}

/** The list page's search predicate: name, prompt, repository, or model. */
export function matchesAutomationSearch(
  automation: Automation,
  query: string,
): boolean {
  const q = query.toLowerCase();
  if (!q) return true;
  return (
    automation.name.toLowerCase().includes(q) ||
    (automation.prompt ?? "").toLowerCase().includes(q) ||
    automation.repository?.toLowerCase().includes(q) === true ||
    automation.model?.toLowerCase().includes(q) === true
  );
}

const STATUS_PREDICATES: Record<
  Exclude<DashboardStatusValue, "all">,
  (automation: Automation, health: AutomationHealth) => boolean
> = {
  active: (automation) => automation.enabled,
  failing: (_automation, health) => health === "failing",
  disabled: (automation) => !automation.enabled,
};

const TRIGGER_PREDICATES: Record<
  Exclude<DashboardTriggerValue, "all">,
  (automation: Automation) => boolean
> = {
  // The backend emits several scheduled-trigger aliases, so "event" is the
  // one exact kind and everything else is a schedule.
  event: (automation) => automation.trigger.type === "event",
  schedule: (automation) => automation.trigger.type !== "event",
};

function runCount(automation: Automation, byId: Summaries): number {
  return byId.get(automation.id)?.summary?.total ?? 0;
}

function lastRunTime(automation: Automation, byId: Summaries): number {
  const startedAt =
    byId.get(automation.id)?.summary?.latestRun?.started_at ??
    automation.last_triggered_at;
  if (!startedAt) return 0;
  const time = new Date(startedAt).getTime();
  return Number.isFinite(time) ? time : 0;
}

const SORT_COMPARATORS: Record<
  DashboardSortValue,
  (a: Automation, b: Automation, byId: Summaries) => number
> = {
  "last-run": (a, b, byId) => lastRunTime(b, byId) - lastRunTime(a, byId),
  runs: (a, b, byId) => runCount(b, byId) - runCount(a, byId),
  name: (a, b) => a.name.localeCompare(b.name),
};

export interface DashboardViewState {
  search: string;
  status: DashboardStatusValue;
  trigger: DashboardTriggerValue;
  sort: DashboardSortValue;
}

/** The automations the dashboard shows, filtered and ordered. */
export function applyDashboardView(
  automations: readonly Automation[],
  view: DashboardViewState,
  byId: Summaries,
): Automation[] {
  return automations
    .filter((automation) => {
      if (!matchesAutomationSearch(automation, view.search)) return false;
      if (
        view.status !== "all" &&
        !STATUS_PREDICATES[view.status](
          automation,
          deriveAutomationHealth(automation, byId.get(automation.id)),
        )
      ) {
        return false;
      }
      if (
        view.trigger !== "all" &&
        !TRIGGER_PREDICATES[view.trigger](automation)
      ) {
        return false;
      }
      return true;
    })
    .sort((a, b) => SORT_COMPARATORS[view.sort](a, b, byId));
}

export interface OverviewTileValue {
  /** The formatted value the tile displays. */
  display: string;
  /** True when the metric's value is zero, which swaps in `zeroDetail`. */
  isZero: boolean;
  /** Values the tile's copy may substitute via `{{name}}`. */
  placeholderValues: Record<string, string | number>;
}

function loadedSummaries(
  automations: readonly Automation[],
  byId: Summaries,
): AutomationRunSummary[] {
  return automations.flatMap((automation) => {
    const state = byId.get(automation.id);
    return state?.summary && !state.isLoading && !state.isError
      ? [state.summary]
      : [];
  });
}

const TILE_COMPUTATIONS: Record<
  OverviewMetric,
  (automations: readonly Automation[], byId: Summaries) => OverviewTileValue
> = {
  automations: (automations) => ({
    display: automations.length.toLocaleString(),
    isZero: automations.length === 0,
    placeholderValues: {
      active: automations.filter((automation) => automation.enabled).length,
    },
  }),
  "needs-attention": (automations, byId) => {
    const failing = automations.filter(
      (automation) =>
        deriveAutomationHealth(automation, byId.get(automation.id)) ===
        "failing",
    ).length;
    return {
      display: failing.toLocaleString(),
      isZero: failing === 0,
      placeholderValues: {},
    };
  },
  "total-runs": (automations, byId) => {
    const loaded = loadedSummaries(automations, byId);
    if (loaded.length === 0 && automations.length > 0) {
      return { display: "—", isZero: false, placeholderValues: {} };
    }
    const total = loaded.reduce((sum, summary) => sum + summary.total, 0);
    return {
      display: total.toLocaleString(),
      isZero: total === 0,
      placeholderValues: {},
    };
  },
  "average-duration": (automations, byId) => {
    const durations = loadedSummaries(automations, byId).flatMap((summary) =>
      summary.averageDurationMs === null ? [] : [summary.averageDurationMs],
    );
    const mean =
      durations.length === 0
        ? null
        : durations.reduce((sum, ms) => sum + ms, 0) / durations.length;
    return {
      display: formatCompactDuration(mean),
      isZero: false,
      placeholderValues: {},
    };
  },
};

/** The value behind one manifest-declared tile. */
export function computeOverviewTile(
  metric: OverviewMetric,
  automations: readonly Automation[],
  byId: Summaries,
): OverviewTileValue {
  return TILE_COMPUTATIONS[metric](automations, byId);
}
