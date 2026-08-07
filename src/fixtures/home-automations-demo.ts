import {
  AUTOMATION_RUN_ACTIVITY_LIMIT,
  type LatestAutomationRunState,
} from "#/hooks/query/use-latest-automation-runs";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";

/** Must stay equal to `MAX_HOME_AUTOMATION_CHIPS` in use-home-automations. */
const DEMO_LIST_LENGTH = 20;

const DEMO_HISTORY_PATTERN: AutomationRunStatus[] = [
  AutomationRunStatus.COMPLETED,
  AutomationRunStatus.COMPLETED,
  AutomationRunStatus.FAILED,
  AutomationRunStatus.COMPLETED,
  AutomationRunStatus.CANCELLED,
  AutomationRunStatus.COMPLETED,
  AutomationRunStatus.COMPLETED,
  AutomationRunStatus.FAILED,
  AutomationRunStatus.COMPLETED,
  AutomationRunStatus.SKIPPED,
  AutomationRunStatus.COMPLETED,
];

/**
 * Local preview of every pinned-card + activity-list state.
 *
 * Enable with any of:
 * - `HOME_AUTOMATIONS_FORCE_DEMO = true` below
 * - `localStorage.setItem("oh:home-automations-demo", "1")`
 * - `?demoAutomations=1` in the URL
 * - `VITE_HOME_AUTOMATIONS_DEMO=true`
 */
export const HOME_AUTOMATIONS_FORCE_DEMO = false;

export const HOME_AUTOMATIONS_DEMO_KEY = "oh:home-automations-demo";

export function isHomeAutomationsDemoEnabled(): boolean {
  if (HOME_AUTOMATIONS_FORCE_DEMO) return true;
  if (import.meta.env.VITE_HOME_AUTOMATIONS_DEMO === "true") return true;
  if (typeof window === "undefined") return false;
  if (window.localStorage.getItem(HOME_AUTOMATIONS_DEMO_KEY) === "1") {
    return true;
  }
  try {
    return (
      new URLSearchParams(window.location.search).get("demoAutomations") === "1"
    );
  } catch {
    return false;
  }
}

const DEMO_CONVERSATION_TITLES: Record<string, string | null> = {
  "demo-conv-titled": "Fix flaky CI on main",
  "demo-conv-untitled": "Summarize open pull requests",
};

/** Returns a demo conversation title, or `undefined` if not a demo id. */
export function getDemoConversationTitle(
  conversationId: string,
): string | null | undefined {
  if (!(conversationId in DEMO_CONVERSATION_TITLES)) return undefined;
  return DEMO_CONVERSATION_TITLES[conversationId];
}

function hoursAgo(hours: number): string {
  return new Date(Date.now() - hours * 3_600_000).toISOString();
}

function makeAutomation(
  id: string,
  name: string,
  trigger: Automation["trigger"],
): Automation {
  return {
    id,
    name,
    trigger,
    enabled: true,
    created_at: hoursAgo(720),
    updated_at: hoursAgo(24),
    prompt: "demo",
  };
}

function makeRun(
  id: string,
  status: AutomationRunStatus,
  options: {
    conversationId?: string | null;
    errorDetail?: string | null;
    startedHoursAgo?: number;
    completedHoursAgo?: number | null;
  } = {},
): AutomationRun {
  const startedHoursAgo = options.startedHoursAgo ?? 2;
  const completedHoursAgo =
    options.completedHoursAgo === undefined ? 1 : options.completedHoursAgo;
  return {
    id,
    status,
    conversation_id:
      options.conversationId === undefined ? null : options.conversationId,
    bash_command_id: null,
    error_detail:
      options.errorDetail === undefined ? null : options.errorDetail,
    started_at: hoursAgo(startedHoursAgo),
    completed_at:
      completedHoursAgo === null ? null : hoursAgo(completedHoursAgo),
  };
}

type DemoRunStateInput = Omit<LatestAutomationRunState, "recentRuns"> & {
  recentRuns?: AutomationRun[];
};

type DemoSpec = {
  automation: Automation;
  runState: DemoRunStateInput;
  /** When true, this automation is pinned in the dashboard. */
  pin?: boolean;
};

function buildDemoRecentRuns(latestRun: AutomationRun | null): AutomationRun[] {
  if (!latestRun) return [];
  // Varied durations (minutes) so sparkline bar heights differ in demo mode.
  const durationMinutes = [2, 8, 45, 3, 15, 1, 25, 90, 5, 12, 6];
  const older = DEMO_HISTORY_PATTERN.slice(
    0,
    AUTOMATION_RUN_ACTIVITY_LIMIT - 1,
  ).map((status, index) => {
    const startedHoursAgo = 24 + index * 6;
    const durationHours = (durationMinutes[index] ?? 5) / 60;
    return makeRun(`${latestRun.id}-hist-${index}`, status, {
      startedHoursAgo,
      completedHoursAgo: startedHoursAgo - durationHours,
      conversationId: null,
      errorDetail:
        status === AutomationRunStatus.FAILED ? "Prior failure" : null,
    });
  });
  return [latestRun, ...older];
}

function toDemoRunState(state: DemoRunStateInput): LatestAutomationRunState {
  return {
    ...state,
    recentRuns: state.recentRuns ?? buildDemoRecentRuns(state.latestRun),
  };
}

/**
 * Explicit card states first (pinned), then fillers to hit
 * DEMO_LIST_LENGTH for list-length stress.
 */
const DEMO_SPECS: DemoSpec[] = [
  {
    pin: true,
    automation: makeAutomation("demo-pin-loading", "Card: loading skeleton", {
      type: "cron",
      schedule: "0 * * * *",
      schedule_human: "Every hour",
    }),
    runState: { latestRun: null, isLoading: true, isError: false },
  },
  {
    pin: true,
    automation: makeAutomation("demo-pin-error", "Card: status unavailable", {
      type: "cron",
      schedule: "0 9 * * *",
      schedule_human: "Daily at 9am",
    }),
    runState: { latestRun: null, isLoading: false, isError: true },
  },
  {
    pin: true,
    automation: makeAutomation("demo-pin-no-runs", "Card: no runs yet", {
      type: "event",
      source: "github",
      on: "pull_request.opened",
    }),
    runState: { latestRun: null, isLoading: false, isError: false },
  },
  {
    pin: true,
    automation: makeAutomation(
      "demo-pin-failed-detail",
      "Card: failed with error detail",
      {
        type: "cron",
        schedule: "*/15 * * * *",
        schedule_human: "Every 15 min",
      },
    ),
    runState: {
      latestRun: makeRun("run-failed-detail", AutomationRunStatus.FAILED, {
        errorDetail:
          "Sandbox provisioning timed out after 120s while waiting for the runtime to become ready.",
        startedHoursAgo: 3,
        completedHoursAgo: 2.5,
      }),
      isLoading: false,
      isError: false,
    },
  },
  {
    pin: true,
    automation: makeAutomation(
      "demo-pin-completed-title",
      "Card: completed + conversation title",
      { type: "event", source: "github", on: "push" },
    ),
    runState: {
      latestRun: makeRun("run-completed-title", AutomationRunStatus.COMPLETED, {
        conversationId: "demo-conv-titled",
        startedHoursAgo: 5,
        completedHoursAgo: 4,
      }),
      isLoading: false,
      isError: false,
    },
  },
  {
    pin: true,
    automation: makeAutomation(
      "demo-pin-completed-untitled",
      "Card: completed + conversation title",
      {
        type: "cron",
        schedule: "0 0 * * 1",
        schedule_human: "Mondays at midnight",
      },
    ),
    runState: {
      latestRun: makeRun(
        "run-completed-untitled",
        AutomationRunStatus.COMPLETED,
        {
          conversationId: "demo-conv-untitled",
          startedHoursAgo: 8,
          completedHoursAgo: 7,
        },
      ),
      isLoading: false,
      isError: false,
    },
  },
  {
    pin: true,
    automation: makeAutomation(
      "demo-pin-completed-no-conv",
      "Card: completed, no conversation",
      { type: "cron", schedule: "0 12 * * *", schedule_human: "Daily at noon" },
    ),
    runState: {
      latestRun: makeRun(
        "run-completed-no-conv",
        AutomationRunStatus.COMPLETED,
        {
          conversationId: null,
          startedHoursAgo: 10,
          completedHoursAgo: 9,
        },
      ),
      isLoading: false,
      isError: false,
    },
  },
  {
    pin: true,
    automation: makeAutomation(
      "demo-pin-failed-no-conv",
      "Card: failed, no conversation",
      { type: "event", source: "slack", on: "message" },
    ),
    runState: {
      latestRun: makeRun("run-failed-no-conv", AutomationRunStatus.FAILED, {
        conversationId: null,
        errorDetail: null,
        startedHoursAgo: 12,
        completedHoursAgo: 11,
      }),
      isLoading: false,
      isError: false,
    },
  },
  {
    pin: true,
    automation: makeAutomation(
      "demo-pin-pending",
      "Card: pending (empty result panel)",
      {
        type: "cron",
        schedule: "0 */6 * * *",
        schedule_human: "Every 6 hours",
      },
    ),
    runState: {
      latestRun: makeRun("run-pending", AutomationRunStatus.PENDING, {
        conversationId: null,
        startedHoursAgo: 0.1,
        completedHoursAgo: null,
      }),
      isLoading: false,
      isError: false,
    },
  },
  {
    pin: true,
    automation: makeAutomation(
      "demo-pin-running",
      "Card: running (empty result panel)",
      { type: "event", source: "github", on: "issues.opened" },
    ),
    runState: {
      latestRun: makeRun("run-running", AutomationRunStatus.RUNNING, {
        // No conversation yet — status strip shows the "Running" label.
        conversationId: null,
        startedHoursAgo: 0.25,
        completedHoursAgo: null,
      }),
      isLoading: false,
      isError: false,
    },
  },
  // Remaining rows fill the list to MAX_HOME_AUTOMATION_CHIPS.
  {
    automation: makeAutomation("demo-list-cancelled", "List: cancelled run", {
      type: "cron",
      schedule: "30 2 * * *",
      schedule_human: "Daily at 2:30am",
    }),
    runState: {
      latestRun: makeRun("run-cancelled", AutomationRunStatus.CANCELLED, {
        startedHoursAgo: 14,
        completedHoursAgo: 13,
      }),
      isLoading: false,
      isError: false,
    },
  },
  {
    automation: makeAutomation("demo-list-skipped", "List: skipped run", {
      type: "event",
      source: "github",
      on: "pull_request.synchronize",
    }),
    runState: {
      latestRun: makeRun("run-skipped", AutomationRunStatus.SKIPPED, {
        startedHoursAgo: 16,
        completedHoursAgo: 16,
      }),
      isLoading: false,
      isError: false,
    },
  },
  {
    automation: makeAutomation(
      "demo-list-running-2",
      "List: second in-flight run",
      { type: "cron", schedule: "*/5 * * * *", schedule_human: "Every 5 min" },
    ),
    runState: {
      latestRun: makeRun("run-running-2", AutomationRunStatus.RUNNING, {
        conversationId: null,
        startedHoursAgo: 0.5,
        completedHoursAgo: null,
      }),
      isLoading: false,
      isError: false,
    },
  },
  {
    automation: makeAutomation("demo-list-success-old", "List: older success", {
      type: "cron",
      schedule: "0 8 * * 1-5",
      schedule_human: "Weekdays at 8am",
    }),
    runState: {
      latestRun: makeRun("run-success-old", AutomationRunStatus.COMPLETED, {
        conversationId: "demo-conv-titled",
        startedHoursAgo: 48,
        completedHoursAgo: 47,
      }),
      isLoading: false,
      isError: false,
    },
  },
  {
    automation: makeAutomation("demo-list-failed-old", "List: older failure", {
      type: "event",
      source: "github",
      on: "release.published",
    }),
    runState: {
      latestRun: makeRun("run-failed-old", AutomationRunStatus.FAILED, {
        errorDetail: "Model returned an empty completion.",
        startedHoursAgo: 72,
        completedHoursAgo: 71,
      }),
      isLoading: false,
      isError: false,
    },
  },
  {
    automation: makeAutomation("demo-list-no-runs-2", "List: never ran", {
      type: "cron",
      schedule: "0 0 1 * *",
      schedule_human: "Monthly on the 1st",
    }),
    runState: { latestRun: null, isLoading: false, isError: false },
  },
  {
    automation: makeAutomation("demo-list-loading-2", "List: loading status", {
      type: "event",
      source: "linear",
      on: "Issue.create",
    }),
    runState: { latestRun: null, isLoading: true, isError: false },
  },
  {
    automation: makeAutomation("demo-list-error-2", "List: fetch error", {
      type: "cron",
      schedule: "15 * * * *",
      schedule_human: "Hourly at :15",
    }),
    runState: { latestRun: null, isLoading: false, isError: true },
  },
  {
    automation: makeAutomation(
      "demo-list-pending-2",
      "List: pending, long name that should truncate in the activity row when the viewport is narrow",
      { type: "cron", schedule: "0 3 * * 0", schedule_human: "Sundays at 3am" },
    ),
    runState: {
      latestRun: makeRun("run-pending-2", AutomationRunStatus.PENDING, {
        startedHoursAgo: 0.05,
        completedHoursAgo: null,
      }),
      isLoading: false,
      isError: false,
    },
  },
  {
    automation: makeAutomation(
      "demo-list-completed-fallback",
      "List: completed with conversation",
      { type: "event", source: "github", on: "workflow_run.completed" },
    ),
    runState: {
      latestRun: makeRun(
        "run-completed-fallback",
        AutomationRunStatus.COMPLETED,
        {
          conversationId: "demo-conv-untitled",
          startedHoursAgo: 20,
          completedHoursAgo: 19,
        },
      ),
      isLoading: false,
      isError: false,
    },
  },
];

if (DEMO_SPECS.length !== DEMO_LIST_LENGTH) {
  throw new Error(
    `home-automations-demo: expected ${DEMO_LIST_LENGTH} specs, got ${DEMO_SPECS.length}`,
  );
}

export const HOME_AUTOMATIONS_DEMO_PINNED_IDS: string[] = DEMO_SPECS.filter(
  (spec) => spec.pin,
).map((spec) => spec.automation.id);

export const HOME_AUTOMATIONS_DEMO_AUTOMATIONS: Automation[] = DEMO_SPECS.map(
  (spec) => spec.automation,
);

export const HOME_AUTOMATIONS_DEMO_RUN_STATES: Map<
  string,
  LatestAutomationRunState
> = new Map(
  DEMO_SPECS.map(
    (spec) => [spec.automation.id, toDemoRunState(spec.runState)] as const,
  ),
);
