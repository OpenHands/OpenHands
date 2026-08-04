import { useAutomationHealth } from "#/hooks/query/use-automation-health";
import { useAutomations } from "#/hooks/query/use-automations";
import {
  useLatestAutomationRuns,
  type LatestAutomationRunState,
} from "#/hooks/query/use-latest-automation-runs";
import {
  HOME_AUTOMATIONS_DEMO_AUTOMATIONS,
  HOME_AUTOMATIONS_DEMO_RUN_STATES,
  isHomeAutomationsDemoEnabled,
} from "#/fixtures/home-automations-demo";

/** Initial visible rows in the home automations list before "View more". */
export const HOME_AUTOMATIONS_PREVIEW_LIMIT = 10;

/** Bounds the per-automation latest-run request fan-out on the home page. */
export const MAX_HOME_AUTOMATION_CHIPS = 20;

export const UNKNOWN_RUN_STATE: LatestAutomationRunState = {
  latestRun: null,
  recentRuns: [],
  isLoading: true,
  isError: false,
};

/**
 * Shared home-page automation queries: health gate, enabled automations, and
 * latest-run state for the recent-activity list and pinned dashboard.
 */
export function useHomeAutomations() {
  const demo = isHomeAutomationsDemoEnabled();
  const { data: healthData, isLoading: isHealthLoading } =
    useAutomationHealth();
  const isBackendHealthy = healthData?.status === "ok";
  const {
    data: automationsData,
    isError,
    isLoading: isAutomationsLoading,
  } = useAutomations({
    limit: 50,
    offset: 0,
    enabled: !demo && isBackendHealthy,
  });

  const allAutomations = demo
    ? HOME_AUTOMATIONS_DEMO_AUTOMATIONS
    : (automationsData?.automations ?? []);

  const enabledAutomations = allAutomations
    .filter((automation) => automation.enabled)
    .slice(0, MAX_HOME_AUTOMATION_CHIPS);

  const knownAutomationIds = new Set(
    allAutomations.map((automation) => automation.id),
  );

  // False while loading or when the backend has more automations than the
  // fetched page — consumers must not prune pins against a partial id set.
  const isAutomationListComplete = demo
    ? true
    : automationsData != null && allAutomations.length >= automationsData.total;

  const liveRunStates = useLatestAutomationRuns(demo ? [] : enabledAutomations);
  const runStates = demo ? HOME_AUTOMATIONS_DEMO_RUN_STATES : liveRunStates;

  if (demo) {
    return {
      isBackendHealthy: true,
      isHealthLoading: false,
      isError: false,
      isAutomationsLoading: false,
      enabledAutomations,
      knownAutomationIds,
      isAutomationListComplete,
      runStates,
    };
  }

  return {
    isBackendHealthy,
    isHealthLoading,
    isError,
    isAutomationsLoading,
    enabledAutomations,
    knownAutomationIds,
    isAutomationListComplete,
    runStates,
  };
}
