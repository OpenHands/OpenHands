import { useAutomationHealth } from "#/hooks/query/use-automation-health";
import { useAutomations } from "#/hooks/query/use-automations";
import {
  useLatestAutomationRuns,
  type LatestAutomationRunState,
} from "#/hooks/query/use-latest-automation-runs";

/** Bounds the per-automation latest-run request fan-out on the home page. */
export const MAX_HOME_AUTOMATION_CHIPS = 20;

export const UNKNOWN_RUN_STATE: LatestAutomationRunState = {
  latestRun: null,
  isLoading: true,
  isError: false,
};

/**
 * Shared home-page automation queries: health gate, enabled automations, and
 * latest-run state for the recent-activity list and pinned dashboard.
 */
export function useHomeAutomations() {
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
    enabled: isBackendHealthy,
  });

  const enabledAutomations = (automationsData?.automations ?? [])
    .filter((automation) => automation.enabled)
    .slice(0, MAX_HOME_AUTOMATION_CHIPS);

  const runStates = useLatestAutomationRuns(enabledAutomations);

  return {
    isBackendHealthy,
    isHealthLoading,
    isError,
    isAutomationsLoading,
    enabledAutomations,
    runStates,
  };
}
