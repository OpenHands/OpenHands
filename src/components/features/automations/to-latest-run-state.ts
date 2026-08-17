import type { LatestAutomationRunState } from "#/hooks/query/use-latest-automation-runs";
import type { RunSummaryState } from "#/manifests/automation-insights";

const EMPTY_RUN_STATE: LatestAutomationRunState = {
  latestRun: null,
  recentRuns: [],
  isLoading: false,
  isError: false,
};

/** Maps dashboard run-summary query state onto the home card/row run shape. */
export function toLatestRunState(
  state: RunSummaryState | undefined,
): LatestAutomationRunState {
  if (!state) return EMPTY_RUN_STATE;
  return {
    latestRun: state.summary?.latestRun ?? null,
    recentRuns: state.summary?.recentRuns ?? [],
    isLoading: state.isLoading,
    isError: state.isError,
  };
}
