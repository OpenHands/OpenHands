import type { LatestAutomationRunState } from "#/hooks/query/use-latest-automation-runs";
import {
  summarizeAutomationRuns,
  type RunSummaryState,
} from "#/manifests/automation-insights";

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
    total: state.summary?.total,
    isLoading: state.isLoading,
    isError: state.isError,
  };
}

/** Maps home run state onto the dashboard stats footer shape. */
export function toRunSummaryState(
  state: LatestAutomationRunState,
): RunSummaryState {
  if (state.recentRuns.length === 0 && (state.isLoading || state.isError)) {
    return {
      summary: null,
      isLoading: state.isLoading,
      isError: state.isError,
    };
  }
  return {
    summary: summarizeAutomationRuns({
      runs: state.recentRuns,
      total: state.total ?? state.recentRuns.length,
    }),
    isLoading: state.isLoading,
    isError: state.isError,
  };
}
