import { useQueries } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
  type AutomationRunsResponse,
} from "#/types/automation";
import { AUTOMATION_RUNS_QUERY_KEY } from "./use-automation-detail";

/**
 * Poll interval while a run is non-terminal. Deliberately slower than the 3s
 * used by `useAutomationRuns`: that hook polls a single automation on its
 * detail page, whereas this one issues one request *per automation* (up to
 * MAX_AUTOMATION_CHIPS). At 15s a fully in-flight home section costs about the
 * same request rate as one open detail page.
 */
const IN_FLIGHT_POLL_INTERVAL_MS = 15_000;

/** Recent runs fetched for the home activity sparkline (newest-first page size). */
export const AUTOMATION_RUN_ACTIVITY_LIMIT = 12;

export interface LatestAutomationRunState {
  /** Newest run of the automation, or null while loading / on error / when none exist. */
  latestRun: AutomationRun | null;
  /** Newest-first recent runs for the activity sparkline (may be empty). */
  recentRuns: AutomationRun[];
  isLoading: boolean;
  isError: boolean;
}

/**
 * Fetch recent runs for each automation (newest-first). The first item is the
 * latest run; the page feeds the home activity sparkline. Each query uses the
 * same key shape as `useAutomationRuns` (distinct `{limit, offset}` part,
 * shared `[...AUTOMATION_RUNS_QUERY_KEY, id]` prefix), so dispatch mutations
 * that invalidate an automation's runs reach these entries too.
 */
export function useLatestAutomationRuns(
  automations: readonly Automation[],
): Map<string, LatestAutomationRunState> {
  const active = useActiveBackend();

  const results = useQueries({
    queries: automations.map((automation) => ({
      queryKey: [
        ...AUTOMATION_RUNS_QUERY_KEY,
        automation.id,
        { limit: AUTOMATION_RUN_ACTIVITY_LIMIT, offset: 0 },
        active.backend.id,
        active.orgId,
      ],
      queryFn: () =>
        AutomationService.getAutomationRuns(
          automation.id,
          AUTOMATION_RUN_ACTIVITY_LIMIT,
          0,
        ),
      staleTime: 60 * 1000,
      // No retries: the home section settles into its degraded "unknown"
      // indicator instead of hammering an unhealthy automation service.
      retry: false,
      // One request per automation already fans out on mount; refetching the
      // whole set again on every tab focus is not worth the run-health delta.
      refetchOnWindowFocus: false,
      // Poll while the latest run is non-terminal so status and
      // conversation_id transitions appear without a manual refresh.
      refetchInterval: (query: {
        state: { data?: AutomationRunsResponse };
      }) => {
        const latest = query.state.data?.runs[0];
        const isInFlight =
          latest?.status === AutomationRunStatus.PENDING ||
          latest?.status === AutomationRunStatus.RUNNING;
        return isInFlight ? IN_FLIGHT_POLL_INTERVAL_MS : false;
      },
    })),
  });

  const runStates = new Map<string, LatestAutomationRunState>();
  automations.forEach((automation, i) => {
    const result = results[i];
    const recentRuns = result?.data?.runs ?? [];
    runStates.set(automation.id, {
      latestRun: recentRuns[0] ?? null,
      recentRuns,
      isLoading: result?.isPending ?? true,
      isError: result?.isError ?? false,
    });
  });
  return runStates;
}
