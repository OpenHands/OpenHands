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

export interface LatestAutomationRunState {
  /** Newest run of the automation, or null while loading / on error / when none exist. */
  latestRun: AutomationRun | null;
  isLoading: boolean;
  isError: boolean;
}

/**
 * Fetch the latest run for each automation. The runs endpoint returns runs
 * newest-first, so a limit-1 page is exactly the latest run. Each query uses
 * the same key shape as `useAutomationRuns` (distinct `{limit, offset}` part,
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
        { limit: 1, offset: 0 },
        active.backend.id,
        active.orgId,
      ],
      queryFn: () => AutomationService.getAutomationRuns(automation.id, 1, 0),
      staleTime: 60 * 1000,
      // No retries: the home section settles into its degraded "unknown"
      // indicator instead of hammering an unhealthy automation service.
      retry: false,
      // Poll while the latest run is non-terminal so status and
      // conversation_id transitions appear without a manual refresh.
      refetchInterval: (query: {
        state: { data?: AutomationRunsResponse };
      }) => {
        const latest = query.state.data?.runs[0];
        const isInFlight =
          latest?.status === AutomationRunStatus.PENDING ||
          latest?.status === AutomationRunStatus.RUNNING;
        return isInFlight ? 3000 : false;
      },
    })),
  });

  const runStates = new Map<string, LatestAutomationRunState>();
  automations.forEach((automation, i) => {
    const result = results[i];
    runStates.set(automation.id, {
      latestRun: result?.data?.runs[0] ?? null,
      isLoading: result?.isPending ?? true,
      isError: result?.isError ?? false,
    });
  });
  return runStates;
}
