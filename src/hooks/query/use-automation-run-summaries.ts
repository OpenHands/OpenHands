import { useQueries } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { automationRunRequestsLimiter } from "#/hooks/query/concurrency-limiter";
import { AUTOMATION_RUNS_QUERY_KEY } from "#/hooks/query/use-automation-detail";
import {
  summarizeAutomationRuns,
  type RunSummaryState,
} from "#/manifests/automation-insights";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
  type AutomationRunsResponse,
} from "#/types/automation";

/**
 * The newest runs sampled per automation. Matches the detail page's default
 * page, so both surfaces share one cache entry per automation.
 */
const RECENT_RUN_SAMPLE_SIZE = 20;

/**
 * Poll interval while the newest run is non-terminal, matching
 * `useLatestAutomationRuns`: one request per listed automation, so a slower
 * cadence than the detail page's 3s. Without it the dashboard's phase and its
 * age freeze at whatever the first fetch saw, and a healthy run moving
 * through its phases reads as one stuck in the first — the opposite of what
 * the age is for.
 */
const IN_FLIGHT_POLL_INTERVAL_MS = 15_000;

const isInFlight = (run: AutomationRun) =>
  run.status === AutomationRunStatus.PENDING ||
  run.status === AutomationRunStatus.RUNNING;

interface UseAutomationRunSummariesOptions {
  enabled?: boolean;
}

/**
 * One runs query per listed automation — a deliberate fan-out, bounded by the
 * list's page size. Summaries drive the dashboard's tiles, health badges,
 * filters, and sorts.
 */
export function useAutomationRunSummaries(
  automations: readonly Automation[],
  options: UseAutomationRunSummariesOptions = {},
): Map<string, RunSummaryState> {
  const { enabled = true } = options;
  const active = useActiveBackend();

  return useQueries({
    queries: automations.map((automation) => ({
      queryKey: [
        ...AUTOMATION_RUNS_QUERY_KEY,
        automation.id,
        { limit: RECENT_RUN_SAMPLE_SIZE, offset: 0 },
        active.backend.id,
        active.orgId,
      ],
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        automationRunRequestsLimiter.run(
          () =>
            AutomationService.getAutomationRuns(
              automation.id,
              RECENT_RUN_SAMPLE_SIZE,
              0,
            ),
          signal,
        ),
      staleTime: 60 * 1000,
      // No retries and no focus refetch: a failed summary settles into the
      // dashboard's degraded "unknown" health state instead of hammering an
      // unhealthy automation service (mirrors useLatestAutomationRuns).
      retry: false,
      refetchOnWindowFocus: false,
      // isError already renders as the degraded indicator; a failing fan-out
      // must not raise one global error toast per automation.
      meta: { disableToast: true },
      enabled: enabled && !!automation.id,
      refetchInterval: (query: {
        state: { data?: AutomationRunsResponse };
      }) => {
        // The newest run only, which is the one the dashboard renders.
        // `.some()` over the whole sample keeps polling forever when an
        // older run was left non-terminal by a crashed dispatcher, and
        // every one of those requests changes nothing on screen.
        const latest = query.state.data?.runs?.[0];
        return latest && isInFlight(latest)
          ? IN_FLIGHT_POLL_INTERVAL_MS
          : false;
      },
    })),
    combine: (results) => {
      const byId = new Map<string, RunSummaryState>();
      automations.forEach((automation, index) => {
        const result = results[index];
        byId.set(automation.id, {
          summary: result.data ? summarizeAutomationRuns(result.data) : null,
          isLoading: result.isLoading,
          isError: result.isError,
        });
      });
      return byId;
    },
  });
}
