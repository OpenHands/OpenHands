import { useQuery } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  AutomationRunStatus,
  type AutomationRunsResponse,
} from "#/types/automation";

export const AUTOMATION_DETAIL_QUERY_KEY = ["automation-detail"] as const;
export const AUTOMATION_RUNS_QUERY_KEY = ["automation-runs"] as const;

interface UseAutomationDetailOptions {
  id: string;
  enabled?: boolean;
}

export function useAutomationDetail(options: UseAutomationDetailOptions) {
  const { id, enabled = true } = options;
  const active = useActiveBackend();
  return useQuery({
    queryKey: [
      ...AUTOMATION_DETAIL_QUERY_KEY,
      id,
      active.backend.id,
      active.orgId,
    ],
    queryFn: () => AutomationService.getAutomation(id),
    staleTime: 5 * 60 * 1000,
    enabled: !!id && enabled,
  });
}

interface UseAutomationRunsOptions {
  id: string;
  limit?: number;
  offset?: number;
  enabled?: boolean;
}

export function useAutomationRuns(options: UseAutomationRunsOptions) {
  const { id, limit = 20, offset = 0, enabled = true } = options;
  const active = useActiveBackend();
  return useQuery({
    queryKey: [
      ...AUTOMATION_RUNS_QUERY_KEY,
      id,
      { limit, offset },
      active.backend.id,
      active.orgId,
    ],
    queryFn: () => AutomationService.getAutomationRuns(id, limit, offset),
    staleTime: 60 * 1000,
    enabled: !!id && enabled,
    // Poll while any run is non-terminal so status and conversation_id
    // transitions appear without a manual refresh.
    refetchInterval: (query) => {
      const data = query.state.data as AutomationRunsResponse | undefined;
      if (!data) return false;
      const hasInFlightRun = data.runs.some(
        (run) =>
          run.status === AutomationRunStatus.PENDING ||
          run.status === AutomationRunStatus.RUNNING,
      );
      return hasInFlightRun ? 3000 : false;
    },
  });
}
