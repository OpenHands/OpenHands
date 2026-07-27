import { useQuery } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

export const AUTOMATION_HEALTH_QUERY_KEY = ["automation-health"] as const;

export function useAutomationHealth() {
  const active = useActiveBackend();
  return useQuery({
    queryKey: [...AUTOMATION_HEALTH_QUERY_KEY, active.backend.id, active.orgId],
    queryFn: () => AutomationService.checkHealth(),
    staleTime: 30 * 1000, // 30 seconds
    retry: false, // Don't retry on failure - we want to show the error state immediately
  });
}
