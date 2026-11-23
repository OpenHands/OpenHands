import { useQuery } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { AUTOMATION_KV_KEY_LIMIT } from "#/constants/automation-kv";
import type { AutomationKvEntry } from "#/types/automation";
import { automationKvRequestsLimiter } from "./concurrency-limiter";

export const AUTOMATION_KV_QUERY_KEY = ["automation-kv"] as const;

export interface AutomationKvSnapshot {
  entries: AutomationKvEntry[];
  total: number;
}

/**
 * The first `AUTOMATION_KV_KEY_LIMIT` keys of an automation's key-value store
 * with their values. Runs write the store, so a snapshot is never reused: the
 * viewer refetches each time it opens.
 */
export function useAutomationKv(automationId: string, enabled: boolean) {
  const active = useActiveBackend();
  return useQuery({
    queryKey: [
      ...AUTOMATION_KV_QUERY_KEY,
      automationId,
      active.backend.id,
      active.orgId,
    ],
    queryFn: async (): Promise<AutomationKvSnapshot> => {
      const { keys, total } = await AutomationService.listKvKeys(automationId, {
        limit: AUTOMATION_KV_KEY_LIMIT,
      });
      const entries = await Promise.all(
        keys.map((key) =>
          automationKvRequestsLimiter.run(() =>
            AutomationService.getKvEntry(automationId, key),
          ),
        ),
      );
      return { entries, total };
    },
    enabled: enabled && !!automationId,
    staleTime: 0,
    gcTime: 0,
    retry: false,
    meta: { disableToast: true },
  });
}
