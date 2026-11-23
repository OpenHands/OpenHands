import { useMutation, useQueryClient } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { AUTOMATION_KV_QUERY_KEY } from "#/hooks/query/use-automation-kv";
import type { KvChange } from "#/utils/automation-kv-drafts";

interface SaveAutomationKvVariables {
  changes: KvChange[];
  /** The store version the edits were made against. */
  ifVersion: number;
}

/** Errors are the dialog's to render: a 409 means the store changed under the edit. */
export function useSaveAutomationKv(automationId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ changes, ifVersion }: SaveAutomationKvVariables) =>
      AutomationService.setKvEntries(automationId, changes, ifVersion),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: [...AUTOMATION_KV_QUERY_KEY, automationId],
      }),
    meta: { disableToast: true },
  });
}
