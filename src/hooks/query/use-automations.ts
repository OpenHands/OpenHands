import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useTracking } from "#/hooks/use-tracking";
import type { Automation, AutomationSpec } from "#/types/automation";
import {
  AUTOMATION_DETAIL_QUERY_KEY,
  AUTOMATION_RUNS_QUERY_KEY,
} from "./use-automation-detail";

export const AUTOMATIONS_QUERY_KEY = ["automations"] as const;

interface UseAutomationsOptions {
  limit?: number;
  offset?: number;
  enabled?: boolean;
}

export function useAutomations(options: UseAutomationsOptions = {}) {
  const { limit = 50, offset = 0, enabled = true } = options;
  const active = useActiveBackend();
  return useQuery({
    queryKey: [
      ...AUTOMATIONS_QUERY_KEY,
      { limit, offset },
      active.backend.id,
      active.orgId,
    ],
    queryFn: () => AutomationService.getAutomations(limit, offset),
    staleTime: 0,
    enabled,
  });
}

export function useToggleAutomation() {
  const queryClient = useQueryClient();
  const active = useActiveBackend();
  const { trackAutomationDisableButton } = useTracking();
  return useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) =>
      AutomationService.toggleAutomation(id, enabled),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: AUTOMATIONS_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: AUTOMATION_DETAIL_QUERY_KEY });
      if (!variables.enabled) {
        trackAutomationDisableButton({ backendKind: active.backend.kind });
      }
    },
  });
}

export function useImportAutomation() {
  const queryClient = useQueryClient();
  const active = useActiveBackend();
  const { trackAutomationImported } = useTracking();
  return useMutation({
    mutationFn: (spec: AutomationSpec) =>
      AutomationService.createAutomation({ ...spec, enabled: false }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AUTOMATIONS_QUERY_KEY });
      trackAutomationImported({ backendKind: active.backend.kind });
    },
  });
}

export function useUpdateAutomation() {
  const queryClient = useQueryClient();
  const active = useActiveBackend();
  const { trackAutomationEdited } = useTracking();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: Partial<Automation> }) =>
      AutomationService.updateAutomation(id, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AUTOMATIONS_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: AUTOMATION_DETAIL_QUERY_KEY });
      trackAutomationEdited({ backendKind: active.backend.kind });
    },
  });
}

export function useDeleteAutomation() {
  const queryClient = useQueryClient();
  const active = useActiveBackend();
  const { trackAutomationDeleted } = useTracking();
  return useMutation({
    mutationFn: (id: string) => AutomationService.deleteAutomation(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AUTOMATIONS_QUERY_KEY });
      trackAutomationDeleted({ backendKind: active.backend.kind });
    },
  });
}

export function useDispatchAutomation() {
  const queryClient = useQueryClient();
  const active = useActiveBackend();
  const { trackAutomationExecuted } = useTracking();
  return useMutation({
    mutationFn: (id: string) => AutomationService.dispatchAutomation(id),
    onSuccess: (_run, id) => {
      queryClient.invalidateQueries({ queryKey: AUTOMATIONS_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: AUTOMATION_DETAIL_QUERY_KEY });
      queryClient.invalidateQueries({
        queryKey: [...AUTOMATION_RUNS_QUERY_KEY, id],
      });
      trackAutomationExecuted({ backendKind: active.backend.kind });
    },
  });
}

export function useCancelAutomationRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ runId }: { automationId: string; runId: string }) =>
      AutomationService.cancelAutomationRun(runId),
    onSuccess: (_run, { automationId }) => {
      queryClient.invalidateQueries({ queryKey: AUTOMATIONS_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: AUTOMATION_DETAIL_QUERY_KEY });
      queryClient.invalidateQueries({
        queryKey: [...AUTOMATION_RUNS_QUERY_KEY, automationId],
      });
    },
  });
}
