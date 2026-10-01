import {
  useInfiniteQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useTracking } from "#/hooks/use-tracking";
import { uniqueById } from "#/utils/unique-by-id";
import type {
  Automation,
  AutomationSpec,
  AutomationsResponse,
} from "#/types/automation";
import {
  AUTOMATION_DETAIL_QUERY_KEY,
  AUTOMATION_RUNS_QUERY_KEY,
} from "./use-automation-detail";

export const AUTOMATIONS_QUERY_KEY = ["automations"] as const;

// The automation service caps `limit` at 100, so lists page by offset.
const AUTOMATIONS_PAGE_SIZE = 50;

interface UseAutomationsOptions {
  /** Automations per request; at most 100. A count-only caller passes 1. */
  pageSize?: number;
  enabled?: boolean;
}

/**
 * The org's automations, newest first, one page at a time. `data` joins the
 * loaded pages into one `AutomationsResponse` and `fetchNextPage` loads the
 * next one. An automation that shifts onto a later page between requests is
 * listed once.
 */
export function useAutomations(options: UseAutomationsOptions = {}) {
  const { pageSize = AUTOMATIONS_PAGE_SIZE, enabled = true } = options;
  const active = useActiveBackend();
  return useInfiniteQuery({
    queryKey: [
      ...AUTOMATIONS_QUERY_KEY,
      { pageSize },
      active.backend.id,
      active.orgId,
    ],
    queryFn: ({ pageParam }) =>
      AutomationService.getAutomations(pageSize, pageParam),
    initialPageParam: 0,
    getNextPageParam: (lastPage, _pages, lastOffset) => {
      const nextOffset = lastOffset + lastPage.automations.length;
      return nextOffset < lastPage.total ? nextOffset : undefined;
    },
    select: (data): AutomationsResponse => ({
      automations: uniqueById(data.pages.flatMap((page) => page.automations)),
      // The last page has the newest count.
      total: data.pages.at(-1)?.total ?? 0,
    }),
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
      // Runs carry a conversation_id surfaced in the sidebar, and the
      // conversation poll is paused on automation routes, so refresh the list
      // explicitly (same prefix all conversation mutations invalidate).
      queryClient.invalidateQueries({ queryKey: ["user", "conversations"] });
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
      // Same staleness hole as dispatch: the cancelled run's conversation
      // shows in the sidebar while the poll is paused on automation routes.
      queryClient.invalidateQueries({ queryKey: ["user", "conversations"] });
    },
  });
}
