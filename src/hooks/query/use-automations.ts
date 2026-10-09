import {
  type InfiniteData,
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
  AutomationCreatedByFilter,
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

// One response from the loaded pages. At module level, so React Query runs it
// only when the pages change, not on every render.
function joinAutomationPages(
  data: InfiniteData<AutomationsResponse>,
): AutomationsResponse {
  return {
    automations: uniqueById(data.pages.flatMap((page) => page.automations)),
    // The last page has the newest count.
    total: data.pages.at(-1)?.total ?? 0,
  };
}

interface UseAutomationsOptions {
  /** Automations per request; at most 100. A count-only caller passes 1. */
  pageSize?: number;
  createdBy?: AutomationCreatedByFilter;
  enabled?: boolean;
}

/**
 * The org's automations, newest first, one page at a time. `data` joins the
 * loaded pages into one `AutomationsResponse` and `fetchNextPage` loads the
 * next one. An automation that shifts onto a later page between requests is
 * listed once.
 */
export function useAutomations(options: UseAutomationsOptions = {}) {
  const {
    pageSize = AUTOMATIONS_PAGE_SIZE,
    createdBy,
    enabled = true,
  } = options;
  const active = useActiveBackend();
  return useInfiniteQuery({
    queryKey: [
      ...AUTOMATIONS_QUERY_KEY,
      { pageSize, createdBy },
      active.backend.id,
      active.orgId,
    ],
    queryFn: ({ pageParam }) =>
      AutomationService.getAutomations(pageSize, pageParam, createdBy),
    initialPageParam: 0,
    getNextPageParam: (lastPage, _pages, lastOffset) => {
      const nextOffset = lastOffset + lastPage.automations.length;
      return nextOffset < lastPage.total ? nextOffset : undefined;
    },
    select: joinAutomationPages,
    staleTime: 0,
    // A new filter keeps the last list on screen until its first page lands;
    // another backend or org starts empty, so its list never shows foreign rows.
    placeholderData: (previousData, previousQuery) => {
      const [, , backendId, orgId] = previousQuery?.queryKey ?? [];
      return backendId === active.backend.id && orgId === active.orgId
        ? previousData
        : undefined;
    },
    enabled,
  });
}

interface UseToggleAutomationOptions {
  /**
   * Skip the global error toast, for a caller that shows its own. Dashboard
   * and detail toggles rely on the global toast, so it stays on by default.
   */
  disableToast?: boolean;
}

export function useToggleAutomation({
  disableToast = false,
}: UseToggleAutomationOptions = {}) {
  const queryClient = useQueryClient();
  const active = useActiveBackend();
  const { trackAutomationDisableButton } = useTracking();
  return useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) =>
      AutomationService.toggleAutomation(id, enabled),
    meta: { disableToast },
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
    // Callers toast the API message themselves; skip the generic global toast.
    meta: { disableToast: true },
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
    // Callers toast the API message themselves; skip the generic global toast.
    meta: { disableToast: true },
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
    // Callers toast the API message themselves; skip the generic global toast.
    meta: { disableToast: true },
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
    // Callers toast the API message themselves; skip the generic global toast.
    meta: { disableToast: true },
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
