import {
  type InfiniteData,
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import {
  getAutomationEndpoint,
  hasAutomationInterface,
} from "#/manifests/automation-interface";
import { useDeploymentCapabilities } from "#/hooks/query/use-manifest-capabilities";
import { isSdkHttpStatusError } from "#/api/agent-server-compatibility";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useTracking } from "#/hooks/use-tracking";
import { uniqueById } from "#/utils/unique-by-id";
import type {
  AutomationCreatedByFilter,
  AutomationSpec,
  AutomationTagCount,
  AutomationsResponse,
  ChangeRunAsRequest,
} from "#/types/automation";
import type { AutomationDraftListResponse } from "#/manifests/types";
import {
  AUTOMATION_DETAIL_QUERY_KEY,
  AUTOMATION_RUNS_QUERY_KEY,
} from "./use-automation-detail";

export const AUTOMATIONS_QUERY_KEY = ["automations"] as const;
export const AUTOMATION_DRAFTS_QUERY_KEY = ["automation-drafts"] as const;
export const AUTOMATION_TAGS_QUERY_KEY = ["automation-tags"] as const;

function getResponseStatus(error: unknown): number | null {
  if (!error || typeof error !== "object") return null;
  const response = (error as Record<string, unknown>).response;
  if (!response || typeof response !== "object") return null;
  const status = (response as Record<string, unknown>).status;
  return typeof status === "number" ? status : null;
}

function isEndpointUnavailable(error: unknown): boolean {
  const status = getResponseStatus(error);
  return (
    status === 404 ||
    status === 405 ||
    isSdkHttpStatusError(error, 404) ||
    isSdkHttpStatusError(error, 405)
  );
}

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
  tag?: string;
  enabled?: boolean;
}

interface UseAutomationDraftsOptions {
  limit?: number;
  offset?: number;
  enabled?: boolean;
}

export function useAutomationDrafts(options: UseAutomationDraftsOptions = {}) {
  const { limit = AUTOMATIONS_PAGE_SIZE, offset = 0, enabled = true } = options;
  const active = useActiveBackend();
  const deploymentCapabilities = useDeploymentCapabilities();
  const supportsAutomationDrafts = AutomationService.supportsAutomationDrafts(
    deploymentCapabilities.data,
  );

  return useQuery<AutomationDraftListResponse>({
    queryKey: [
      ...AUTOMATION_DRAFTS_QUERY_KEY,
      { limit, offset },
      active.backend.id,
      active.orgId,
      supportsAutomationDrafts,
    ],
    queryFn: async () => {
      if (!supportsAutomationDrafts) return { drafts: [], total: 0 };
      try {
        return await AutomationService.listServerDrafts({ limit, offset });
      } catch (error) {
        if (isEndpointUnavailable(error)) {
          return { drafts: [], total: 0 };
        }
        throw error;
      }
    },
    staleTime: 0,
    enabled: enabled && !deploymentCapabilities.isLoading,
  });
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
    tag,
    enabled = true,
  } = options;
  const active = useActiveBackend();
  return useInfiniteQuery({
    queryKey: [
      ...AUTOMATIONS_QUERY_KEY,
      { pageSize, createdBy, tag },
      active.backend.id,
      active.orgId,
    ],
    queryFn: ({ pageParam }) =>
      AutomationService.getAutomations(pageSize, pageParam, createdBy, tag),
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

/**
 * The org's tags with counts, or null when this deployment has no tags: the
 * manifest declares no tags endpoint, or the service predates it.
 */
export function useAutomationTags({ enabled = true } = {}) {
  const active = useActiveBackend();
  // The tags endpoint is optional in the manifest, which omits it before the
  // service supports tags.
  const supportsTags =
    hasAutomationInterface() && getAutomationEndpoint("tags") !== "";
  return useQuery<AutomationTagCount[] | null>({
    queryKey: [...AUTOMATION_TAGS_QUERY_KEY, active.backend.id, active.orgId],
    queryFn: async () => {
      try {
        return await AutomationService.listAutomationTags();
      } catch (error) {
        if (isEndpointUnavailable(error)) return null;
        throw error;
      }
    },
    staleTime: 0,
    enabled: enabled && supportsTags,
  });
}

/** Replace an automation's tags; the service lets only its creator do so. */
export function useUpdateAutomationTags() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, tags }: { id: string; tags: string[] }) =>
      AutomationService.updateAutomation(id, { tags }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AUTOMATIONS_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: AUTOMATION_DETAIL_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: AUTOMATION_TAGS_QUERY_KEY });
    },
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
      queryClient.invalidateQueries({ queryKey: AUTOMATION_DRAFTS_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: AUTOMATION_DETAIL_QUERY_KEY });
      if (!variables.enabled) {
        trackAutomationDisableButton({ backendKind: active.backend.kind });
      }
    },
  });
}

/**
 * Move an automation to a service account, or back to the caller. The service
 * turns it off, so every automation view is refetched. Callers toast the
 * service's `detail` themselves.
 */
export function useChangeRunAs() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: ChangeRunAsRequest }) =>
      AutomationService.changeRunAs(id, body),
    meta: { disableToast: true },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AUTOMATIONS_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: AUTOMATION_DETAIL_QUERY_KEY });
    },
  });
}

export function useDeleteAutomationDraft() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => AutomationService.deleteServerDraft(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AUTOMATION_DRAFTS_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: AUTOMATIONS_QUERY_KEY });
    },
  });
}

export function useDispatchAutomationDraft() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => AutomationService.dispatchServerDraft(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AUTOMATION_DRAFTS_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: AUTOMATIONS_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: ["user", "conversations"] });
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
