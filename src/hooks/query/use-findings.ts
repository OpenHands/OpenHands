/**
 * TanStack Query hooks for Findings Service (PROJETOSIN-188).
 * @spec PROJETOSIN-188 — findings queries / triage mutation
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import FindingsService from "#/api/pentest/findings-service";
import {
  FindingsServiceError,
  type FindingStatus,
  type ListFindingsParams,
} from "#/api/pentest/findings-types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { FINDINGS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { usePentestCapabilitiesQuery } from "#/hooks/use-pentest-capabilities";

export function useFindingsList(
  params: ListFindingsParams | null,
  options?: { enabled?: boolean },
) {
  const { backend, orgId } = useActiveBackend();
  const enabled = Boolean(params?.engagement_id) && (options?.enabled ?? true);

  return useQuery({
    queryKey: params
      ? [
          ...FINDINGS_QUERY_KEYS.list(params.engagement_id, {
            status: params.status ?? null,
            severity: params.severity ?? null,
            source_tool: params.source_tool ?? null,
            page: params.page ?? 1,
            page_size: params.page_size ?? 20,
          }),
          backend.id,
          orgId,
        ]
      : [...FINDINGS_QUERY_KEYS.lists, "disabled", backend.id, orgId],
    queryFn: () => FindingsService.listFindings(params!),
    enabled,
    retry: false,
    meta: { disableToast: true },
  });
}

export function useFindingsStats(
  engagementId: string | null,
  options?: { enabled?: boolean },
) {
  const { backend, orgId } = useActiveBackend();
  const enabled = Boolean(engagementId) && (options?.enabled ?? true);

  return useQuery({
    queryKey: engagementId
      ? [...FINDINGS_QUERY_KEYS.stats(engagementId), backend.id, orgId]
      : [...FINDINGS_QUERY_KEYS.all, "stats", "disabled", backend.id, orgId],
    queryFn: () => FindingsService.getStats(engagementId!),
    enabled,
    retry: false,
    meta: { disableToast: true },
  });
}

export function useFindingDetail(
  findingId: string | null,
  options?: { enabled?: boolean },
) {
  const { backend, orgId } = useActiveBackend();
  const enabled = Boolean(findingId) && (options?.enabled ?? true);

  return useQuery({
    queryKey: findingId
      ? [...FINDINGS_QUERY_KEYS.detail(findingId), backend.id, orgId]
      : [...FINDINGS_QUERY_KEYS.details, "disabled", backend.id, orgId],
    queryFn: () => FindingsService.getFinding(findingId!),
    enabled,
    retry: false,
    meta: { disableToast: true },
  });
}

export function useTriageFinding() {
  const queryClient = useQueryClient();
  const { data: capabilities } = usePentestCapabilitiesQuery();
  const profile = capabilities?.profile ?? "agent-canvas";

  return useMutation({
    mutationFn: (input: {
      findingId: string;
      newStatus: FindingStatus;
      fpReason?: string;
      currentStatus?: string;
    }) =>
      FindingsService.triageFinding(
        input.findingId,
        {
          new_status: input.newStatus,
          fp_reason: input.fpReason,
          triaged_by: profile,
        },
        input.currentStatus,
      ),
    onSuccess: (finding) => {
      void queryClient.invalidateQueries({
        queryKey: FINDINGS_QUERY_KEYS.all,
      });
      void queryClient.setQueryData(
        FINDINGS_QUERY_KEYS.detail(finding.id),
        finding,
      );
    },
  });
}

export function isFindingsForbiddenError(error: unknown): boolean {
  return error instanceof FindingsServiceError && error.status === 403;
}
