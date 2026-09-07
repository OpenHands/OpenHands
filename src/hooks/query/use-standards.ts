import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import StandardsService from "#/api/standards-service/standards-service.api";
import type {
  StandardsAuditParams,
  StandardsConfig,
  StandardsRunRequest,
} from "#/api/standards-service/standards-types";
import { STANDARDS_QUERY_KEYS } from "#/hooks/query/query-keys";

function useInvalidateStandards() {
  const queryClient = useQueryClient();
  return () => {
    queryClient.invalidateQueries({ queryKey: STANDARDS_QUERY_KEYS.all });
  };
}

export function useStandardsPlugins(root?: string) {
  return useQuery({
    queryKey: [...STANDARDS_QUERY_KEYS.plugins(), root ?? ""] as const,
    queryFn: () => StandardsService.listPlugins(root),
  });
}

export function useStandardsConfig(root?: string) {
  return useQuery({
    queryKey: [...STANDARDS_QUERY_KEYS.config(), root ?? ""] as const,
    queryFn: () => StandardsService.getConfig(root),
  });
}

export function usePutStandardsConfig() {
  const invalidate = useInvalidateStandards();
  return useMutation({
    mutationFn: (payload: Partial<StandardsConfig> & { root?: string }) =>
      StandardsService.putConfig(payload),
    onSuccess: invalidate,
  });
}

export function useRunStandards() {
  const invalidate = useInvalidateStandards();
  return useMutation({
    mutationFn: (payload: StandardsRunRequest) => StandardsService.run(payload),
    onSuccess: invalidate,
  });
}

export function useStandardsAudit(params: StandardsAuditParams = {}) {
  return useQuery({
    queryKey: [...STANDARDS_QUERY_KEYS.audit(), params] as const,
    queryFn: () => StandardsService.getAudit(params),
  });
}
