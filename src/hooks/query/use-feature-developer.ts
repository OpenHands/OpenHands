import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import FeatureDeveloperService from "#/api/feature-developer-service/feature-developer-service.api";
import type { StartFeatureDevRunPayload } from "#/api/feature-developer-service/feature-developer-types";
import { FEATURE_DEV_QUERY_KEYS } from "#/hooks/query/query-keys";

const LIVE_STATUSES = new Set(["pending", "running", "paused"]);

export function useFeatureDevRuns() {
  return useQuery({
    queryKey: FEATURE_DEV_QUERY_KEYS.list(),
    queryFn: () => FeatureDeveloperService.listRuns(),
  });
}

export function useFeatureDevRun(runId: string | null) {
  return useQuery({
    queryKey: FEATURE_DEV_QUERY_KEYS.detail(runId ?? ""),
    queryFn: () => FeatureDeveloperService.getRun(runId!),
    enabled: Boolean(runId),
    refetchInterval: (query) =>
      LIVE_STATUSES.has(String(query.state.data?.status ?? "")) ? 2000 : false,
  });
}

export function useFeatureDevReport(runId: string | null) {
  return useQuery({
    queryKey: FEATURE_DEV_QUERY_KEYS.report(runId ?? ""),
    queryFn: () => FeatureDeveloperService.getReport(runId!),
    enabled: Boolean(runId),
  });
}

function useInvalidateFeatureDev(runId?: string | null) {
  const queryClient = useQueryClient();
  return () => {
    queryClient.invalidateQueries({ queryKey: FEATURE_DEV_QUERY_KEYS.all });
    if (runId) {
      queryClient.invalidateQueries({
        queryKey: FEATURE_DEV_QUERY_KEYS.detail(runId),
      });
    }
  };
}

export function useStartFeatureDevRun() {
  const invalidate = useInvalidateFeatureDev();
  return useMutation({
    mutationFn: (payload: StartFeatureDevRunPayload) =>
      FeatureDeveloperService.startRun(payload),
    onSuccess: invalidate,
  });
}
