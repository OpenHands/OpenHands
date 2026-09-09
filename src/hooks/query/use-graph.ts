import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import GraphService from "#/api/graph-service/graph-service.api";
import type {
  GraphConfig,
  GraphIndexRequest,
  GraphQueryParams,
} from "#/api/graph-service/graph-types";
import { GRAPH_QUERY_KEYS } from "#/hooks/query/query-keys";

function useInvalidateGraph() {
  const queryClient = useQueryClient();
  return () => {
    queryClient.invalidateQueries({ queryKey: GRAPH_QUERY_KEYS.all });
  };
}

export function useGraphStatus(root?: string) {
  return useQuery({
    queryKey: [...GRAPH_QUERY_KEYS.status(), root ?? ""] as const,
    queryFn: () => GraphService.getStatus(root),
  });
}

export function useGraphConfig() {
  return useQuery({
    queryKey: GRAPH_QUERY_KEYS.config(),
    queryFn: () => GraphService.getConfig(),
  });
}

export function usePutGraphConfig() {
  const invalidate = useInvalidateGraph();
  return useMutation({
    mutationFn: (payload: Partial<GraphConfig>) =>
      GraphService.putConfig(payload),
    onSuccess: invalidate,
  });
}

export function useGraphIndex() {
  const invalidate = useInvalidateGraph();
  return useMutation({
    mutationFn: (payload: GraphIndexRequest = {}) =>
      GraphService.index(payload),
    onSuccess: invalidate,
  });
}

export function useClearGraphIndex() {
  const invalidate = useInvalidateGraph();
  return useMutation({
    mutationFn: (root?: string) => GraphService.clearIndex(root),
    onSuccess: invalidate,
  });
}

export function useRetriggerGraphIndex() {
  const invalidate = useInvalidateGraph();
  return useMutation({
    mutationFn: (root?: string) => GraphService.retrigger(root),
    onSuccess: invalidate,
  });
}

export function useGraphQuery() {
  return useMutation({
    mutationFn: (params: GraphQueryParams) => GraphService.query(params),
  });
}

export function useImportGraphProjectConfig() {
  const invalidate = useInvalidateGraph();
  return useMutation({
    mutationFn: (path: string) => GraphService.importProjectConfig(path),
    onSuccess: invalidate,
  });
}
