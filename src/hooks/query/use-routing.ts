import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import RoutingService from "#/api/routing-service/routing-service.api";
import type {
  RoutingConfig,
  RoutingResolveRequest,
  RoutingRouterModelConfig,
  RoutingTaxonomy,
} from "#/api/routing-service/routing-types";
import { ROUTING_QUERY_KEYS } from "#/hooks/query/query-keys";

function useInvalidateRouting() {
  const queryClient = useQueryClient();
  return () => {
    queryClient.invalidateQueries({ queryKey: ROUTING_QUERY_KEYS.all });
  };
}

export function useRoutingConfig() {
  return useQuery({
    queryKey: ROUTING_QUERY_KEYS.config(),
    queryFn: () => RoutingService.getConfig(),
  });
}

export function usePutRoutingConfig() {
  const invalidate = useInvalidateRouting();
  return useMutation({
    mutationFn: (payload: Partial<RoutingConfig>) =>
      RoutingService.putConfig(payload),
    onSuccess: invalidate,
  });
}

export function useRoutingTaxonomy() {
  return useQuery({
    queryKey: ROUTING_QUERY_KEYS.taxonomy(),
    queryFn: () => RoutingService.getTaxonomy(),
  });
}

export function usePutRoutingTaxonomy() {
  const invalidate = useInvalidateRouting();
  return useMutation({
    mutationFn: (payload: Partial<RoutingTaxonomy> & { reset?: boolean }) =>
      RoutingService.putTaxonomy(payload),
    onSuccess: invalidate,
  });
}

export function useRoutingRegistry() {
  return useQuery({
    queryKey: ROUTING_QUERY_KEYS.registry(),
    queryFn: () => RoutingService.getRegistry(),
  });
}

export function useRoutingSources() {
  return useQuery({
    queryKey: ROUTING_QUERY_KEYS.sources(),
    queryFn: () => RoutingService.getSources(),
  });
}

export function useIngestRoutingBenchmarks() {
  const invalidate = useInvalidateRouting();
  return useMutation({
    mutationFn: (sources?: string[]) =>
      RoutingService.ingestBenchmarks(sources),
    onSuccess: invalidate,
  });
}

export function useRoutingRouterModel() {
  return useQuery({
    queryKey: ROUTING_QUERY_KEYS.routerModel(),
    queryFn: () => RoutingService.getRouterModel(),
  });
}

export function usePutRoutingRouterModel() {
  const invalidate = useInvalidateRouting();
  return useMutation({
    mutationFn: (payload: Partial<RoutingRouterModelConfig>) =>
      RoutingService.putRouterModel(payload),
    onSuccess: invalidate,
  });
}

export function useRoutingLocalRuntimes() {
  return useQuery({
    queryKey: ROUTING_QUERY_KEYS.localRuntimes(),
    queryFn: () => RoutingService.getLocalRuntimes(),
  });
}

export function useRoutingAudit() {
  return useQuery({
    queryKey: ROUTING_QUERY_KEYS.audit(),
    queryFn: () => RoutingService.getAudit(),
  });
}

export function useRoutingResolve() {
  return useMutation({
    mutationFn: (payload: RoutingResolveRequest) =>
      RoutingService.resolve(payload),
  });
}
