import React from "react";
import { QueryClientContext, useQuery } from "@tanstack/react-query";
import RunningVersionsService from "#/api/running-versions-service";
import AutomationService from "#/api/automation-service/automation-service.api";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { getQueryClient } from "#/query-client-config";
import { RUNNING_VERSIONS_QUERY_KEYS } from "./query-keys";

const VERSION_CACHE_TIME_MS = 60_000;

export function useRunningVersions() {
  const active = useActiveBackend();
  const { backend } = active;
  const providerQueryClient = React.useContext(QueryClientContext);
  const queryClient = providerQueryClient ?? getQueryClient();
  const enabled = !isNoBackend(backend);
  const backendIdentity = [
    backend.id,
    backend.kind,
    backend.host,
    active.orgId,
    backend.connectionRevision ?? 0,
  ] as const;

  const backendQuery = useQuery(
    // The query key intentionally omits credentials; this fetch uses its render's backend snapshot.
    // eslint-disable-next-line @tanstack/query/exhaustive-deps
    {
      queryKey: RUNNING_VERSIONS_QUERY_KEYS.forBackend(...backendIdentity),
      queryFn: () => RunningVersionsService.getBackendVersions(active),
      enabled,
      staleTime: VERSION_CACHE_TIME_MS,
      gcTime: VERSION_CACHE_TIME_MS,
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
    },
    queryClient,
  );

  const automationQuery = useQuery(
    // The query key intentionally omits credentials; this fetch uses its render's backend snapshot.
    // eslint-disable-next-line @tanstack/query/exhaustive-deps
    {
      queryKey: RUNNING_VERSIONS_QUERY_KEYS.forAutomation(...backendIdentity),
      queryFn: () => AutomationService.getVersionInfo(active),
      enabled,
      staleTime: VERSION_CACHE_TIME_MS,
      gcTime: VERSION_CACHE_TIME_MS,
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
    },
    queryClient,
  );

  return {
    backendVersions: backendQuery.data ?? null,
    automationVersions: automationQuery.data ?? null,
    isLoading: backendQuery.isLoading || automationQuery.isLoading,
    isFetching: backendQuery.isFetching || automationQuery.isFetching,
    refetch: async () => {
      await Promise.all([backendQuery.refetch(), automationQuery.refetch()]);
    },
  };
}
