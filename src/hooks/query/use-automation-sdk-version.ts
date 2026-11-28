import React from "react";
import { QueryClientContext, useQuery } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { isNoBackend } from "#/api/backend-registry/active-store";
import type { ResolvedActiveBackend } from "#/api/backend-registry/types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { getQueryClient } from "#/query-client-config";

export const AUTOMATION_SDK_VERSION_CACHE_NAMESPACE = "automation-sdk-version";

const AUTOMATION_SDK_VERSION_CACHE_TIME_MS = 60 * 60 * 1000;
async function fetchAutomationSdkVersion(
  active: ResolvedActiveBackend,
): Promise<string | null> {
  if (typeof AutomationService.getSdkVersion !== "function") {
    return null;
  }

  try {
    return await AutomationService.getSdkVersion(active);
  } catch {
    return null;
  }
}

export function useAutomationSdkVersion() {
  const active = useActiveBackend();
  const { backend } = active;
  const providerQueryClient = React.useContext(QueryClientContext);
  const canFetchSdkVersion =
    !isNoBackend(backend) &&
    typeof AutomationService.getSdkVersion === "function";
  const { data } = useQuery(
    // The query key intentionally omits credentials; this fetch uses its render's backend snapshot.
    // eslint-disable-next-line @tanstack/query/exhaustive-deps
    {
      queryKey: [
        AUTOMATION_SDK_VERSION_CACHE_NAMESPACE,
        backend.id,
        backend.kind,
        backend.host,
        active.orgId ?? "",
      ],
      queryFn: () => fetchAutomationSdkVersion(active),
      enabled: canFetchSdkVersion,
      initialData: canFetchSdkVersion ? undefined : null,
      staleTime: AUTOMATION_SDK_VERSION_CACHE_TIME_MS,
      gcTime: AUTOMATION_SDK_VERSION_CACHE_TIME_MS,
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
    },
    providerQueryClient ?? getQueryClient(),
  );

  return data ?? null;
}
