import { useQuery } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { useActiveBackend } from "#/contexts/active-backend-context";

const AUTOMATION_SDK_VERSION_CACHE_TIME_MS = 60 * 60 * 1000;

async function getAutomationSdkVersion(): Promise<string | null> {
  const getSdkVersion = AutomationService.getSdkVersion;
  if (typeof getSdkVersion !== "function") return null;

  try {
    return await getSdkVersion();
  } catch {
    return null;
  }
}

export function useAutomationSdkVersion(): string | null {
  const active = useActiveBackend();
  const { backend } = active;
  const isSdkVersionSupported =
    typeof AutomationService.getSdkVersion === "function";
  const query = useQuery<string | null>({
    queryKey: [
      "automation-sdk-version",
      backend.id,
      backend.kind,
      backend.host,
      active.orgId,
    ],
    queryFn: getAutomationSdkVersion,
    enabled: isSdkVersionSupported && !isNoBackend(backend),
    staleTime: AUTOMATION_SDK_VERSION_CACHE_TIME_MS,
    gcTime: AUTOMATION_SDK_VERSION_CACHE_TIME_MS,
    retry: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    meta: { disableToast: true },
  });

  return query.data ?? null;
}
