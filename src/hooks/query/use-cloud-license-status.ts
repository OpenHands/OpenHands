import { useQuery } from "@tanstack/react-query";
import { getLockedCloudHost, isSameCloudHost } from "#/api/agent-server-config";
import { fetchCloudLicenseStatus } from "#/api/cloud/license-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

// The instance checks its license every 30 minutes by default, so a few
// minutes of staleness is fine.
const LICENSE_STATUS_REFETCH_MS = 5 * 60_000;

/**
 * The license status of the OpenHands Enterprise instance this Canvas is
 * locked to, for the license banner. Idle unless the active backend is the
 * locked cloud host. Polling stops once the instance says license enforcement
 * is off (`disabled`) or turns out not to serve the route (`null`).
 */
export function useCloudLicenseStatus() {
  const { backend } = useActiveBackend();
  const lockedHost = getLockedCloudHost();

  return useQuery({
    queryKey: [
      "cloud-license-status",
      backend.id,
      backend.connectionRevision ?? 0,
    ],
    queryFn: fetchCloudLicenseStatus,
    enabled:
      lockedHost !== null &&
      backend.kind === "cloud" &&
      isSameCloudHost(backend.host, lockedHost),
    staleTime: LICENSE_STATUS_REFETCH_MS,
    refetchInterval: (query) =>
      query.state.data === null || query.state.data?.state === "disabled"
        ? false
        : LICENSE_STATUS_REFETCH_MS,
    retry: false,
    meta: { disableToast: true },
  });
}
