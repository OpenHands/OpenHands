import { useQuery } from "@tanstack/react-query";
import { batchGetCloudSandboxes } from "#/api/cloud/sandbox-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

export interface UseCloudSandboxOptions {
  /** Poll until the cloud runtime reports a forwarded worker URL. */
  pollForWorkerUrls?: boolean;
}

const WORKER_URL_PREFIX = "WORKER_";

export const useCloudSandbox = (
  sandboxId: string | null | undefined,
  options: UseCloudSandboxOptions = {},
) => {
  const active = useActiveBackend();
  const isCloud = active.backend.kind === "cloud";

  return useQuery({
    queryKey: ["cloud", "sandbox", active.backend.id, active.orgId, sandboxId],
    queryFn: async () => {
      if (!sandboxId) return null;
      const [sandbox] = await batchGetCloudSandboxes([sandboxId]);
      return sandbox ?? null;
    },
    enabled: isCloud && !!sandboxId,
    staleTime: options.pollForWorkerUrls ? 0 : 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
    refetchInterval: options.pollForWorkerUrls
      ? (query) => {
          const sandbox = query.state.data;
          if (
            !sandboxId ||
            sandbox?.status === "ERROR" ||
            sandbox?.status === "MISSING"
          ) {
            return false;
          }
          const hasWorkerUrl = sandbox?.exposed_urls?.some((url) =>
            url.name.startsWith(WORKER_URL_PREFIX),
          );
          return hasWorkerUrl
            ? false
            : sandbox?.status === "RUNNING"
              ? 1500
              : 3000;
        }
      : false,
  });
};
