import { useQuery } from "@tanstack/react-query";
import { batchGetCloudSandboxes } from "#/api/cloud/sandbox-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

export const useCloudSandbox = (sandboxId: string | null | undefined) => {
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
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
  });
};
