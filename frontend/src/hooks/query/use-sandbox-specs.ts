import { useQuery } from "@tanstack/react-query";
import { SandboxService } from "#/api/sandbox-service/sandbox-service.api";

export const useSandboxSpecs = () =>
  useQuery({
    queryKey: ["sandbox-specs"],
    queryFn: () => SandboxService.searchSandboxSpecs(),
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
