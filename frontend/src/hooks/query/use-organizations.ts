import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useIsAuthed } from "./use-is-authed";
import { useConfig } from "./use-config";

export const useOrganizations = () => {
  const { data: userIsAuthenticated } = useIsAuthed();
  const { data: config } = useConfig();

  // Organizations are a SaaS-only feature - disable in OSS mode
  const isOssMode = config?.app_mode === "oss";

  return useQuery({
    queryKey: ["organizations"],
    queryFn: organizationService.getOrganizations,
    staleTime: 1000 * 60 * 5, // 5 minutes
    enabled: !!userIsAuthenticated && !isOssMode,
    select: (data) => ({
      // Sort organizations with personal workspace first, then alphabetically by name
      organizations: [...data.items].sort((a, b) => {
        const aIsPersonal = a.is_personal ?? false;
        const bIsPersonal = b.is_personal ?? false;
        if (aIsPersonal && !bIsPersonal) return -1;
        if (!aIsPersonal && bIsPersonal) return 1;
        return (a.name ?? "").localeCompare(b.name ?? "", undefined, {
          sensitivity: "base",
        });
      }),
      currentOrgId: data.currentOrgId,
    }),
  });
};
