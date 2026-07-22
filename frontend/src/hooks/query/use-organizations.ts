import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useIsAuthed } from "./use-is-authed";
import { useConfig } from "./use-config";

export const useOrganizations = () => {
  const { data: userIsAuthenticated } = useIsAuthed();
  const { data: config } = useConfig();

  // Organizations are a SaaS-only feature - disable in OSS mode
  const isOssMode = config?.app_mode === "oss";
  const hidePersonalWorkspaces =
    config?.feature_flags?.hide_personal_workspaces === true;

  return useQuery({
    queryKey: ["organizations"],
    queryFn: organizationService.getOrganizations,
    staleTime: 1000 * 60 * 5, // 5 minutes
    enabled: !!userIsAuthenticated && !isOssMode,
    select: (data) => {
      // In org-only installs, hide personal workspaces — but only when the
      // user belongs to at least one team org, so a user whose only
      // workspace is personal (e.g. the default org isn't created yet) is
      // never left with zero workspaces.
      const hasTeamOrg = data.items.some((org) => !(org.is_personal ?? false));
      const visible =
        hidePersonalWorkspaces && hasTeamOrg
          ? data.items.filter((org) => !(org.is_personal ?? false))
          : data.items;

      // Sort organizations with personal workspace first, then alphabetically by name
      const organizations = [...visible].sort((a, b) => {
        const aIsPersonal = a.is_personal ?? false;
        const bIsPersonal = b.is_personal ?? false;
        if (aIsPersonal && !bIsPersonal) return -1;
        if (!aIsPersonal && bIsPersonal) return 1;
        return (a.name ?? "").localeCompare(b.name ?? "", undefined, {
          sensitivity: "base",
        });
      });

      return {
        organizations,
        currentOrgId: data.currentOrgId,
      };
    },
  });
};
