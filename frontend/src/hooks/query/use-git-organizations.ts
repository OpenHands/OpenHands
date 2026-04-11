import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import UserService from "#/api/user-service/user-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const gitClaimsQueryKey = (organizationId: string | null) =>
  ["organizations", organizationId, "git-claims"] as const;

export const useUserGitOrganizations = () =>
  useQuery({
    queryKey: ["user", "git-organizations"],
    queryFn: () => UserService.getGitOrganizations(),
    staleTime: 1000 * 60 * 5,
    meta: { disableToast: true },
  });

export const useGitClaims = () => {
  const { organizationId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: gitClaimsQueryKey(organizationId),
    queryFn: () => organizationService.getGitClaims({ orgId: organizationId! }),
    enabled: !!organizationId,
    staleTime: 1000 * 30,
  });
};
