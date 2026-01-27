import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useOrganizations } from "#/hooks/query/use-organizations";

export const useOrgTypeAndAccess = () => {
  const { organizationId } = useSelectedOrganizationId();
  const { data: organizations } = useOrganizations();

  const selectedOrg = organizations?.find((org) => org.id === organizationId);
  const isPersonalOrg = selectedOrg?.is_personal === true;
  // Team org = any org that is not explicitly marked as personal (includes undefined)
  const isTeamOrg = !!selectedOrg && !selectedOrg.is_personal;
  const canViewOrgRoutes = isTeamOrg && !!organizationId;

  return {
    selectedOrg,
    isPersonalOrg,
    isTeamOrg,
    canViewOrgRoutes,
    organizationId,
  };
};
