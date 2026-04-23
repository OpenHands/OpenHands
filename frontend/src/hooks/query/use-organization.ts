import { useMemo } from "react";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useOrganizations } from "#/hooks/query/use-organizations";

export const useOrganization = () => {
  const { organizationId } = useSelectedOrganizationId();
  const organizationsQuery = useOrganizations();

  const organization = useMemo(
    () =>
      organizationsQuery.data?.organizations.find(
        (candidate) => candidate.id === organizationId,
      ),
    [organizationId, organizationsQuery.data?.organizations],
  );

  return {
    ...organizationsQuery,
    data: organization,
  };
};
