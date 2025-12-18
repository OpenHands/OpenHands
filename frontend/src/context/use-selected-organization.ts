import { useSelectedOrgId } from "#/hooks/query/use-selected-organization-id";
import { useUpdateSelectedOrganizationId } from "#/hooks/mutation/use-update-selected-organization-id";

export const useSelectedOrganizationId = () => {
  const { data: orgId } = useSelectedOrgId();
  const updateState = useUpdateSelectedOrganizationId();

  return {
    orgId,
    setOrgId: (newValue: string | null) => updateState.mutate(newValue),
  };
};
