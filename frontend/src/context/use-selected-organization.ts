import { useRevalidator } from "react-router";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";

export const useSelectedOrganizationId = () => {
  const revalidator = useRevalidator();
  const { orgId, setOrgId: setOrgIdStore } = useSelectedOrganizationStore();

  const setOrgId = (newOrgId: string | null) => {
    setOrgIdStore(newOrgId);
    // Revalidate route to ensure the latest orgId is used.
    // This is useful for redirecting the user away from admin-only org pages.
    revalidator.revalidate();
  };

  return { orgId, setOrgId };
};
