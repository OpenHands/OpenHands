import { useRevalidator } from "react-router";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";

interface SetOrganizationIdOptions {
  /** Skip route revalidation. Useful for initial auto-selection to avoid duplicate API calls. */
  skipRevalidation?: boolean;
}

export const useSelectedOrganizationId = () => {
  const revalidator = useRevalidator();
  const { organizationId, setOrganizationId: setOrganizationIdStore } =
    useSelectedOrganizationStore();

  const setOrganizationId = (
    newOrganizationId: string | null,
    options?: SetOrganizationIdOptions,
  ) => {
    setOrganizationIdStore(newOrganizationId);
    // Revalidate route to ensure the latest orgId is used.
    // This is useful for redirecting the user away from admin-only org pages.
    // Skip revalidation for initial auto-selection to avoid duplicate API calls.
    if (!options?.skipRevalidation) {
      revalidator.revalidate();
    }
  };

  return { organizationId, setOrganizationId };
};
