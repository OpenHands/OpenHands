import React from "react";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useOrganizations } from "#/hooks/query/use-organizations";

/**
 * Hook that automatically selects the first organization when:
 * - No organization is currently selected
 * - Organizations data is available
 *
 * This hook should be called from a component that always renders (e.g., root layout)
 * to ensure organization selection happens even when the OrgSelector component is hidden.
 */
export function useAutoSelectOrganization() {
  const { organizationId, setOrganizationId } = useSelectedOrganizationId();
  const { data: organizations } = useOrganizations();

  React.useEffect(() => {
    if (!organizationId && organizations && organizations.length > 0) {
      // Skip revalidation for initial auto-selection to avoid duplicate API calls.
      // Revalidation is only needed when user explicitly switches organizations
      // to redirect away from admin-only pages they may no longer have access to.
      setOrganizationId(organizations[0].id, { skipRevalidation: true });
    }
  }, [organizationId, organizations, setOrganizationId]);
}
