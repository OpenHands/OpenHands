import React from "react";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useOrganizations } from "#/hooks/query/use-organizations";
import { setSelectedOrg } from "#/utils/local-storage";

/**
 * Hook that automatically selects an organization when:
 * - No organization is currently selected in the frontend store
 * - Organizations data is available
 *
 * Selection priority:
 * 1. Backend's current_org_id (user's last selected organization, persisted server-side)
 * 2. First organization in the list (fallback for new users)
 *
 * This hook should be called from a component that always renders (e.g., root layout)
 * to ensure organization selection happens even when the OrgSelector component is hidden.
 */
export function useAutoSelectOrganization() {
  const { organizationId, setOrganizationId } = useSelectedOrganizationId();
  const { data } = useOrganizations();
  const organizations = data?.organizations;
  const currentOrgId = data?.currentOrgId;

  React.useEffect(() => {
    if (!organizationId && organizations && organizations.length > 0) {
      // Prefer backend's current_org_id (last selected org), fall back to first org
      const initialOrgId = currentOrgId ?? organizations[0].id;
      // Skip revalidation for initial auto-selection to avoid duplicate API calls.
      // Revalidation is only needed when user explicitly switches organizations
      // to redirect away from admin-only pages they may no longer have access to.
      setOrganizationId(initialOrgId, { skipRevalidation: true });
      // Broadcast org selection to other apps (e.g. Automations) via localStorage
      setSelectedOrg(initialOrgId);
    }
  }, [organizationId, organizations, currentOrgId, setOrganizationId]);
}
