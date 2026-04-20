import { redirect } from "react-router";
import { queryClient } from "#/query-client-config";
import OptionService from "#/api/option-service/option-service.api";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { CONFIG_CACHE_OPTIONS, QUERY_KEYS } from "#/hooks/query/query-keys";
import { getSelectedOrganizationIdFromStore } from "#/stores/selected-organization-store";
import { OrganizationsQueryData } from "#/types/org";

const FALLBACK_REDIRECT_PATH = "/settings/user";

const fetchConfig = () =>
  queryClient.fetchQuery({
    queryKey: QUERY_KEYS.WEB_CLIENT_CONFIG,
    queryFn: OptionService.getConfig,
    ...CONFIG_CACHE_OPTIONS,
  });

const fetchOrganizations = () =>
  queryClient.fetchQuery<OrganizationsQueryData>({
    queryKey: ["organizations"],
    queryFn: organizationService.getOrganizations,
    staleTime: 1000 * 60 * 5,
  });

// Fails open (returns null, allowing access) when org context cannot be
// resolved — config fetch, org fetch, or active-org lookup. Backend permission
// checks remain authoritative, so an outage degrades to "show the page" rather
// than locking every user out of LLM settings.
export const requireOrgDefaultsRedirect =
  (redirectPath: string = FALLBACK_REDIRECT_PATH) =>
  async ({ request }: { request: Request }) => {
    const config = await fetchConfig();

    if (config?.app_mode !== "saas") return null;

    let organizationsData: OrganizationsQueryData | undefined;
    try {
      organizationsData = await fetchOrganizations();
    } catch {
      return null;
    }

    const orgId =
      getSelectedOrganizationIdFromStore() ??
      organizationsData?.currentOrgId ??
      organizationsData?.items?.[0]?.id ??
      null;

    if (!orgId) return null;

    const activeOrg = organizationsData?.items.find((o) => o.id === orgId);
    if (!activeOrg) return null;

    const currentPath = new URL(request.url).pathname;
    if (currentPath === redirectPath) return null;

    return redirect(redirectPath);
  };
