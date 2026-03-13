import { useOrganizations } from "#/hooks/query/use-organizations";
import { useConfig } from "#/hooks/query/use-config";

export function useShouldHideOrgSelector() {
  const { data: config } = useConfig();
  const { data } = useOrganizations();
  const organizations = data?.organizations;

  // Always hide in OSS mode - organizations are a SaaS feature
  if (config?.app_mode === "oss") {
    return true;
  }

  // In SaaS mode, hide if user only has one personal org
  return organizations?.length === 1 && organizations[0]?.is_personal === true;
}
