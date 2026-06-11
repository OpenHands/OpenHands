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
  if (organizations?.length === 1 && organizations[0]?.is_personal === true) {
    return true;
  }

  // When personal workspaces are hidden and only one org is visible,
  // there is nothing to switch between.
  return (
    config?.feature_flags?.hide_personal_workspaces === true &&
    organizations?.length === 1
  );
}
