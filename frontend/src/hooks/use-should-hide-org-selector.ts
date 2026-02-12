import { useOrganizations } from "#/hooks/query/use-organizations";

export function useShouldHideOrgSelector() {
  const { data: organizations } = useOrganizations();
  return organizations?.length === 1 && organizations[0]?.is_personal === true;
}
