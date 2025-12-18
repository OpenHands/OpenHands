import { queryClient } from "#/query-client-config";
import { OrganizationMember } from "#/types/org";
import { SELECTED_ORGANIZATION_QUERY_KEY } from "#/hooks/query/use-selected-organization-id";

export const getMeFromQueryClient = (orgId: string | undefined) =>
  queryClient.getQueryData<OrganizationMember>(["organizations", orgId, "me"]);

export const getSelectedOrgFromQueryClient = () =>
  queryClient.getQueryData<string>([SELECTED_ORGANIZATION_QUERY_KEY]);
