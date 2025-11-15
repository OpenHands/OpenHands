import { queryClient } from "#/query-client-config";
import { OrganizationMember } from "#/types/org";

export const getMeFromQueryClient = (orgId: string | undefined) =>
  queryClient.getQueryData<OrganizationMember>(["organizations", orgId, "me"]);

export const getSelectedOrgFromQueryClient = () =>
  queryClient.getQueryData<string>(["selected_organization"]);
