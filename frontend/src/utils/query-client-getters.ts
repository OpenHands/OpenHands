import { queryClient } from "#/query-client-config";
import { OrganizationMember } from "#/types/org";

export const getMeFromQueryClient = (orgId: string | null) =>
  queryClient.getQueryData<OrganizationMember>(["organizations", orgId, "me"]);
