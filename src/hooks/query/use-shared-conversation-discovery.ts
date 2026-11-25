import { useQuery } from "@tanstack/react-query";
import { getCloudOrganizations } from "#/api/cloud/organization-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

/**
 * Whether the active cloud org lets its members discover the conversations
 * shared with it. An Org Admin turns this off on Settings > Organization.
 * `false` while the org list loads. Only an explicit `false` from the org
 * turns it off, so an app-server that predates the setting leaves it on. If
 * the org list fails to load it stays on: the server still returns an empty
 * list for an org that turned discovery off.
 */
export function useSharedConversationDiscovery(): boolean {
  const { backend, orgId } = useActiveBackend();

  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  const { data, isError } = useQuery({
    queryKey: [
      "shared-conversation-discovery",
      backend.id,
      orgId,
      backend.connectionRevision ?? 0,
    ],
    queryFn: () => getCloudOrganizations(backend),
    enabled: backend.kind === "cloud" && !!orgId,
    staleTime: 1000 * 60 * 5,
    retry: false,
    meta: { disableToast: true },
  });

  if (isError) return true;
  if (!data) return false;
  const org = data.items.find((item) => item.id === orgId);
  return org?.enable_shared_conversation_discovery !== false;
}
