import { useQuery } from "@tanstack/react-query";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  getConversationServerInfo,
  usesIsolatedWorkspace,
} from "#/api/conversation-workspace";

export function useConversationWorkspace() {
  const { backend } = useActiveBackend();
  const query = useQuery({
    queryKey: [
      "conversation-workspace",
      backend.id,
      backend.host,
      backend.connectionRevision,
    ],
    queryFn: () => getConversationServerInfo(),
    enabled: backend.kind === "local" && !!backend.host,
    staleTime: 60_000,
    retry: false,
    meta: { disableToast: true },
  });
  const isolated =
    backend.kind === "local" && usesIsolatedWorkspace(query.data);
  // Docker conversations bind-mount an attached host folder, so isolation
  // never blocks the workspace pickers.
  return { isolated, unsupportedMessage: null };
}
