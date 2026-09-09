import {
  supportsConversationRuntimeRoutes,
  CONVERSATION_RUNTIME_CLIENT_UPGRADE_MESSAGE,
} from "#/api/agent-server-client-options";
import { useQuery } from "@tanstack/react-query";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  getConversationServerInfo,
  usesIsolatedWorkspace,
  ISOLATED_WORKSPACE_MESSAGE,
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
  const clientUnsupported = isolated && !supportsConversationRuntimeRoutes();
  return {
    clientUnsupported,
    clientUnsupportedMessage: clientUnsupported
      ? CONVERSATION_RUNTIME_CLIENT_UPGRADE_MESSAGE
      : null,
    isolated,
    unsupportedMessage: isolated ? ISOLATED_WORKSPACE_MESSAGE : null,
  };
}
