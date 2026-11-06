import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo } from "react";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { AcpPermissionsService } from "#/api/acp-service/acp-permissions.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { ACP_PERMISSION_QUERY_KEYS } from "./query-keys";

const PERMISSION_POLL_MS = 1000;

export function useAcpPermissions(
  conversation: AppConversation | null | undefined,
) {
  const { backend, orgId } = useActiveBackend();
  const queryClient = useQueryClient();
  const authorization = useMemo(
    () => ({
      key: conversation?.session_api_key,
      generation: crypto.randomUUID(),
    }),
    [conversation?.session_api_key],
  );
  const queryKey = ACP_PERMISSION_QUERY_KEYS.pending(
    backend.id,
    backend.connectionRevision,
    orgId,
    conversation?.id,
    conversation?.conversation_url,
    authorization.generation,
  );
  const scope = JSON.stringify(queryKey);
  const enabled =
    !!conversation?.id &&
    conversation.agent_kind === "acp" &&
    (backend.kind !== "cloud" ||
      (!!conversation.conversation_url &&
        !!conversation.session_api_key &&
        conversation.sandbox_status === "RUNNING"));
  const target = {
    id: conversation?.id ?? "",
    conversationUrl: conversation?.conversation_url,
    sessionApiKey: authorization.key,
  };
  // Credentials stay in the transport closure. Opaque authorization generation,
  // backend revision and runtime URL isolate every replacement without caching keys.
  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  const pending = useQuery({
    queryKey,
    queryFn: () => AcpPermissionsService.list(target),
    enabled,
    retry: false,
    refetchInterval: PERMISSION_POLL_MS,
    refetchIntervalInBackground: false,
    staleTime: 0,
    gcTime: 0,
  });
  const decision = useMutation({
    mutationFn: ({
      requestId,
      optionId,
      requestScope,
    }: {
      requestId: string;
      optionId: string | null;
      requestScope: string;
    }) => {
      if (!enabled || requestScope !== scope)
        throw new Error("Permission conversation changed");
      return AcpPermissionsService.decide(target, requestId, {
        option_id: optionId,
      });
    },
    retry: false,
    onSettled: () => queryClient.invalidateQueries({ queryKey, exact: true }),
  });
  return { pending, decision, scope, enabled };
}
