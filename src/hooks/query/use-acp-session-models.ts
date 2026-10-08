import { useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { getAcpProvider, type ACPModelOption } from "#/constants/acp-providers";
import { rememberAcpModels } from "#/utils/remembered-acp-models";

const NO_MODELS: ACPModelOption[] = [];

/**
 * The models an ACP conversation's session reported. Cloud conversations
 * carry no list, so it is read from the conversation's sandbox. Lists from
 * built-in agents are remembered for pickers outside the conversation.
 */
export function useAcpSessionModels(
  conversation: AppConversation | null | undefined,
): ACPModelOption[] {
  const { backend } = useActiveBackend();
  const isCloud = backend.kind === "cloud";
  const id = conversation?.id;
  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;
  const enabled =
    isCloud &&
    !!id &&
    !!conversationUrl &&
    !!sessionApiKey &&
    conversation?.sandbox_status === "RUNNING";

  const runtime = useQuery({
    queryKey: ["acp-session-models", id, conversationUrl, sessionApiKey],
    queryFn: async () =>
      (
        await AgentServerConversationService.getRuntimeConversation(
          id as string,
          conversationUrl,
          sessionApiKey,
        )
      ).available_models,
    enabled,
    // The session reports its list once it starts; stop asking after that.
    refetchInterval: (query) => (query.state.data?.length ? false : 10_000),
    staleTime: Infinity,
    retry: false,
  });

  const models = useMemo(
    () =>
      (isCloud ? runtime.data : conversation?.acp_available_models) ??
      NO_MODELS,
    [isCloud, runtime.data, conversation?.acp_available_models],
  );
  const server = conversation?.acp_server;
  useEffect(() => {
    if (server && getAcpProvider(server)) {
      rememberAcpModels(backend.id, server, models);
    }
  }, [backend.id, server, models]);

  return models;
}
