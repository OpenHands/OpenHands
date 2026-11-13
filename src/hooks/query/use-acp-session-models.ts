import { useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  getAcpProvider,
  type ACPEffortChoice,
  type ACPModelOption,
} from "#/constants/acp-providers";
import {
  acpModelScope,
  rememberAcpModels,
} from "#/utils/remembered-acp-models";

const NO_MODELS: ACPModelOption[] = [];
const NO_EFFORTS: ACPEffortChoice[] = [];

/** Query-key prefix of a Cloud conversation's sandbox session state. */
export const ACP_SESSION_RUNTIME_QUERY_KEY = "acp-session-runtime";

/** A Cloud conversation's session state, read from its sandbox. */
function useCloudSessionRuntime(
  conversation: AppConversation | null | undefined,
) {
  const { backend } = useActiveBackend();
  const id = conversation?.id;
  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;
  const enabled =
    backend.kind === "cloud" &&
    !!id &&
    !!conversationUrl &&
    !!sessionApiKey &&
    conversation?.sandbox_status === "RUNNING";

  return useQuery({
    queryKey: [
      ACP_SESSION_RUNTIME_QUERY_KEY,
      id,
      conversationUrl,
      sessionApiKey,
    ],
    queryFn: () =>
      AgentServerConversationService.getRuntimeConversation(
        id as string,
        conversationUrl,
        sessionApiKey,
      ),
    enabled,
    refetchInterval: (query) =>
      query.state.data?.available_models.length ? false : 10_000,
    staleTime: Infinity,
    retry: false,
  });
}

/**
 * The models an ACP conversation's session reported. Cloud conversations
 * carry no list, so it is read from the conversation's sandbox. Lists from
 * built-in agents are remembered under the secrets the conversation may use.
 */
export function useAcpSessionModels(
  conversation: AppConversation | null | undefined,
): ACPModelOption[] {
  const { backend } = useActiveBackend();
  const isCloud = backend.kind === "cloud";
  const runtime = useCloudSessionRuntime(conversation);

  const models = useMemo(
    () =>
      (isCloud
        ? runtime.data?.available_models
        : conversation?.acp_available_models) ?? NO_MODELS,
    [isCloud, runtime.data, conversation?.acp_available_models],
  );
  const server = conversation?.acp_server;
  const scope = acpModelScope(
    conversation?.launched_agent_profile?.secret_refs,
  );
  useEffect(() => {
    if (server && getAcpProvider(server)) {
      rememberAcpModels(backend.id, server, scope, models);
    }
  }, [backend.id, server, scope, models]);

  return models;
}

/** The reasoning effort an ACP conversation's session reports, and its levels. */
export function useAcpSessionEffort(
  conversation: AppConversation | null | undefined,
): { efforts: ACPEffortChoice[]; currentEffort: string | null } {
  const { backend } = useActiveBackend();
  const runtime = useCloudSessionRuntime(conversation);
  if (backend.kind === "cloud") {
    return {
      efforts: runtime.data?.available_efforts ?? NO_EFFORTS,
      currentEffort: runtime.data?.current_effort ?? null,
    };
  }
  return {
    efforts: conversation?.acp_available_efforts ?? NO_EFFORTS,
    currentEffort: conversation?.acp_current_effort ?? null,
  };
}
