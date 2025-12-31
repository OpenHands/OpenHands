import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { I18nKey } from "#/i18n/declaration";
import { transformVSCodeUrl } from "#/utils/vscode-url-helper";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";

// Define the return type for the VS Code URL query
interface VSCodeUrlResult {
  url: string | null;
  error: string | null;
}

export const useVSCodeUrl = () => {
  const { t } = useTranslation();
  const { conversationId } = useConversationId();
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady();

  const isV1Conversation = conversation?.conversation_version === "V1";

  return useQuery<VSCodeUrlResult>({
    queryKey: [
      "vscode_url",
      conversationId,
      isV1Conversation,
      conversation?.url,
      conversation?.session_api_key,
    ],
    queryFn: async () => {
      if (!conversationId) throw new Error("No conversation ID");

      // Use appropriate API based on conversation version
      const data = isV1Conversation
        ? await V1ConversationService.getVSCodeUrl(
            conversationId,
            conversation?.url,
            conversation?.session_api_key,
          )
        : await ConversationService.getVSCodeUrl(conversationId);

      if (data.vscode_url) {
        return {
          url: transformVSCodeUrl(data.vscode_url),
          error: null,
        };
      }
      return {
        url: null,
        error: t(I18nKey.VSCODE$URL_NOT_AVAILABLE),
      };
    },
    enabled: runtimeIsReady && !!conversationId,
    refetchOnMount: true,
    retry: 3,
  });
};
