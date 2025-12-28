import { FaExternalLinkAlt } from "react-icons/fa";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { RUNTIME_INACTIVE_STATES } from "#/types/agent-state";
import { transformVSCodeUrl } from "#/utils/vscode-url-helper";
import { useConversationId } from "#/hooks/use-conversation-id";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { useAgentState } from "#/hooks/use-agent-state";

export function VSCodeTooltipContent() {
  const { curAgentState } = useAgentState();

  const { t } = useTranslation();
  const { conversationId } = useConversationId();

  const handleVSCodeClick = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (conversationId) {
      try {
        const data = await ConversationService.getVSCodeUrl(conversationId);
        if (data.vscode_url) {
          const transformedUrl = transformVSCodeUrl(data.vscode_url);
          if (transformedUrl) {
            window.open(transformedUrl, "_blank");
          }
        }
      } catch (err) {
        // Silently handle the error
      }
    }
  };

  return (
    <div className="flex items-center gap-2">
      <span>{t(I18nKey.COMMON$CODE)}</span>
      {!RUNTIME_INACTIVE_STATES.includes(curAgentState) ? (
        <FaExternalLinkAlt
          className="w-3 h-3 text-inherit cursor-pointer"
          onClick={handleVSCodeClick}
        />
      ) : null}
    </div>
  );
}
