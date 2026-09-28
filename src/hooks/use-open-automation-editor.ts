/**
 * Open an existing automation in the setup page.
 *
 * Create already lives in a conversation with the setup form beside the
 * agent. Edit starts the same conversation, seeded from the automation,
 * and tags it so a reload still knows which automation Save and Test
 * should update.
 */
import { useCallback, useState } from "react";
import { useTranslation } from "react-i18next";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { setAutomationSetupDraft } from "#/api/automation-setup-draft-store";
import { useNavigation } from "#/context/navigation-context";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { setupDraftFromAutomation } from "#/utils/automation-edit-draft";
import { buildAutomationEditTags } from "#/utils/automation-draft-tags";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

export function useOpenAutomationEditor() {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const createConversationMutation = useCreateConversation();
  const [openingAutomationId, setOpeningAutomationId] = useState<string | null>(
    null,
  );

  const openEditor = useCallback(
    (automation: Automation) => {
      if (openingAutomationId) return;
      setOpeningAutomationId(automation.id);
      createConversationMutation.mutate(
        {
          query: automation.prompt?.trim() || automation.name,
          automationSetup: true,
          entryPoint: "automation_edit",
        },
        {
          onSuccess: async (conversation) => {
            const conversationId = conversation.conversation_id;
            setAutomationSetupDraft(
              conversationId,
              setupDraftFromAutomation(automation),
            );
            try {
              const [conversationDetails] =
                await AgentServerConversationService.batchGetAppConversations([
                  conversationId,
                ]);
              await AgentServerConversationService.updateConversationTags(
                conversationId,
                buildAutomationEditTags(
                  conversationDetails?.tags ?? null,
                  automation.id,
                ),
              );
            } catch {
              // The session draft still opens the form. The edit id is on
              // that draft, so Save and Test work even if tagging fails.
            }
            navigate?.(`/conversations/${conversationId}`);
          },
          onError: (error) => {
            displayErrorToast(
              getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)),
            );
          },
          onSettled: () => setOpeningAutomationId(null),
        },
      );
    },
    [createConversationMutation, navigate, openingAutomationId, t],
  );

  return { openEditor, openingAutomationId };
}
