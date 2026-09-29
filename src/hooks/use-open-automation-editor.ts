import { useCallback, useState } from "react";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { initializeAutomationFormSession } from "#/api/automation-form-session";
import { useNavigation } from "#/context/navigation-context";
import { useCreateAutomationSetupConversation } from "#/hooks/use-create-automation-setup-conversation";
import type { Automation } from "#/types/automation";
import { setupDraftFromAutomation } from "#/utils/automation-edit-draft";
import { buildAutomationEditTags } from "#/utils/automation-draft-tags";

/**
 * Open an existing automation in the setup page.
 *
 * Create already lives in a conversation with the setup form beside the
 * agent. Edit starts the same conversation, seeded from the automation,
 * and tags it so a reload still knows which automation Save and Test
 * should update.
 */
export function useOpenAutomationEditor() {
  const { navigate } = useNavigation();
  const { startAutomationSetupConversation } =
    useCreateAutomationSetupConversation();
  const [openingAutomationId, setOpeningAutomationId] = useState<string | null>(
    null,
  );

  const openEditor = useCallback(
    (automation: Automation) => {
      if (openingAutomationId) return;
      setOpeningAutomationId(automation.id);
      const started = startAutomationSetupConversation({
        query: automation.prompt?.trim() || automation.name,
        entryPoint: "automation_edit",
        onSuccess: async (conversation) => {
          const conversationId = conversation.conversation_id;
          initializeAutomationFormSession(
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
            // The in-memory form draft still opens the form for this navigation.
          }
          navigate?.(`/conversations/${conversationId}`);
        },
        onSettled: () => setOpeningAutomationId(null),
      });
      if (!started) setOpeningAutomationId(null);
    },
    [navigate, openingAutomationId, startAutomationSetupConversation],
  );

  return { openEditor, openingAutomationId };
}
