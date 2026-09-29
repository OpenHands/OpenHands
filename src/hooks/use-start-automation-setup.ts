import { useCallback } from "react";
import {
  PENDING_AUTOMATION_SETUP_ID,
  clearAutomationFormSession,
  getAutomationFormSession,
  initializeAutomationFormSession,
} from "#/api/automation-form-session";
import { markAutomationSetupHandoff } from "#/api/automation-setup-handoff-store";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import { useCreateAutomationSetupConversation } from "#/hooks/use-create-automation-setup-conversation";
import { useTracking } from "#/hooks/use-tracking";
import { buildAutomationDraftTags } from "#/utils/automation-draft-tags";

/**
 * Open a new automation on the setup form.
 *
 * The automations page used to create an agent conversation immediately.
 * The form now opens with the conversation drawer hidden, and a conversation
 * starts only after the user sends a prompt.
 */
export function useStartAutomationSetup() {
  const active = useActiveBackend();
  const { navigate } = useNavigation();
  const { startAutomationSetupConversation, isPending } =
    useCreateAutomationSetupConversation();
  const { trackAutomationCreatedButton } = useTracking();

  const startSetup = useCallback(() => {
    trackAutomationCreatedButton({ backendKind: active.backend.kind });
    navigate?.("/automations/setup");
  }, [active.backend.kind, navigate, trackAutomationCreatedButton]);

  const startConversationFromPrompt = useCallback(
    (prompt: string) => {
      const draft = getAutomationFormSession(PENDING_AUTOMATION_SETUP_ID) ?? {
        prompt: "",
        kind: "prompt" as const,
      };
      startAutomationSetupConversation({
        query: prompt,
        entryPoint: draft.serverDraftId
          ? "automation_draft_resume"
          : "automations_add",
        onSuccess: async (conversation) => {
          const conversationId = conversation.conversation_id;
          initializeAutomationFormSession(conversationId, draft);
          markAutomationSetupHandoff(conversationId);
          clearAutomationFormSession(PENDING_AUTOMATION_SETUP_ID);
          if (draft.serverDraftId) {
            try {
              const [conversationDetails] =
                await AgentServerConversationService.batchGetAppConversations([
                  conversationId,
                ]);
              await AgentServerConversationService.updateConversationTags(
                conversationId,
                buildAutomationDraftTags(
                  conversationDetails?.tags ?? null,
                  draft.serverDraftId,
                  draft.materializedAutomationId,
                ),
              );
            } catch {
              // The in-memory form draft still opens for this navigation.
            }
          }
          navigate?.(`/conversations/${conversationId}`);
        },
      });
    },
    [navigate, startAutomationSetupConversation],
  );

  return {
    startSetup,
    startConversationFromPrompt,
    isPending,
  };
}
