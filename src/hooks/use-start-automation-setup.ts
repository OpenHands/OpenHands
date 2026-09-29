import { useCallback } from "react";
import {
  PENDING_AUTOMATION_SETUP_ID,
  clearAutomationFormSession,
  getAutomationFormSession,
  initializeAutomationFormSession,
} from "#/api/automation-form-session";
import { markAutomationSetupHandoff } from "#/api/automation-setup-handoff-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import { useCreateAutomationSetupConversation } from "#/hooks/use-create-automation-setup-conversation";
import { useTracking } from "#/hooks/use-tracking";

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
        entryPoint: "automations_add",
        onSuccess: (conversation) => {
          initializeAutomationFormSession(conversation.conversation_id, draft);
          markAutomationSetupHandoff(conversation.conversation_id);
          clearAutomationFormSession(PENDING_AUTOMATION_SETUP_ID);
          navigate?.(`/conversations/${conversation.conversation_id}`);
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
