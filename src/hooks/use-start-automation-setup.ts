import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import {
  PENDING_AUTOMATION_SETUP_ID,
  clearAutomationFormSession,
  getAutomationFormSession,
  initializeAutomationFormSession,
} from "#/api/automation-form-session";
import { markAutomationSetupHandoff } from "#/api/automation-setup-handoff-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useTracking } from "#/hooks/use-tracking";
import { I18nKey } from "#/i18n/declaration";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

/**
 * Open a new automation on the setup form.
 *
 * The automations page used to create an agent conversation immediately.
 * The form now opens with the conversation drawer hidden, and a conversation
 * starts only after the user sends a prompt.
 */
export function useStartAutomationSetup() {
  const { t } = useTranslation("openhands");
  const active = useActiveBackend();
  const { navigate } = useNavigation();
  const createConversation = useCreateConversation();
  const { trackAutomationCreatedButton } = useTracking();

  const startSetup = useCallback(() => {
    trackAutomationCreatedButton({ backendKind: active.backend.kind });
    navigate?.("/automations/setup");
  }, [active.backend.kind, navigate, trackAutomationCreatedButton]);

  const startConversationFromPrompt = useCallback(
    (prompt: string) => {
      const text = prompt.trim();
      if (!text || createConversation.isPending) return;
      const draft = getAutomationFormSession(PENDING_AUTOMATION_SETUP_ID) ?? {
        prompt: "",
        kind: "prompt" as const,
      };
      createConversation.mutate(
        {
          query: text,
          automationSetup: true,
          entryPoint: "automations_add",
        },
        {
          onSuccess: (conversation) => {
            initializeAutomationFormSession(
              conversation.conversation_id,
              draft,
            );
            markAutomationSetupHandoff(conversation.conversation_id);
            clearAutomationFormSession(PENDING_AUTOMATION_SETUP_ID);
            navigate?.(`/conversations/${conversation.conversation_id}`);
          },
          onError: (error) => {
            displayErrorToast(
              getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)),
            );
          },
        },
      );
    },
    [createConversation, navigate, t],
  );

  return {
    startSetup,
    startConversationFromPrompt,
    isPending: createConversation.isPending,
  };
}
