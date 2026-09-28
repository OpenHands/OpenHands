import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import {
  PENDING_AUTOMATION_SETUP_ID,
  clearAutomationSetupDraft,
  getAutomationSetupDraft,
  setAutomationSetupDraft,
} from "#/api/automation-setup-draft-store";
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
    setAutomationSetupDraft(PENDING_AUTOMATION_SETUP_ID, {
      prompt: "",
      kind: "prompt",
    });
    navigate?.("/automations/setup");
  }, [active.backend.kind, navigate, trackAutomationCreatedButton]);

  const startConversationFromPrompt = useCallback(
    (prompt: string) => {
      const text = prompt.trim();
      if (!text || createConversation.isPending) return;
      const draft = getAutomationSetupDraft(PENDING_AUTOMATION_SETUP_ID) ?? {
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
            setAutomationSetupDraft(conversation.conversation_id, draft);
            clearAutomationSetupDraft(PENDING_AUTOMATION_SETUP_ID);
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
