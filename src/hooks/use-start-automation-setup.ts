/**
 * Start a blank automation in the setup page.
 *
 * Add Automation used to stop on an instructions modal. It now opens the
 * same conversation-and-form page as every other setup, with an empty prompt
 * the user or the agent can fill in.
 */
import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { setAutomationSetupDraft } from "#/api/automation-setup-draft-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useTracking } from "#/hooks/use-tracking";
import { I18nKey } from "#/i18n/declaration";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

export function useStartAutomationSetup() {
  const { t } = useTranslation("openhands");
  const active = useActiveBackend();
  const { navigate } = useNavigation();
  const createConversation = useCreateConversation();
  const { trackAutomationCreatedButton } = useTracking();

  const startSetup = useCallback(() => {
    trackAutomationCreatedButton({ backendKind: active.backend.kind });
    createConversation.mutate(
      {
        query: t(I18nKey.AUTOMATIONS$CREATE_AUTOMATION_PROMPT),
        automationSetup: true,
        entryPoint: "automations_add",
      },
      {
        onSuccess: (conversation) => {
          setAutomationSetupDraft(conversation.conversation_id, {
            prompt: "",
            kind: "prompt",
          });
          navigate?.(`/conversations/${conversation.conversation_id}`);
        },
        onError: (error) => {
          displayErrorToast(
            getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)),
          );
        },
      },
    );
  }, [
    active.backend.kind,
    createConversation,
    navigate,
    t,
    trackAutomationCreatedButton,
  ]);

  return { startSetup, isPending: createConversation.isPending };
}
