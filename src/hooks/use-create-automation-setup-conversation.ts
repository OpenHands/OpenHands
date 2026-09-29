import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { I18nKey } from "#/i18n/declaration";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

interface AutomationSetupConversation {
  conversation_id: string;
}

interface StartAutomationSetupConversationOptions {
  query: string;
  entryPoint: "automations_add" | "automation_edit" | "automation_draft_resume";
  onSuccess: (
    conversation: AutomationSetupConversation,
  ) => void | Promise<void>;
  onSettled?: () => void;
}

export function useCreateAutomationSetupConversation() {
  const { t } = useTranslation("openhands");
  const createConversation = useCreateConversation();

  const startAutomationSetupConversation = useCallback(
    ({
      query,
      entryPoint,
      onSuccess,
      onSettled,
    }: StartAutomationSetupConversationOptions) => {
      const text = query.trim();
      if (!text || createConversation.isPending) return false;

      createConversation.mutate(
        {
          query: text,
          automationSetup: true,
          entryPoint,
        },
        {
          onSuccess,
          onError: (error) => {
            displayErrorToast(
              getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)),
            );
          },
          onSettled,
        },
      );
      return true;
    },
    [createConversation, t],
  );

  return {
    startAutomationSetupConversation,
    isPending: createConversation.isPending,
  };
}
