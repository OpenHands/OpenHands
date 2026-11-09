import { useCallback, useRef } from "react";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";

interface AutomationSetupConversation {
  conversation_id: string;
}

interface StartAutomationSetupConversationOptions {
  query: string;
  entryPoint: "automations_add" | "automation_edit" | "automation_draft_resume";
  automationSetupTags?: Record<string, string>;
  automationSetupContext?: string;
  onStart: () => void;
  onError: (error: unknown) => void;
  onSuccess: (
    conversation: AutomationSetupConversation,
  ) => void | Promise<void>;
  onSettled?: () => void;
}

export function useCreateAutomationSetupConversation() {
  const createConversation = useCreateConversation({ disableToast: true });
  const inFlight = useRef(false);

  const startAutomationSetupConversation = useCallback(
    ({
      query,
      entryPoint,
      automationSetupTags,
      automationSetupContext,
      onStart,
      onError,
      onSuccess,
      onSettled,
    }: StartAutomationSetupConversationOptions) => {
      const text = query.trim();
      if (!text || inFlight.current || createConversation.isPending)
        return false;
      inFlight.current = true;
      onStart();

      // Promise handlers survive navigation unmounting this composer.
      void createConversation
        .mutateAsync({
          query: text,
          automationSetup: true,
          automationSetupContext,
          ...(automationSetupTags ? { automationSetupTags } : {}),
          entryPoint,
        })
        .then(onSuccess, onError)
        .finally(() => {
          inFlight.current = false;
          onSettled?.();
        });
      return true;
    },
    [createConversation],
  );

  return {
    startAutomationSetupConversation,
    isPending: createConversation.isPending,
  };
}
