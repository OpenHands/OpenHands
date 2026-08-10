import { useCallback } from "react";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useIsCreatingConversation } from "#/hooks/use-is-creating-conversation";
import { useTracking } from "#/hooks/use-tracking";
import { useConversationStore } from "#/stores/conversation-store";
import {
  setConversationState,
  setPendingTaskDraft,
} from "#/utils/conversation-local-storage";

export type AutomationConversationIntent =
  | "find_opportunities"
  | "add_automation";

export type AutomationConversationSource =
  | "dashboard_header"
  | "templates_banner"
  | "empty_state";

export interface AutomationConversationLaunchRequest {
  intent: AutomationConversationIntent;
  source: AutomationConversationSource;
  prompt: string;
}

interface LaunchAutomationConversationOptions {
  onSuccess?: () => void;
  onError?: () => void;
}

export function useLaunchAutomationConversation() {
  const active = useActiveBackend();
  const { navigate } = useNavigation();
  const createConversation = useCreateConversation();
  const isCreatingConversation = useIsCreatingConversation();
  const setMessageToSend = useConversationStore(
    (state) => state.setMessageToSend,
  );
  const { trackAutomationCreatedButton } = useTracking();

  const launchAutomationConversation = useCallback(
    (
      request: AutomationConversationLaunchRequest,
      options: LaunchAutomationConversationOptions = {},
    ) => {
      if (createConversation.isPending || isCreatingConversation) {
        return;
      }

      trackAutomationCreatedButton({
        backendKind: active.backend.kind,
        intent: request.intent,
        source: request.source,
      });

      createConversation.mutate(
        {},
        {
          onSuccess: (conversation) => {
            if (
              conversation.conversation_id.startsWith("task-") &&
              conversation.task_id
            ) {
              setPendingTaskDraft(conversation.task_id, request.prompt);
            } else {
              setConversationState(conversation.conversation_id, {
                draftMessage: request.prompt,
              });
            }

            navigate(`/conversations/${conversation.conversation_id}`);
            options.onSuccess?.();
            window.setTimeout(() => setMessageToSend(request.prompt), 0);
          },
          onError: () => {
            options.onError?.();
          },
        },
      );
    },
    [
      active.backend.kind,
      createConversation,
      isCreatingConversation,
      navigate,
      setMessageToSend,
      trackAutomationCreatedButton,
    ],
  );

  return {
    launchAutomationConversation,
    isLaunching: createConversation.isPending || isCreatingConversation,
  };
}
