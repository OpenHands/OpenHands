import { useCallback, useMemo, useRef, useState } from "react";
import { useNavigation } from "#/context/navigation-context";
import {
  STAGED_AGENT_NOTIFICATIONS,
  isAgentNotificationsStagingEnabled,
  type AgentNotification,
} from "#/components/features/chat/agent-notifications.constants";
import { writeAgentNotificationPendingPrompts } from "#/components/features/chat/agent-notifications-pending-prompts";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useIsCreatingConversation } from "#/hooks/use-is-creating-conversation";
import {
  setConversationState,
  setPendingTaskDraft,
} from "#/utils/conversation-local-storage";

export function useSidebarOnboardingAgentNotifications() {
  const { navigate } = useNavigation();
  const createConversation = useCreateConversation();
  const isCreatingConversation = useIsCreatingConversation();
  const [isCreating, setIsCreating] = useState(false);
  const launchInFlightRef = useRef(false);

  const agentNotifications = useMemo((): AgentNotification[] => {
    if (isAgentNotificationsStagingEnabled()) {
      return STAGED_AGENT_NOTIFICATIONS;
    }
    return [];
  }, []);

  const hasAgentNotifications = agentNotifications.length > 0;

  const createAll = useCallback(
    (selectedIds: string[]) => {
      if (
        selectedIds.length === 0 ||
        launchInFlightRef.current ||
        createConversation.isPending ||
        isCreatingConversation ||
        isCreating
      ) {
        return;
      }

      const selectedAgentNotifications = agentNotifications.filter(
        (agentNotification) => selectedIds.includes(agentNotification.id),
      );
      if (selectedAgentNotifications.length === 0) {
        return;
      }

      const prompts = selectedAgentNotifications.map(
        (agentNotification) => agentNotification.prompt,
      );

      launchInFlightRef.current = true;
      setIsCreating(true);

      createConversation.mutate(
        {},
        {
          onSuccess: (conversation) => {
            const conversationId = conversation.conversation_id;
            writeAgentNotificationPendingPrompts(conversationId, prompts);

            const firstPrompt = prompts[0];
            if (
              conversationId.startsWith("task-") &&
              conversation.task_id &&
              firstPrompt
            ) {
              setPendingTaskDraft(conversation.task_id, firstPrompt);
            } else if (firstPrompt) {
              setConversationState(conversationId, {
                draftMessage: firstPrompt,
              });
            }

            navigate?.(`/conversations/${conversationId}`);
          },
          onError: () => {
            launchInFlightRef.current = false;
            setIsCreating(false);
          },
          onSettled: () => {
            launchInFlightRef.current = false;
            setIsCreating(false);
          },
        },
      );
    },
    [
      createConversation,
      isCreating,
      isCreatingConversation,
      navigate,
      agentNotifications,
    ],
  );

  return {
    agentNotifications,
    hasAgentNotifications,
    createAll,
    isCreating:
      isCreating || createConversation.isPending || isCreatingConversation,
  };
}
