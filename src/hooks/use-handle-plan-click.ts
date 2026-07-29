import { useCallback, useEffect, type MouseEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useConversationStore } from "#/stores/conversation-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { displaySuccessToast } from "#/utils/custom-toast-handlers";
import {
  getConversationState,
  setConversationState,
} from "#/utils/conversation-local-storage";
import { useActiveBackend } from "#/contexts/active-backend-context";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { getStoredConversationMetadata } from "#/api/conversation-metadata-store";
import {
  CONVERSATION_QUERY_KEYS,
  LOCAL_PLANNER_MUTATION_KEYS,
} from "#/hooks/query/query-keys";

function useCreateLocalPlanningConversationMutation(options: {
  onCreated: (planningConversationId: string) => void;
  onInitialized: () => void;
}) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationKey: LOCAL_PLANNER_MUTATION_KEYS.create,
    mutationFn: (parentConversationId: string) =>
      AgentServerConversationService.createLocalPlanningConversation(
        parentConversationId,
      ),
    onSuccess: (planningConversation) => {
      options.onCreated(planningConversation.id);
      queryClient.invalidateQueries({ queryKey: CONVERSATION_QUERY_KEYS.all });
      queryClient.invalidateQueries({
        queryKey: CONVERSATION_QUERY_KEYS.subConversations,
      });
      options.onInitialized();
    },
  });
}

function restorePlanningConversationIds(options: {
  conversationId: string;
  subConversationTaskId: string | null;
  localPlanningConversationId: string | null;
  setSubConversationTaskId: (taskId: string | null) => void;
  setLocalPlanningConversationId: (conversationId: string | null) => void;
}) {
  const storedState = getConversationState(options.conversationId);
  if (storedState.subConversationTaskId && !options.subConversationTaskId) {
    options.setSubConversationTaskId(storedState.subConversationTaskId);
  }

  const metadata = getStoredConversationMetadata(options.conversationId);
  if (
    metadata?.local_planning_conversation_id &&
    !options.localPlanningConversationId
  ) {
    options.setLocalPlanningConversationId(
      metadata.local_planning_conversation_id,
    );
  }
}

/**
 * Custom hook that encapsulates the logic for handling plan creation.
 * Returns a function that can be called to create a plan conversation and
 * the pending state of the conversation creation.
 *
 * @returns An object containing handlePlanClick function and isCreatingConversation boolean
 */
export const useHandlePlanClick = () => {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const {
    setConversationMode,
    setSubConversationTaskId,
    subConversationTaskId,
    setLocalPlanningConversationId,
    localPlanningConversationId,
  } = useConversationStore();
  const { data: conversation } = useActiveConversation();
  const { mutate: createConversation, isPending: isCreatingCloudConversation } =
    useCreateConversation();
  const {
    mutate: createLocalPlanningConversation,
    isPending: isCreatingLocalPlanningConversation,
  } = useCreateLocalPlanningConversationMutation({
    onCreated: setLocalPlanningConversationId,
    onInitialized: () => {
      displaySuccessToast(
        t(I18nKey.PLANNING_AGENTT$PLANNING_AGENT_INITIALIZED),
      );
    },
  });

  // Restore planning conversation ids on conversation load. This handles page
  // refreshes while cloud or local planning conversation creation is in progress.
  useEffect(() => {
    if (!conversation?.id) return;

    restorePlanningConversationIds({
      conversationId: conversation.id,
      subConversationTaskId,
      localPlanningConversationId,
      setSubConversationTaskId,
      setLocalPlanningConversationId,
    });
  }, [
    conversation?.id,
    localPlanningConversationId,
    setLocalPlanningConversationId,
    subConversationTaskId,
    setSubConversationTaskId,
  ]);

  const handlePlanClick = useCallback(
    (event?: MouseEvent<HTMLButtonElement> | KeyboardEvent) => {
      event?.preventDefault();
      event?.stopPropagation();

      setConversationMode("plan");

      if (backend.kind !== "cloud") {
        if (!conversation?.id || localPlanningConversationId) {
          return;
        }
        createLocalPlanningConversation(conversation.id);
        return;
      }

      if (
        (conversation?.sub_conversation_ids &&
          conversation.sub_conversation_ids.length > 0) ||
        !conversation?.id ||
        subConversationTaskId
      ) {
        return;
      }

      createConversation(
        {
          parentConversationId: conversation.id,
          agentType: "plan",
          entryPoint: "plan_sub_conversation",
        },
        {
          onSuccess: (data) => {
            displaySuccessToast(
              t(I18nKey.PLANNING_AGENTT$PLANNING_AGENT_INITIALIZED),
            );
            if (data.task_id) {
              setSubConversationTaskId(data.task_id);
              setConversationState(conversation.id, {
                subConversationTaskId: data.task_id,
              });
            }
          },
        },
      );
    },
    [
      backend.kind,
      conversation,
      createConversation,
      createLocalPlanningConversation,
      localPlanningConversationId,
      setConversationMode,
      setSubConversationTaskId,
      subConversationTaskId,
      t,
    ],
  );

  return {
    handlePlanClick,
    isCreatingConversation:
      isCreatingCloudConversation || isCreatingLocalPlanningConversation,
  };
};
