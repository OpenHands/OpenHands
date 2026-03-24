import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router";
import { useTranslation } from "react-i18next";
import toast from "react-hot-toast";
import { I18nKey } from "#/i18n/declaration";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import {
  displayErrorToast,
  displaySuccessToast,
  TOAST_OPTIONS,
} from "#/utils/custom-toast-handlers";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";

export const useNewConversationCommand = () => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { t } = useTranslation();
  const { data: conversation } = useActiveConversation();

  const mutation = useMutation({
    mutationFn: async () => {
      if (!conversation?.conversation_id || !conversation.sandbox_id) {
        throw new Error("No active conversation or sandbox");
      }

      // Fetch V1 conversation data to get llm_model (not available in legacy type)
      const v1Conversations =
        await V1ConversationService.batchGetAppConversations([
          conversation.conversation_id,
        ]);
      const llmModel = v1Conversations?.[0]?.llm_model;

      // Start a new conversation reusing the existing sandbox directly.
      // We pass sandbox_id instead of parent_conversation_id so that the
      // new conversation is NOT marked as a sub-conversation and will
      // appear in the conversation list.
      const startTask = await V1ConversationService.createConversation(
        conversation.selected_repository ?? undefined, // selectedRepository
        conversation.git_provider ?? undefined, // git_provider
        undefined, // initialUserMsg
        conversation.selected_branch ?? undefined, // selected_branch
        undefined, // conversationInstructions
        undefined, // suggestedTask
        undefined, // trigger
        undefined, // parent_conversation_id
        undefined, // agent_type
        undefined, // plugins
        conversation.sandbox_id ?? undefined, // sandbox_id - reuse the same sandbox
        llmModel ?? undefined, // llm_model - preserve the LLM model
      );

      // Poll for the task to complete and get the new conversation ID
      let task = await V1ConversationService.getStartTask(startTask.id);
      const maxAttempts = 60; // 60 seconds timeout
      let attempts = 0;

      /* eslint-disable no-await-in-loop */
      while (
        task &&
        !["READY", "ERROR"].includes(task.status) &&
        attempts < maxAttempts
      ) {
        // eslint-disable-next-line no-await-in-loop
        await new Promise((resolve) => {
          setTimeout(resolve, 1000);
        });
        task = await V1ConversationService.getStartTask(startTask.id);
        attempts += 1;
      }

      if (!task || task.status !== "READY" || !task.app_conversation_id) {
        throw new Error(
          task?.detail || "Failed to create new conversation in sandbox",
        );
      }

      return {
        newConversationId: task.app_conversation_id,
        oldConversationId: conversation.conversation_id,
      };
    },
    onMutate: () => {
      toast.loading(t(I18nKey.CONVERSATION$CLEARING), {
        ...TOAST_OPTIONS,
        id: "clear-conversation",
      });
    },
    onSuccess: (data) => {
      toast.dismiss("clear-conversation");
      displaySuccessToast(t(I18nKey.CONVERSATION$CLEAR_SUCCESS));
      navigate(`/conversations/${data.newConversationId}`);

      // Refresh the sidebar to show the new conversation.
      queryClient.invalidateQueries({
        queryKey: ["user", "conversations"],
      });
      queryClient.invalidateQueries({
        queryKey: ["v1-batch-get-app-conversations"],
      });
    },
    onError: (error) => {
      toast.dismiss("clear-conversation");
      let clearError = t(I18nKey.CONVERSATION$CLEAR_UNKNOWN_ERROR);
      if (error instanceof Error) {
        clearError = error.message;
      } else if (typeof error === "string") {
        clearError = error;
      }
      displayErrorToast(
        t(I18nKey.CONVERSATION$CLEAR_FAILED, { error: clearError }),
      );
    },
  });

  return mutation;
};
