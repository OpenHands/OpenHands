import { useMutation } from "@tanstack/react-query";
import { Feedback } from "#/api/open-hands.types";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { useConversationId } from "#/hooks/use-conversation-id";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

type SubmitFeedbackArgs = {
  feedback: Feedback;
};

export const useSubmitFeedback = () => {
  const { conversationId } = useConversationId();

  return useMutation({
    mutationFn: ({ feedback }: SubmitFeedbackArgs) =>
      ConversationService.submitFeedback(conversationId, feedback),
    onError: (error) => {
      displayErrorToast(error.message);
    },
    retry: 2,
    retryDelay: 500,
  });
};
