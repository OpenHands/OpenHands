import { useMutation } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import toast from "react-hot-toast";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
  TOAST_OPTIONS,
} from "#/utils/custom-toast-handlers";
import { useNavigation } from "#/context/navigation-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";

export const useNewConversationCommand = () => {
  const { navigate } = useNavigation();
  const { t } = useTranslation("openhands");
  const { data: conversation } = useActiveConversation();
  const { mutateAsync: createConversation } = useCreateConversation();

  const mutation = useMutation({
    mutationFn: async () => {
      if (!conversation?.id) {
        throw new Error("No active conversation");
      }

      // Reuse the current sandbox without creating a parent/child relation.
      // Delegating to the shared New Chat mutation also preserves the active
      // AgentProfile/LLM profile and its local conversation metadata.
      return createConversation({
        sandboxId: conversation.sandbox_id ?? undefined,
        entryPoint: "new_command",
      });
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
      navigate(`/conversations/${data.conversation_id}`);
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
