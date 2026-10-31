import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import { useNavigation } from "#/context/navigation-context";
import { useConversationStore } from "#/stores/conversation-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import {
  consumePendingTaskAttachments,
  usePendingTaskAttachmentsStore,
} from "#/stores/pending-task-attachments-store";
import { I18nKey } from "#/i18n/declaration";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { ConversationStartupProgress } from "./conversation-startup-progress";

export function TaskStartupProgress() {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const queryClient = useQueryClient();
  const { taskId, task, taskStatus, taskDetail, taskError, isLoadingTask } =
    useTaskPolling();
  const attachmentFailure = usePendingTaskAttachmentsStore((state) =>
    taskId ? state.byTaskId[taskId] : undefined,
  );
  if (!taskId) return null;
  const failed = taskStatus === "ERROR";
  const missingTask = !isLoadingTask && task === null;
  const error =
    attachmentFailure?.error ||
    (failed
      ? taskDetail || t(I18nKey.CONVERSATION$FAILED_TO_START_FROM_TASK)
      : taskError
        ? getApiErrorMessage(taskError, t(I18nKey.ERROR$GENERIC))
        : missingTask
          ? t(I18nKey.ERROR$GENERIC)
          : null);

  const editFailedStart = () => {
    if (attachmentFailure?.error && attachmentFailure.conversationId) {
      usePendingTaskAttachmentsStore
        .getState()
        .setPendingTaskAttachments(taskId, {
          ...attachmentFailure,
          recoverInConversation: true,
        });
      navigate(`/conversations/${attachmentFailure.conversationId}`);
      return;
    }
    const attachments = consumePendingTaskAttachments(taskId);
    const messages = useOptimisticUserMessageStore.getState();
    const pending = messages.pendingMessages.filter(
      (message) => message.conversationId === `task-${taskId}`,
    );
    const initialText =
      task?.request.initial_message?.content
        .filter((part) => part.type === "text")
        .map((part) => part.text)
        .join("\n") ?? "";
    const store = useConversationStore.getState();
    store.setMessageToSend(
      attachments?.content ?? pending[0]?.text ?? initialText,
      "home",
    );
    if (attachments) {
      store.addImages(attachments.images);
      store.addFiles(attachments.files);
      for (const name of attachments.imagesMarkedUploadAsFile) {
        if (!store.imagesMarkedUploadAsFile.includes(name))
          store.toggleImageUploadAsFile(name);
      }
    }
    pending.forEach((message) => messages.removePendingMessage(message.id));
    navigate("/conversations");
  };

  return (
    <ConversationStartupProgress
      key={taskId}
      editLabel={
        attachmentFailure?.error
          ? t(I18nKey.CONVERSATION$OPEN_CREATED)
          : undefined
      }
      status={taskStatus}
      error={error}
      onEdit={failed || attachmentFailure?.error ? editFailedStart : undefined}
      onRetry={
        (taskError || missingTask) && !failed && !attachmentFailure?.error
          ? () => {
              void queryClient.refetchQueries({
                queryKey: ["start-task", taskId],
                exact: true,
              });
            }
          : undefined
      }
    />
  );
}
