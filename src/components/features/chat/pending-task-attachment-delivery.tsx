import { getApiErrorMessage } from "#/utils/api-error-message";
import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { usePendingTaskAttachmentsStore } from "#/stores/pending-task-attachments-store";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { flushPendingTaskAttachments } from "#/utils/flush-pending-task-attachments";
import { isRetryableError } from "#/api/with-retry";

function TaskDelivery({ taskId }: { taskId: string }) {
  const { data: task, error } = useQuery({
    queryKey: ["start-task", taskId],
    queryFn: () => AgentServerConversationService.getStartTask(taskId),
    retry: (count, error) => count < 3 && isRetryableError(error),
    meta: { disableToast: true },
    refetchInterval: (query) =>
      query.state.status === "error" ||
      query.state.data === null ||
      query.state.data?.status === "READY" ||
      query.state.data?.status === "ERROR"
        ? false
        : 3000,
  });
  useEffect(() => {
    const store = usePendingTaskAttachmentsStore.getState();
    const pending = store.byTaskId[taskId];
    if (!pending) return;
    const readError = error
      ? getApiErrorMessage(error, i18n.t(I18nKey.ERROR$GENERIC))
      : undefined;
    if (pending.readError !== readError) {
      store.setPendingTaskAttachments(taskId, { ...pending, readError });
    }
  }, [error, taskId]);
  useEffect(() => {
    if (task?.status !== "READY" || !task.app_conversation_id) return;
    // Delivery owns the error state; failed sends require explicit user recovery.
    void flushPendingTaskAttachments(taskId, task.app_conversation_id).catch(
      () => {},
    );
  }, [task, taskId]);
  return null;
}

/** Keep delivery alive while the user visits other pages in this environment. */
export function PendingTaskAttachmentDelivery() {
  const { backend, orgId } = useActiveBackend();
  const pending = usePendingTaskAttachmentsStore((state) => state.byTaskId);
  return Object.entries(pending)
    .filter(
      ([, item]) =>
        !item.error && item.backendId === backend.id && item.orgId === orgId,
    )
    .map(([taskId]) => <TaskDelivery key={taskId} taskId={taskId} />);
}
