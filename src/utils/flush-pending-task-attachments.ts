import { getActiveBackend } from "#/api/backend-registry/active-store";
import { assertActiveEnvironment } from "#/utils/assert-active-environment";
import i18n, { OPENHANDS_I18N_NAMESPACE, waitForI18n } from "#/i18n";
import {
  consumePendingTaskAttachments,
  usePendingTaskAttachmentsStore,
} from "#/stores/pending-task-attachments-store";
import { I18nKey } from "#/i18n/declaration";
import { sendMessageWithAttachments } from "#/utils/send-message-with-attachments";

/**
 * Sends attachments queued during cloud start-task provisioning once the
 * real conversation UUID is available.
 */
const inFlight = new Map<string, Promise<void>>();

export async function flushPendingTaskAttachments(
  taskId: string,
  conversationId: string,
): Promise<void> {
  const existing = inFlight.get(taskId);
  if (existing) return existing;
  const store = usePendingTaskAttachmentsStore.getState();
  const pending = store.byTaskId[taskId];
  if (!pending) return;
  if (pending.error) throw new Error(pending.error);
  store.setPendingTaskAttachments(taskId, { ...pending, conversationId });
  const environment = getActiveBackend();
  const delivery = (async () => {
    try {
      await waitForI18n();
      assertActiveEnvironment({
        backend: {
          ...environment.backend,
          id: pending.backendId ?? environment.backend.id,
        },
        orgId: pending.backendId ? (pending.orgId ?? null) : environment.orgId,
      });
      await sendMessageWithAttachments({
        ...pending,
        conversationId,
        requireAllAttachments: true,
        t: i18n.getFixedT(null, OPENHANDS_I18N_NAMESPACE),
      });
      consumePendingTaskAttachments(taskId);
    } catch (error) {
      store.setPendingTaskAttachments(taskId, {
        ...pending,
        conversationId,
        recoverInConversation: true,
        error:
          error instanceof Error
            ? error.message
            : i18n.t(I18nKey.ERROR$GENERIC),
      });
      throw error;
    } finally {
      inFlight.delete(taskId);
    }
  })();
  inFlight.set(taskId, delivery);
  return delivery;
}
