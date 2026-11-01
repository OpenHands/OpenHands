import { getActiveBackend } from "#/api/backend-registry/active-store";
import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import { useErrorMessageStore } from "#/stores/error-message-store";
import {
  consumePendingTaskAttachments,
  usePendingTaskAttachmentsStore,
} from "#/stores/pending-task-attachments-store";
import { useConversationStore } from "#/stores/conversation-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";

export const STARTUP_UPLOAD_ERROR_CODE = "StartupAttachmentUploadFailed";

export function restoreStartupAttachmentDraft(conversationId: string): void {
  const environment = getActiveBackend();
  for (const [pendingTaskId, payload] of Object.entries(
    usePendingTaskAttachmentsStore.getState().byTaskId,
  )) {
    if (
      !payload.recoverInConversation ||
      payload.conversationId !== conversationId ||
      (payload.backendId &&
        (payload.backendId !== environment.backend.id ||
          payload.orgId !== environment.orgId))
    )
      continue;
    consumePendingTaskAttachments(pendingTaskId);
    const composer = useConversationStore.getState();
    composer.addImages(payload.images);
    composer.addFiles(payload.files);
    for (const name of payload.imagesMarkedUploadAsFile) {
      if (!composer.imagesMarkedUploadAsFile.includes(name))
        composer.toggleImageUploadAsFile(name);
    }
    composer.setMessageToSend(payload.content, "conversation");
    useErrorMessageStore
      .getState()
      .setErrorMessage(
        i18n.t(I18nKey.CONVERSATION$START_UPLOAD_FAILED),
        "conversation",
        STARTUP_UPLOAD_ERROR_CODE,
      );
    const messages = useOptimisticUserMessageStore.getState();
    messages.pendingMessages
      .filter((message) => message.conversationId === `task-${pendingTaskId}`)
      .forEach((message) => messages.removePendingMessage(message.id));
  }
}
