import { getActiveBackend } from "#/api/backend-registry/active-store";
import { assertActiveEnvironment } from "#/utils/assert-active-environment";
import type { TFunction } from "i18next";
import {
  resolveConversationRuntime,
  uploadFilesToConversation,
} from "#/api/conversation-file-upload.api";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { convertImageToBase64 } from "#/utils/convert-image-to-base-64";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { partitionImagesForUpload } from "#/components/features/chat/utils/chat-input.utils";
import { validateFiles } from "#/utils/file-validation";
import {
  buildUserMessageContent,
  buildUserMessagePrompt,
} from "#/utils/send-message-content";

export interface SendMessageWithAttachmentsResult {
  text: string;
  content: string;
  imageUrls: string[];
  fileUrls: string[];
  timestamp: string;
}

export async function sendMessageWithAttachments(options: {
  conversationId: string;
  content: string;
  images: File[];
  files: File[];
  imagesMarkedUploadAsFile: string[];
  t: TFunction;
  requireAllAttachments?: boolean;
}): Promise<SendMessageWithAttachmentsResult> {
  const environment = getActiveBackend();
  const {
    conversationId,
    content,
    images,
    files,
    imagesMarkedUploadAsFile,
    t,
  } = options;

  const { imagesToEmbed, imagesAsFiles } = partitionImagesForUpload(
    images,
    imagesMarkedUploadAsFile,
  );
  const filesToUpload = [...files, ...imagesAsFiles];

  const validation = validateFiles([...imagesToEmbed, ...filesToUpload]);
  if (!validation.isValid) {
    throw new Error(validation.errorMessage ?? "Invalid attachments");
  }

  const imageUrls = await Promise.all(
    imagesToEmbed.map((image) => convertImageToBase64(image)),
  );

  assertActiveEnvironment(environment);
  const runtime = await resolveConversationRuntime(conversationId);
  assertActiveEnvironment(environment);

  const { skipped_files: skippedFiles, uploaded_files: uploadedFiles } =
    filesToUpload.length > 0
      ? await uploadFilesToConversation(conversationId, filesToUpload)
      : { skipped_files: [], uploaded_files: [] };

  if (options.requireAllAttachments && skippedFiles.length > 0) {
    throw new Error(skippedFiles.map((file) => file.reason).join("\n"));
  }
  skippedFiles.forEach((file) => displayErrorToast(file.reason));

  const timestamp = new Date().toISOString();

  const messageContent = {
    role: "user" as const,
    content: buildUserMessageContent({
      text: content,
      imageUrls,
      fileUrls: uploadedFiles,
      t,
    }),
  };

  assertActiveEnvironment(environment);
  await AgentServerConversationService.sendMessage(
    conversationId,
    messageContent,
    runtime,
  );

  return {
    text: content,
    // The optimistic bubble mirrors the sent prompt, file augmentation
    // included.
    content: buildUserMessagePrompt({
      text: content,
      fileUrls: uploadedFiles,
      t,
    }),
    imageUrls,
    fileUrls: uploadedFiles,
    timestamp,
  };
}
