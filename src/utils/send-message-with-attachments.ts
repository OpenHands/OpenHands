import { getActiveBackend } from "#/api/backend-registry/active-store";
import { assertActiveEnvironment } from "#/utils/assert-active-environment";
import type { TFunction } from "i18next";
import {
  resolveConversationRuntime,
  uploadFilesToConversation,
} from "#/api/conversation-file-upload.api";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { SendMessageRequest } from "#/api/conversation-service/agent-server-conversation-service.types";
import { convertImageToBase64 } from "#/utils/convert-image-to-base-64";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { partitionImagesForUpload } from "#/components/features/chat/utils/chat-input.utils";
import {
  resolveUploadLimits,
  validateFiles,
  validateIndividualFileSizes,
} from "#/utils/file-validation";
import { MAX_EMBEDDED_IMAGE_BYTES } from "#/utils/provider-upload-limits";
import { I18nKey } from "#/i18n/declaration";

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
  /** The provider that will receive the attachments, for its image limit. */
  provider?: string | null;
}): Promise<SendMessageWithAttachmentsResult> {
  const environment = getActiveBackend();
  const {
    conversationId,
    content,
    images,
    files,
    imagesMarkedUploadAsFile,
    t,
    provider,
  } = options;

  const { imagesToEmbed, imagesAsFiles } = partitionImagesForUpload(
    images,
    imagesMarkedUploadAsFile,
  );
  const filesToUpload = [...files, ...imagesAsFiles];

  const limits = resolveUploadLimits(provider ?? null);

  // An embedded image is base64-encoded into the chat event, which the agent
  // server forwards to the browser as a WebSocket frame. The server drops the
  // connection for a frame larger than 4 MiB, so the embed path stays bounded
  // by the transport even when the provider would accept more.
  const embedLimitBytes = Math.min(
    limits.imageLimitBytes,
    MAX_EMBEDDED_IMAGE_BYTES,
  );
  const embedValidation = validateIndividualFileSizes(imagesToEmbed, {
    ...limits,
    imageLimitBytes: embedLimitBytes,
    imageLimitMb: embedLimitBytes / (1024 * 1024),
  });
  if (!embedValidation.isValid) {
    throw new Error(embedValidation.errorMessage ?? "Invalid attachments");
  }

  // Files uploaded to the workspace travel over HTTP and are bounded only by
  // the provider's limit for images and by the generic limit for other types.
  const uploadValidation = validateFiles(filesToUpload, [], limits);
  if (!uploadValidation.isValid) {
    throw new Error(uploadValidation.errorMessage ?? "Invalid attachments");
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

  const filePrompt = `${t(I18nKey.CHAT_INTERFACE$AUGMENTED_PROMPT_FILES_TITLE)}: ${uploadedFiles.join("\n\n")}`;
  const prompt =
    uploadedFiles.length > 0 ? `${content}\n\n${filePrompt}` : content;

  const timestamp = new Date().toISOString();

  const messageContent: SendMessageRequest = {
    role: "user",
    content: [{ type: "text", text: prompt }],
  };

  if (imageUrls.length > 0) {
    messageContent.content.push({
      type: "image",
      image_urls: imageUrls,
    });
  }

  assertActiveEnvironment(environment);
  await AgentServerConversationService.sendMessage(
    conversationId,
    messageContent,
    runtime,
  );

  return {
    text: content,
    content: prompt,
    imageUrls,
    fileUrls: uploadedFiles,
    timestamp,
  };
}
