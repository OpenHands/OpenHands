import type { TFunction } from "i18next";
import type { MessageContent } from "#/api/conversation-service/agent-server-conversation-service.types";
import { I18nKey } from "#/i18n/declaration";

/**
 * The prompt text a user message carries: the typed text with uploaded-file
 * references appended under the localized files title (files have no
 * dedicated content part on the wire). Shared by the payload builder and by
 * the optimistic bubble, whose text must match the server's echo exactly.
 */
export function buildUserMessagePrompt(options: {
  text: string;
  fileUrls?: string[];
  t: TFunction;
}): string {
  const { text, fileUrls = [], t } = options;
  if (fileUrls.length === 0) return text;
  const filePrompt = `${t(I18nKey.CHAT_INTERFACE$AUGMENTED_PROMPT_FILES_TITLE)}: ${fileUrls.join("\n\n")}`;
  return `${text}\n\n${filePrompt}`;
}

/**
 * The single assembler for user chat message content. Both send paths — the
 * conversation WebSocket (`useSendMessage`) and the agent-server REST client
 * (`sendMessageWithAttachments`) — build their payload through this, so
 * attachments and uploaded-file references cannot diverge between them.
 */
export function buildUserMessageContent(options: {
  text: string;
  imageUrls?: string[];
  fileUrls?: string[];
  t: TFunction;
}): MessageContent[] {
  const { text, imageUrls = [], fileUrls = [], t } = options;

  if (!text && imageUrls.length === 0 && fileUrls.length === 0) {
    throw new Error("Cannot send an empty message.");
  }

  const content: MessageContent[] = [
    { type: "text", text: buildUserMessagePrompt({ text, fileUrls, t }) },
  ];
  if (imageUrls.length > 0) {
    content.push({ type: "image", image_urls: imageUrls });
  }
  return content;
}
