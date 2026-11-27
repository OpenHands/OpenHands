import { partitionImagesForUpload } from "#/components/features/chat/utils/chat-input.utils";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { convertImageToBase64 } from "#/utils/convert-image-to-base-64";

/**
 * Shows the user's first message immediately on cloud start-task routes
 * (`/conversations/task-{uuid}`) while the sandbox provisions.
 */
export async function enqueueHomeTaskPendingMessage(options: {
  conversationId: string;
  text: string;
  images: File[];
  imagesMarkedUploadAsFile: string[];
}): Promise<void> {
  const { imagesToEmbed } = partitionImagesForUpload(
    options.images,
    options.imagesMarkedUploadAsFile,
  );
  const imageUrls = await Promise.all(
    imagesToEmbed.map((image) => convertImageToBase64(image)),
  );

  useOptimisticUserMessageStore.getState().enqueuePendingMessage({
    conversationId: options.conversationId,
    text: options.text,
    content: options.text,
    imageUrls,
    fileUrls: [],
  });
}
