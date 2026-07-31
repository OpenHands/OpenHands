import { useCallback } from "react";
import { useScopedConversationId } from "#/hooks/use-conversation-events";
import {
  HOME_COMPOSER_KEY,
  getComposerBucket,
  useConversationStore,
  type ConversationComposerBucket,
  type IMessageToSend,
} from "#/stores/conversation-store";

export type ConversationComposer = ConversationComposerBucket & {
  conversationKey: string;
  addImages: (images: File[]) => void;
  addFiles: (files: File[]) => void;
  toggleImageUploadAsFile: (fileName: string) => void;
  markImagesAsPasted: (fileNames: string[]) => void;
  removeImage: (index: number) => void;
  removeFile: (index: number) => void;
  clearImages: () => void;
  clearFiles: () => void;
  clearAllFiles: () => void;
  addFileLoading: (fileName: string) => void;
  removeFileLoading: (fileName: string) => void;
  addImageLoading: (imageName: string) => void;
  removeImageLoading: (imageName: string) => void;
  clearAllLoading: () => void;
  setMessageToSend: (text: string) => void;
  clearMessageToSend: () => void;
  restoreMessageToInputIfEmpty: (text: string) => void;
  clearMessageRestoreIfEmpty: () => void;
  setSubmittedMessage: (message: string | null) => void;
  clearComposer: () => void;
};

/**
 * Composer state + actions for the conversation currently in navigation
 * scope (primary route or nested popout). Falls back to {@link HOME_COMPOSER_KEY}
 * when no conversation id is available (home launcher).
 */
export function useConversationComposer(
  conversationId?: string | null,
): ConversationComposer {
  const scopedId = useScopedConversationId(conversationId);
  const conversationKey = scopedId ?? HOME_COMPOSER_KEY;

  const bucket = useConversationStore((state) =>
    getComposerBucket(state, conversationKey),
  );

  const addImages = useCallback(
    (images: File[]) => {
      useConversationStore.getState().addImages(conversationKey, images);
    },
    [conversationKey],
  );
  const addFiles = useCallback(
    (files: File[]) => {
      useConversationStore.getState().addFiles(conversationKey, files);
    },
    [conversationKey],
  );
  const toggleImageUploadAsFile = useCallback(
    (fileName: string) => {
      useConversationStore
        .getState()
        .toggleImageUploadAsFile(conversationKey, fileName);
    },
    [conversationKey],
  );
  const markImagesAsPasted = useCallback(
    (fileNames: string[]) => {
      useConversationStore
        .getState()
        .markImagesAsPasted(conversationKey, fileNames);
    },
    [conversationKey],
  );
  const removeImage = useCallback(
    (index: number) => {
      useConversationStore.getState().removeImage(conversationKey, index);
    },
    [conversationKey],
  );
  const removeFile = useCallback(
    (index: number) => {
      useConversationStore.getState().removeFile(conversationKey, index);
    },
    [conversationKey],
  );
  const clearImages = useCallback(() => {
    useConversationStore.getState().clearImages(conversationKey);
  }, [conversationKey]);
  const clearFiles = useCallback(() => {
    useConversationStore.getState().clearFiles(conversationKey);
  }, [conversationKey]);
  const clearAllFiles = useCallback(() => {
    useConversationStore.getState().clearAllFiles(conversationKey);
  }, [conversationKey]);
  const addFileLoading = useCallback(
    (fileName: string) => {
      useConversationStore.getState().addFileLoading(conversationKey, fileName);
    },
    [conversationKey],
  );
  const removeFileLoading = useCallback(
    (fileName: string) => {
      useConversationStore
        .getState()
        .removeFileLoading(conversationKey, fileName);
    },
    [conversationKey],
  );
  const addImageLoading = useCallback(
    (imageName: string) => {
      useConversationStore
        .getState()
        .addImageLoading(conversationKey, imageName);
    },
    [conversationKey],
  );
  const removeImageLoading = useCallback(
    (imageName: string) => {
      useConversationStore
        .getState()
        .removeImageLoading(conversationKey, imageName);
    },
    [conversationKey],
  );
  const clearAllLoading = useCallback(() => {
    useConversationStore.getState().clearAllLoading(conversationKey);
  }, [conversationKey]);
  const setMessageToSend = useCallback(
    (text: string) => {
      useConversationStore.getState().setMessageToSend(conversationKey, text);
    },
    [conversationKey],
  );
  const clearMessageToSend = useCallback(() => {
    useConversationStore.getState().clearMessageToSend(conversationKey);
  }, [conversationKey]);
  const restoreMessageToInputIfEmpty = useCallback(
    (text: string) => {
      useConversationStore
        .getState()
        .restoreMessageToInputIfEmpty(conversationKey, text);
    },
    [conversationKey],
  );
  const clearMessageRestoreIfEmpty = useCallback(() => {
    useConversationStore.getState().clearMessageRestoreIfEmpty(conversationKey);
  }, [conversationKey]);
  const setSubmittedMessage = useCallback(
    (message: string | null) => {
      useConversationStore
        .getState()
        .setSubmittedMessage(conversationKey, message);
    },
    [conversationKey],
  );
  const clearComposer = useCallback(() => {
    useConversationStore.getState().clearComposer(conversationKey);
  }, [conversationKey]);

  return {
    conversationKey,
    images: bucket.images,
    files: bucket.files,
    imagesMarkedUploadAsFile: bucket.imagesMarkedUploadAsFile,
    pastedImageNames: bucket.pastedImageNames,
    loadingFiles: bucket.loadingFiles,
    loadingImages: bucket.loadingImages,
    messageToSend: bucket.messageToSend,
    messageRestoreIfEmpty: bucket.messageRestoreIfEmpty,
    submittedMessage: bucket.submittedMessage,
    addImages,
    addFiles,
    toggleImageUploadAsFile,
    markImagesAsPasted,
    removeImage,
    removeFile,
    clearImages,
    clearFiles,
    clearAllFiles,
    addFileLoading,
    removeFileLoading,
    addImageLoading,
    removeImageLoading,
    clearAllLoading,
    setMessageToSend,
    clearMessageToSend,
    restoreMessageToInputIfEmpty,
    clearMessageRestoreIfEmpty,
    setSubmittedMessage,
    clearComposer,
  };
}

/** Read a composer field without subscribing (imperative / non-React paths). */
export function getConversationComposer(
  conversationId: string | null | undefined,
): ConversationComposerBucket {
  const key = conversationId ?? HOME_COMPOSER_KEY;
  return getComposerBucket(useConversationStore.getState(), key);
}

export type { IMessageToSend };
