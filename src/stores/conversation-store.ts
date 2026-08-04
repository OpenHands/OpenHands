import { create } from "zustand";
import { devtools } from "zustand/middleware";
import {
  getConversationState,
  setConversationState,
} from "#/utils/conversation-local-storage";

export type ConversationTab =
  | "files"
  | "browser"
  | "terminal"
  | "planner"
  | "tasklist";

export type ConversationMode = "code" | "plan";

export interface IMessageToSend {
  text: string;
  timestamp: number;
}

/**
 * Composer bucket key used on the home page (and other surfaces with no
 * conversation id yet). Attachments / programmatic sends for an in-progress
 * launch live here until a real conversation id exists.
 */
export const HOME_COMPOSER_KEY = "__home__";

/**
 * Mutable composer state for one conversation (or the home launcher).
 * Keyed separately from process-wide chrome so primary + popout composers do
 * not share attachments, drafts, or one-shot message commands.
 */
export interface ConversationComposerBucket {
  images: File[];
  files: File[];
  /** Image file names (e.g. pasted screenshots) to send via file upload instead of vision embed. */
  imagesMarkedUploadAsFile: string[];
  /** Image file names attached in chat (controls per-image upload-as-file UI). */
  pastedImageNames: string[];
  loadingFiles: string[];
  loadingImages: string[];
  messageToSend: IMessageToSend | null;
  /** One-shot restore request consumed by the chat input when empty. */
  messageRestoreIfEmpty: IMessageToSend | null;
  submittedMessage: string | null;
}

export const EMPTY_COMPOSER_BUCKET: ConversationComposerBucket = Object.freeze({
  images: Object.freeze([]) as unknown as File[],
  files: Object.freeze([]) as unknown as File[],
  imagesMarkedUploadAsFile: Object.freeze([]) as unknown as string[],
  pastedImageNames: Object.freeze([]) as unknown as string[],
  loadingFiles: Object.freeze([]) as unknown as string[],
  loadingImages: Object.freeze([]) as unknown as string[],
  messageToSend: null,
  messageRestoreIfEmpty: null,
  submittedMessage: null,
});

const createComposerBucket = (): ConversationComposerBucket => ({
  images: [],
  files: [],
  imagesMarkedUploadAsFile: [],
  pastedImageNames: [],
  loadingFiles: [],
  loadingImages: [],
  messageToSend: null,
  messageRestoreIfEmpty: null,
  submittedMessage: null,
});

interface ConversationChromeState {
  isRightPanelShown: boolean;
  selectedTab: ConversationTab | null;
  shouldShownAgentLoading: boolean;
  shouldHideSuggestions: boolean;
  hasRightPanelToggled: boolean;
  planContent: string | null;
  conversationMode: ConversationMode;
  subConversationTaskId: string | null;
}

interface ConversationState extends ConversationChromeState {
  byConversation: Record<string, ConversationComposerBucket>;
}

interface ConversationActions {
  setIsRightPanelShown: (isRightPanelShown: boolean) => void;
  setSelectedTab: (selectedTab: ConversationTab | null) => void;
  setShouldShownAgentLoading: (shouldShownAgentLoading: boolean) => void;
  setShouldHideSuggestions: (shouldHideSuggestions: boolean) => void;
  addImages: (conversationId: string, images: File[]) => void;
  addFiles: (conversationId: string, files: File[]) => void;
  toggleImageUploadAsFile: (conversationId: string, fileName: string) => void;
  markImagesAsPasted: (conversationId: string, fileNames: string[]) => void;
  removeImage: (conversationId: string, index: number) => void;
  removeFile: (conversationId: string, index: number) => void;
  clearImages: (conversationId: string) => void;
  clearFiles: (conversationId: string) => void;
  clearAllFiles: (conversationId: string) => void;
  addFileLoading: (conversationId: string, fileName: string) => void;
  removeFileLoading: (conversationId: string, fileName: string) => void;
  addImageLoading: (conversationId: string, imageName: string) => void;
  removeImageLoading: (conversationId: string, imageName: string) => void;
  clearAllLoading: (conversationId: string) => void;
  setMessageToSend: (conversationId: string, text: string) => void;
  clearMessageToSend: (conversationId: string) => void;
  restoreMessageToInputIfEmpty: (conversationId: string, text: string) => void;
  clearMessageRestoreIfEmpty: (conversationId: string) => void;
  setSubmittedMessage: (conversationId: string, message: string | null) => void;
  clearComposer: (conversationId: string) => void;
  resetConversationState: () => void;
  setHasRightPanelToggled: (hasRightPanelToggled: boolean) => void;
  setConversationMode: (conversationMode: ConversationMode) => void;
  setSubConversationTaskId: (taskId: string | null) => void;
  setPlanContent: (planContent: string | null) => void;
}

type ConversationStore = ConversationState & ConversationActions;

const getConversationIdFromLocation = (): string | null => {
  if (typeof window === "undefined") {
    return null;
  }

  const match = window.location.pathname.match(/\/conversations\/([^/]+)/);
  return match ? match[1] : null;
};

const getInitialConversationMode = (): ConversationMode => {
  if (typeof window === "undefined") {
    return "code";
  }

  const conversationId = getConversationIdFromLocation();
  if (!conversationId) {
    return "code";
  }

  const state = getConversationState(conversationId);
  return state.conversationMode;
};

export const getComposerBucket = (
  state: Pick<ConversationState, "byConversation">,
  conversationId: string,
): ConversationComposerBucket =>
  state.byConversation[conversationId] ?? EMPTY_COMPOSER_BUCKET;

const withComposerBucket = (
  state: ConversationState,
  conversationId: string,
  bucket: ConversationComposerBucket,
): ConversationState => ({
  ...state,
  byConversation: {
    ...state.byConversation,
    [conversationId]: bucket,
  },
});

const updateComposerBucket = (
  state: ConversationState,
  conversationId: string,
  updater: (bucket: ConversationComposerBucket) => ConversationComposerBucket,
): ConversationState => {
  const current =
    state.byConversation[conversationId] ?? createComposerBucket();
  const next = updater(current);
  if (next === current && state.byConversation[conversationId]) {
    return state;
  }
  return withComposerBucket(state, conversationId, next);
};

export const useConversationStore = create<ConversationStore>()(
  devtools(
    (set) => ({
      // Initial state.
      //
      // The right-side drawer (`isRightPanelShown` / `hasRightPanelToggled`)
      // is intentionally *session-only* state: it always starts closed on
      // app load (or on opening a fresh/existing conversation after a
      // restart), but it survives in-app navigation because the Zustand
      // store stays alive across React Router transitions. Persisting the
      // open/closed state in localStorage made the panel feel sticky in
      // a way users didn't expect — they want a clean, focused chat view
      // when they come back to the app and only want the panel back when
      // they themselves opened it during the current session.
      isRightPanelShown: false,
      selectedTab: "files" as ConversationTab,
      shouldShownAgentLoading: false,
      shouldHideSuggestions: false,
      hasRightPanelToggled: false,
      planContent: null,
      conversationMode: getInitialConversationMode(),
      subConversationTaskId: null,
      byConversation: {},

      // Actions
      setIsRightPanelShown: (isRightPanelShown) =>
        set({ isRightPanelShown }, false, "setIsRightPanelShown"),

      setSelectedTab: (selectedTab) =>
        set({ selectedTab }, false, "setSelectedTab"),

      setShouldShownAgentLoading: (shouldShownAgentLoading) =>
        set({ shouldShownAgentLoading }, false, "setShouldShownAgentLoading"),

      setShouldHideSuggestions: (shouldHideSuggestions) =>
        set({ shouldHideSuggestions }, false, "setShouldHideSuggestions"),

      addImages: (conversationId, images) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => ({
              ...bucket,
              images: [...bucket.images, ...images],
            })),
          false,
          "addImages",
        ),

      addFiles: (conversationId, files) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => ({
              ...bucket,
              files: [...bucket.files, ...files],
            })),
          false,
          "addFiles",
        ),

      toggleImageUploadAsFile: (conversationId, fileName) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => {
              const marked = new Set(bucket.imagesMarkedUploadAsFile);
              if (marked.has(fileName)) {
                marked.delete(fileName);
              } else {
                marked.add(fileName);
              }
              return { ...bucket, imagesMarkedUploadAsFile: [...marked] };
            }),
          false,
          "toggleImageUploadAsFile",
        ),

      markImagesAsPasted: (conversationId, fileNames) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => {
              const merged = new Set([
                ...bucket.pastedImageNames,
                ...fileNames,
              ]);
              return { ...bucket, pastedImageNames: [...merged] };
            }),
          false,
          "markImagesAsPasted",
        ),

      removeImage: (conversationId, index) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => {
              const removed = bucket.images[index];
              const newImages = [...bucket.images];
              newImages.splice(index, 1);
              return {
                ...bucket,
                images: newImages,
                imagesMarkedUploadAsFile: removed
                  ? bucket.imagesMarkedUploadAsFile.filter(
                      (name) => name !== removed.name,
                    )
                  : bucket.imagesMarkedUploadAsFile,
                pastedImageNames: removed
                  ? bucket.pastedImageNames.filter(
                      (name) => name !== removed.name,
                    )
                  : bucket.pastedImageNames,
              };
            }),
          false,
          "removeImage",
        ),

      removeFile: (conversationId, index) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => {
              const newFiles = [...bucket.files];
              newFiles.splice(index, 1);
              return { ...bucket, files: newFiles };
            }),
          false,
          "removeFile",
        ),

      clearImages: (conversationId) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => ({
              ...bucket,
              images: [],
            })),
          false,
          "clearImages",
        ),

      clearFiles: (conversationId) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => ({
              ...bucket,
              files: [],
            })),
          false,
          "clearFiles",
        ),

      clearAllFiles: (conversationId) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => ({
              ...bucket,
              images: [],
              files: [],
              imagesMarkedUploadAsFile: [],
              pastedImageNames: [],
              loadingFiles: [],
              loadingImages: [],
            })),
          false,
          "clearAllFiles",
        ),

      addFileLoading: (conversationId, fileName) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => {
              if (bucket.loadingFiles.includes(fileName)) {
                return bucket;
              }
              return {
                ...bucket,
                loadingFiles: [...bucket.loadingFiles, fileName],
              };
            }),
          false,
          "addFileLoading",
        ),

      removeFileLoading: (conversationId, fileName) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => ({
              ...bucket,
              loadingFiles: bucket.loadingFiles.filter(
                (name) => name !== fileName,
              ),
            })),
          false,
          "removeFileLoading",
        ),

      addImageLoading: (conversationId, imageName) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => {
              if (bucket.loadingImages.includes(imageName)) {
                return bucket;
              }
              return {
                ...bucket,
                loadingImages: [...bucket.loadingImages, imageName],
              };
            }),
          false,
          "addImageLoading",
        ),

      removeImageLoading: (conversationId, imageName) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => ({
              ...bucket,
              loadingImages: bucket.loadingImages.filter(
                (name) => name !== imageName,
              ),
            })),
          false,
          "removeImageLoading",
        ),

      clearAllLoading: (conversationId) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => ({
              ...bucket,
              loadingFiles: [],
              loadingImages: [],
            })),
          false,
          "clearAllLoading",
        ),

      setMessageToSend: (conversationId, text) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => ({
              ...bucket,
              messageToSend: {
                text,
                timestamp: Date.now(),
              },
            })),
          false,
          "setMessageToSend",
        ),

      // One-shot consume: clear after the composer applies it, so a never-sent
      // value can't replay into another conversation's composer on remount.
      clearMessageToSend: (conversationId) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => ({
              ...bucket,
              messageToSend: null,
            })),
          false,
          "clearMessageToSend",
        ),

      restoreMessageToInputIfEmpty: (conversationId, text) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => ({
              ...bucket,
              messageRestoreIfEmpty: {
                text,
                timestamp: Date.now(),
              },
            })),
          false,
          "restoreMessageToInputIfEmpty",
        ),

      clearMessageRestoreIfEmpty: (conversationId) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => ({
              ...bucket,
              messageRestoreIfEmpty: null,
            })),
          false,
          "clearMessageRestoreIfEmpty",
        ),

      setSubmittedMessage: (conversationId, submittedMessage) =>
        set(
          (state) =>
            updateComposerBucket(state, conversationId, (bucket) => ({
              ...bucket,
              submittedMessage,
            })),
          false,
          "setSubmittedMessage",
        ),

      clearComposer: (conversationId) =>
        set((state) => {
          if (!(conversationId in state.byConversation)) {
            return state;
          }
          const { [conversationId]: _removed, ...byConversation } =
            state.byConversation;
          return { byConversation };
        }),

      resetConversationState: () =>
        set(
          {
            shouldHideSuggestions: false,
            conversationMode: getInitialConversationMode(),
            subConversationTaskId: null,
            planContent: null,
          },
          false,
          "resetConversationState",
        ),

      setHasRightPanelToggled: (hasRightPanelToggled) =>
        set({ hasRightPanelToggled }, false, "setHasRightPanelToggled"),

      setConversationMode: (conversationMode) => {
        const conversationId = getConversationIdFromLocation();
        if (conversationId) {
          setConversationState(conversationId, { conversationMode });
        }
        set({ conversationMode }, false, "setConversationMode");
      },

      setSubConversationTaskId: (subConversationTaskId) =>
        set({ subConversationTaskId }, false, "setSubConversationTaskId"),

      setPlanContent: (planContent) =>
        set({ planContent }, false, "setPlanContent"),
    }),
    {
      name: "conversation-store",
    },
  ),
);
