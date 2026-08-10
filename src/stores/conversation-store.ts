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

interface ConversationState {
  isRightPanelShown: boolean;
  selectedTab: ConversationTab | null;
  images: File[];
  files: File[];
  /** Image file names (e.g. pasted screenshots) to send via file upload instead of vision embed. */
  imagesMarkedUploadAsFile: string[];
  /** Image file names attached in chat (controls per-image upload-as-file UI). */
  pastedImageNames: string[];
  loadingFiles: string[]; // File names currently being processed
  loadingImages: string[]; // Image names currently being processed
  messageToSend: IMessageToSend | null;
  /** One-shot restore request consumed by the chat input when empty. */
  messageRestoreIfEmpty: IMessageToSend | null;
  shouldShownAgentLoading: boolean;
  submittedMessage: string | null;
  shouldHideSuggestions: boolean; // New state to hide suggestions when input expands
  hasRightPanelToggled: boolean;
  planContent: string | null;
  conversationMode: ConversationMode;
  subConversationTaskId: string | null; // Task ID for sub-conversation creation
}

interface ConversationActions {
  setIsRightPanelShown: (isRightPanelShown: boolean) => void;
  setSelectedTab: (selectedTab: ConversationTab | null) => void;
  setShouldShownAgentLoading: (shouldShownAgentLoading: boolean) => void;
  setShouldHideSuggestions: (shouldHideSuggestions: boolean) => void;
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
      images: [],
      files: [],
      imagesMarkedUploadAsFile: [],
      pastedImageNames: [],
      loadingFiles: [],
      loadingImages: [],
      messageToSend: null,
      messageRestoreIfEmpty: null,
      shouldShownAgentLoading: false,
      submittedMessage: null,
      shouldHideSuggestions: false,
      hasRightPanelToggled: false,
      planContent: null,
      conversationMode: getInitialConversationMode(),
      subConversationTaskId: null,

      // Actions
      setIsRightPanelShown: (isRightPanelShown) =>
        set({ isRightPanelShown }, false, "setIsRightPanelShown"),

      setSelectedTab: (selectedTab) =>
        set({ selectedTab }, false, "setSelectedTab"),

      setShouldShownAgentLoading: (shouldShownAgentLoading) =>
        set({ shouldShownAgentLoading }, false, "setShouldShownAgentLoading"),

      setShouldHideSuggestions: (shouldHideSuggestions) =>
        set({ shouldHideSuggestions }, false, "setShouldHideSuggestions"),

      addImages: (images) =>
        set(
          (state) => ({ images: [...state.images, ...images] }),
          false,
          "addImages",
        ),

      addFiles: (files) =>
        set(
          (state) => ({ files: [...state.files, ...files] }),
          false,
          "addFiles",
        ),

      toggleImageUploadAsFile: (fileName) =>
        set(
          (state) => {
            const marked = new Set(state.imagesMarkedUploadAsFile);
            if (marked.has(fileName)) {
              marked.delete(fileName);
            } else {
              marked.add(fileName);
            }
            return { imagesMarkedUploadAsFile: [...marked] };
          },
          false,
          "toggleImageUploadAsFile",
        ),

      markImagesAsPasted: (fileNames) =>
        set(
          (state) => {
            const merged = new Set([...state.pastedImageNames, ...fileNames]);
            return { pastedImageNames: [...merged] };
          },
          false,
          "markImagesAsPasted",
        ),

      removeImage: (index) =>
        set(
          (state) => {
            const removed = state.images[index];
            const newImages = [...state.images];
            newImages.splice(index, 1);
            return {
              images: newImages,
              imagesMarkedUploadAsFile: removed
                ? state.imagesMarkedUploadAsFile.filter(
                    (name) => name !== removed.name,
                  )
                : state.imagesMarkedUploadAsFile,
              pastedImageNames: removed
                ? state.pastedImageNames.filter((name) => name !== removed.name)
                : state.pastedImageNames,
            };
          },
          false,
          "removeImage",
        ),

      removeFile: (index) =>
        set(
          (state) => {
            const newFiles = [...state.files];
            newFiles.splice(index, 1);
            return { files: newFiles };
          },
          false,
          "removeFile",
        ),

      clearImages: () => set({ images: [] }, false, "clearImages"),

      clearFiles: () => set({ files: [] }, false, "clearFiles"),

      clearAllFiles: () =>
        set(
          {
            images: [],
            files: [],
            imagesMarkedUploadAsFile: [],
            pastedImageNames: [],
            loadingFiles: [],
            loadingImages: [],
          },
          false,
          "clearAllFiles",
        ),

      addFileLoading: (fileName) =>
        set(
          (state) => {
            if (!state.loadingFiles.includes(fileName)) {
              return { loadingFiles: [...state.loadingFiles, fileName] };
            }
            return state;
          },
          false,
          "addFileLoading",
        ),

      removeFileLoading: (fileName) =>
        set(
          (state) => ({
            loadingFiles: state.loadingFiles.filter(
              (name) => name !== fileName,
            ),
          }),
          false,
          "removeFileLoading",
        ),

      addImageLoading: (imageName) =>
        set(
          (state) => {
            if (!state.loadingImages.includes(imageName)) {
              return { loadingImages: [...state.loadingImages, imageName] };
            }
            return state;
          },
          false,
          "addImageLoading",
        ),

      removeImageLoading: (imageName) =>
        set(
          (state) => ({
            loadingImages: state.loadingImages.filter(
              (name) => name !== imageName,
            ),
          }),
          false,
          "removeImageLoading",
        ),

      clearAllLoading: () =>
        set({ loadingFiles: [], loadingImages: [] }, false, "clearAllLoading"),

      setMessageToSend: (text) =>
        set(
          {
            messageToSend: {
              text,
              timestamp: Date.now(),
            },
          },
          false,
          "setMessageToSend",
        ),

      // One-shot consume: clear after the composer applies it, so a never-sent
      // value can't replay into another conversation's composer on remount.
      clearMessageToSend: () =>
        set({ messageToSend: null }, false, "clearMessageToSend"),

      restoreMessageToInputIfEmpty: (text) =>
        set(
          {
            messageRestoreIfEmpty: {
              text,
              timestamp: Date.now(),
            },
          },
          false,
          "restoreMessageToInputIfEmpty",
        ),

      clearMessageRestoreIfEmpty: () =>
        set(
          { messageRestoreIfEmpty: null },
          false,
          "clearMessageRestoreIfEmpty",
        ),

      setSubmittedMessage: (submittedMessage) =>
        set({ submittedMessage }, false, "setSubmittedMessage"),

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
