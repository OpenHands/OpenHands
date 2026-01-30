import { create } from "zustand";
import { devtools } from "zustand/middleware";
import {
  getConversationState,
  setConversationState,
} from "#/utils/conversation-local-storage";

export type ConversationTab =
  | "editor"
  | "browser"
  | "served"
  | "vscode"
  | "terminal"
  | "planner";

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
  loadingFiles: string[]; // File names currently being processed
  loadingImages: string[]; // Image names currently being processed
  messageToSend: IMessageToSend | null;
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

const parseStoredBoolean = (value: string | null): boolean | null => {
  if (value === null) {
    return null;
  }

  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
};

const getInitialRightPanelState = (): boolean => {
  if (typeof window === "undefined") {
    return true;
  }

  const conversationId = getConversationIdFromLocation();
  const keysToCheck = conversationId
    ? [`conversation-right-panel-shown-${conversationId}`]
    : [];

  // Fallback to legacy global key for users who haven't switched tabs yet
  keysToCheck.push("conversation-right-panel-shown");

  for (const key of keysToCheck) {
    const parsed = parseStoredBoolean(localStorage.getItem(key));
    if (parsed !== null) {
      return parsed;
    }
  }

  return true;
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
      // Initial state
      isRightPanelShown: getInitialRightPanelState(),
      selectedTab: "editor" as ConversationTab,
      images: [],
      files: [],
      loadingFiles: [],
      loadingImages: [],
      messageToSend: null,
      shouldShownAgentLoading: false,
      submittedMessage: null,
      shouldHideSuggestions: false,
      hasRightPanelToggled: true,
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

      removeImage: (index) =>
        set(
          (state) => {
            const newImages = [...state.images];
            newImages.splice(index, 1);
            return { images: newImages };
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
