import { create } from "zustand";
import { devtools } from "zustand/middleware";

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
}

type ConversationStore = ConversationState & ConversationActions;

// Helper function to get initial right panel state from localStorage
const getInitialRightPanelState = (): boolean => {
  const stored = localStorage.getItem("conversation-right-panel-shown");
  return stored !== null ? JSON.parse(stored) : true;
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
      planContent: `
# Improve Developer Onboarding and Examples

## Overview

Based on the analysis of Browser-Use's current documentation and examples, this plan addresses gaps in developer onboarding by creating a progressive learning path, troubleshooting resources, and practical examples that address real-world scenarios (like the LM Studio/local LLM integration issues encountered).

## Current State Analysis

**Strengths:**

- Good quickstart documentation in \`docs/quickstart.mdx\`
- Extensive examples across multiple categories (60+ example files)
- Well-structured docs with multiple LLM provider examples
- Active community support via Discord

**Gaps Identified:**

- No progressive tutorial series that builds complexity gradually
- Limited troubleshooting documentation for common issues
- Sparse comments in example files explaining what's happening
- Local LLM setup (Ollama/LM Studio) not prominently featured
- No "first 10 minutes" success path
- Missing visual/conceptual architecture guides for beginners
- Error messages don't always point to solutions

## Proposed Improvements

### 1. Create Interactive Tutorial Series (\`examples/tutorials/\`)

**New folder structure:**

\`\`\`
examples/tutorials/
├── README.md              # Tutorial overview and prerequisites
├── 00_hello_world.py      # Absolute minimal example
├── 01_your_first_search.py # Basic search with detailed comments
├── 02_understanding_actions.py # How actions work
├── 03_data_extraction_basics.py # Extract data step-by-step
├── 04_error_handling.py   # Common errors and solutions
├── 05_custom_tools_intro.py # First custom tool
├── 06_local_llm_setup.py  # Ollama/LM Studio complete guide
└── 07_debugging_tips.py   # Debugging strategies
\`\`\`

**Key Features:**

- Each file 50–80 lines max
- Extensive inline comments explaining every concept
- Clear learning objectives at the top of each file
- "What you'll learn" and "Prerequisites" sections
- Common pitfalls highlighted
- Expected output shown in comments

### 2. Troubleshooting Guide (\`docs/troubleshooting.mdx\`)

**Sections:**

- Installation issues (Chromium, dependencies, virtual environments)
- LLM provider connection errors (API keys, timeouts, rate limits)
- Local LLM setup (Ollama vs LM Studio, model compatibility)
- Browser automation issues (element not found, timeout errors)
- Common error messages with solutions
- Performance optimization tips
- When to ask for help (Discord/GitHub)

**Format:**

**Error: "LLM call timed out after 60 seconds"**

**What it means:**
The model took too long to respond

**Common causes:**

1. Model is too slow for the task
2. LM Studio/Ollama not responding properly
3. Complex page overwhelming the model

**Solutions:**

- Use flash_mode for faster execution
- Try a faster model (Gemini Flash, GPT-4 Turbo Mini)
- Simplify the task
- Check model server logs`,
      conversationMode: "code",
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
            conversationMode: "code",
            subConversationTaskId: null,
          },
          false,
          "resetConversationState",
        ),

      setHasRightPanelToggled: (hasRightPanelToggled) =>
        set({ hasRightPanelToggled }, false, "setHasRightPanelToggled"),

      setConversationMode: (conversationMode) =>
        set({ conversationMode }, false, "setConversationMode"),

      setSubConversationTaskId: (subConversationTaskId) =>
        set({ subConversationTaskId }, false, "setSubConversationTaskId"),
    }),
    {
      name: "conversation-store",
    },
  ),
);
