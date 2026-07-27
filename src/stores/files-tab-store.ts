import { create } from "zustand";

interface FilesTabState {
  selectedPath: string | null;
  // The conversation a selection belongs to. A file picked in one
  // conversation must not leak into another (it usually doesn't exist in the
  // other conversation's workspace, see issue #1350), so every selection is
  // tagged with its conversation and the files tab ignores a path owned by a
  // different conversation.
  selectedConversationId: string | null;
  setSelectedPath: (
    path: string | null,
    conversationId?: string | null,
  ) => void;
}

// Hoisted out of files-tab.tsx local state so non-React callers (e.g. the
// canvas_ui tool dispatcher in the WebSocket context) can drive selection.
export const useFilesTabStore = create<FilesTabState>((set) => ({
  selectedPath: null,
  selectedConversationId: null,
  setSelectedPath: (selectedPath, conversationId = null) =>
    set({ selectedPath, selectedConversationId: conversationId }),
}));
