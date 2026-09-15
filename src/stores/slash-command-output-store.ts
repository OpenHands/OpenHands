import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { v4 as uuidv4 } from "uuid";
import type { SlashCommandItem } from "#/types/slash-command";

interface SlashCommandOutputBase {
  id: string;
  anchorEventId: string | null;
}

export type SlashCommandOutputEntry = SlashCommandOutputBase & {
  kind: "help";
  commands: SlashCommandItem[];
};

interface SlashCommandOutputState {
  entriesByConversation: Record<string, SlashCommandOutputEntry[]>;
  showHelp: (
    conversationId: string,
    anchorEventId: string | null,
    commands: SlashCommandItem[],
  ) => void;
  clear: (conversationId: string) => void;
  clearAll: () => void;
}

function appendEntry(
  entriesByConversation: SlashCommandOutputState["entriesByConversation"],
  conversationId: string,
  entry: SlashCommandOutputEntry,
) {
  return {
    ...entriesByConversation,
    [conversationId]: [...(entriesByConversation[conversationId] ?? []), entry],
  };
}

export const useSlashCommandOutputStore = create<SlashCommandOutputState>()(
  devtools(
    (set) => ({
      entriesByConversation: {},
      showHelp: (conversationId, anchorEventId, commands) =>
        set((state) => ({
          entriesByConversation: appendEntry(
            state.entriesByConversation,
            conversationId,
            { id: uuidv4(), anchorEventId, kind: "help", commands },
          ),
        })),
      clear: (conversationId) =>
        set((state) => {
          const entriesByConversation = { ...state.entriesByConversation };
          delete entriesByConversation[conversationId];
          return { entriesByConversation };
        }),
      clearAll: () => set({ entriesByConversation: {} }),
    }),
    { name: "SlashCommandOutputStore" },
  ),
);
