import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { randomUUID } from "#/utils/random-uuid";

/**
 * A rendered-in-chat answer to a display-only slash command (/help, /skills).
 * Unlike /btw entries these are synchronous: the body is complete when added.
 */
export interface SlashCommandInfoEntry {
  id: string;
  title: string;
  body: string;
}

interface SlashCommandInfoState {
  entriesByConversation: Record<string, SlashCommandInfoEntry[]>;
}

interface SlashCommandInfoActions {
  addInfo: (conversationId: string, title: string, body: string) => string;
  dismiss: (conversationId: string, id: string) => void;
}

type SlashCommandInfoStore = SlashCommandInfoState & SlashCommandInfoActions;

const initialState: SlashCommandInfoState = { entriesByConversation: {} };

const updateEntries = (
  state: SlashCommandInfoState,
  conversationId: string,
  updater: (entries: SlashCommandInfoEntry[]) => SlashCommandInfoEntry[],
): Pick<SlashCommandInfoState, "entriesByConversation"> => ({
  entriesByConversation: {
    ...state.entriesByConversation,
    [conversationId]: updater(
      state.entriesByConversation[conversationId] ?? [],
    ),
  },
});

export const useSlashCommandInfoStore = create<SlashCommandInfoStore>()(
  devtools(
    (set) => ({
      ...initialState,
      addInfo: (conversationId, title, body) => {
        const id = randomUUID();
        set((s) =>
          updateEntries(s, conversationId, (entries) => [
            ...entries,
            { id, title, body },
          ]),
        );
        return id;
      },
      dismiss: (conversationId, id) =>
        set((s) =>
          updateEntries(s, conversationId, (entries) =>
            entries.filter((e) => e.id !== id),
          ),
        ),
    }),
    { name: "SlashCommandInfoStore" },
  ),
);
