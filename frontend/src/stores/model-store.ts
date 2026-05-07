import { create } from "zustand";
import { devtools } from "zustand/middleware";
import type { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";

export interface ModelListEntry {
  id: string;
  /**
   * Id of the chat event after which this entry should render, or `null` to
   * pin it to the top of the chat history (no events at the time of /model).
   */
  anchorEventId: string | null;
  profiles: LlmProfileSummary[];
  switchedTo?: string;
}

interface ModelState {
  entriesByConversation: Record<string, ModelListEntry[]>;
}

interface ModelActions {
  show: (
    conversationId: string,
    anchorEventId: string | null,
    profiles: LlmProfileSummary[],
  ) => void;
  recordSwitch: (
    conversationId: string,
    anchorEventId: string | null,
    profileName: string,
  ) => void;
}

type ModelStore = ModelState & ModelActions;

export const useModelStore = create<ModelStore>()(
  devtools(
    (set) => ({
      entriesByConversation: {},
      show: (conversationId, anchorEventId, profiles) =>
        set((s) => ({
          entriesByConversation: {
            ...s.entriesByConversation,
            [conversationId]: [
              ...(s.entriesByConversation[conversationId] ?? []),
              { id: crypto.randomUUID(), anchorEventId, profiles },
            ],
          },
        })),
      recordSwitch: (conversationId, anchorEventId, profileName) =>
        set((s) => ({
          entriesByConversation: {
            ...s.entriesByConversation,
            [conversationId]: [
              ...(s.entriesByConversation[conversationId] ?? []),
              {
                id: crypto.randomUUID(),
                anchorEventId,
                profiles: [],
                switchedTo: profileName,
              },
            ],
          },
        })),
    }),
    { name: "ModelStore" },
  ),
);
