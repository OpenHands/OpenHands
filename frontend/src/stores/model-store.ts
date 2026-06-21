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
  /**
   * The profile most recently switched-to per conversation, set optimistically
   * when a switch succeeds. The chat switch button reads this so it reflects
   * the choice immediately and unambiguously — without waiting for the
   * conversation record's `llm_model` to round-trip back (which can lag, and in
   * SaaS collides when several managed profiles share the same `litellm_proxy/`
   * model string).
   */
  activeProfileByConversation: Record<string, string>;
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
      activeProfileByConversation: {},
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
          activeProfileByConversation: {
            ...s.activeProfileByConversation,
            [conversationId]: profileName,
          },
        })),
    }),
    { name: "ModelStore" },
  ),
);
