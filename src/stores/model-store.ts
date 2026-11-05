import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { v4 as uuidv4 } from "uuid";
import type { ProfileInfo } from "#/api/profiles-service/profiles-service.api";

export interface ModelSwitchDetails {
  id?: string;
  initiator?: "user" | "agent" | "router";
  reason?: string | null;
}

export interface ModelListEntry extends ModelSwitchDetails {
  id: string;
  /**
   * Id of the chat event after which this entry should render, or `null` to
   * pin it to the top of the chat history (no rendered events at the time
   * of /model).
   */
  anchorEventId: string | null;
  profiles: ProfileInfo[];
  switchedTo?: string;
}

/**
 * A historical switch to seed via `seedSwitches`. `id` must be stable across
 * reloads (derived from the source observation event id) so re-seeding the
 * same loaded history is idempotent.
 */
export interface SeededSwitch extends ModelSwitchDetails {
  id: string;
  anchorEventId: string | null;
  profileName: string;
}

interface ModelState {
  entriesByConversation: Record<string, ModelListEntry[]>;
  /**
   * Most-recently-switched profile name per conversation. Updated by
   * `recordSwitch` so the UI (button label, popover check mark) reflects the
   * new selection instantly, before the conversation refetch from the agent
   * server lands.
   */
  activeProfileByConversation: Record<string, string>;
  /**
   * A cloud member's home-page pick, sent with the next home launch instead of
   * activating the profile org-wide (which needs org-admin permission).
   */
  homeLlmProfile: string | null;
  /**
   * A model pick (LLM profile or ACP model) was made on the home page before
   * the first message. Suppresses "Run on first message" routing for that one
   * launch so the explicit pick isn't overridden on turn 1. Set at the UI pick
   * handlers (the single chokepoint covering every pick path) and read once by
   * `useCreateConversation`, which clears it as it consumes it — so it never
   * leaks to child/`/new`/other launches.
   */
  homeManualModelPick: boolean;
}

interface ModelActions {
  show: (
    conversationId: string,
    anchorEventId: string | null,
    profiles: ProfileInfo[],
  ) => void;
  recordSwitch: (
    conversationId: string,
    anchorEventId: string | null,
    profileName: string,
    details?: ModelSwitchDetails,
  ) => void;
  /**
   * Seeds "Switched to" entries reconstructed from loaded history. Skips any
   * whose `id` is already present, so it can run on every history (re)load
   * without duplicating, and it preserves live-recorded entries.
   */
  seedSwitches: (conversationId: string, switches: SeededSwitch[]) => void;
  /**
   * Sets the optimistic active-profile entry for a conversation without
   * appending a chat entry — the reload/history path re-derives the stamp
   * from past SwitchLLM observations, which seed their chat entries through
   * `seedSwitches` instead of `recordSwitch`.
   */
  setActiveProfile: (conversationId: string, profileName: string) => void;
  /** Drops only the optimistic active-profile entry for a conversation. */
  clearActiveProfile: (conversationId: string) => void;
  setHomeLlmProfile: (profileName: string | null) => void;
  setHomeManualModelPick: (value: boolean) => void;
  clear: (conversationId: string) => void;
  clearAll: () => void;
}

type ModelStore = ModelState & ModelActions;

const appendEntry = (
  state: ModelState,
  conversationId: string,
  entry: ModelListEntry,
): Pick<ModelState, "entriesByConversation"> => ({
  entriesByConversation: {
    ...state.entriesByConversation,
    [conversationId]: [
      ...(state.entriesByConversation[conversationId] ?? []),
      entry,
    ],
  },
});

export const useModelStore = create<ModelStore>()(
  devtools(
    (set) => ({
      entriesByConversation: {},
      activeProfileByConversation: {},
      homeLlmProfile: null,
      homeManualModelPick: false,
      show: (conversationId, anchorEventId, profiles) =>
        set((s) =>
          appendEntry(s, conversationId, {
            id: uuidv4(),
            anchorEventId,
            profiles,
          }),
        ),
      recordSwitch: (
        conversationId,
        anchorEventId,
        profileName,
        details = {},
      ) =>
        set((s) => ({
          ...(details.id &&
          s.entriesByConversation[conversationId]?.some(
            (entry) => entry.id === details.id,
          )
            ? {}
            : appendEntry(s, conversationId, {
                ...details,
                id: details.id ?? uuidv4(),
                initiator: details.initiator ?? "user",
                anchorEventId,
                profiles: [],
                switchedTo: profileName,
              })),
          activeProfileByConversation: {
            ...s.activeProfileByConversation,
            [conversationId]: profileName,
          },
        })),
      seedSwitches: (conversationId, switches) =>
        set((s) => {
          const existing = s.entriesByConversation[conversationId] ?? [];
          const existingIds = new Set(existing.map((e) => e.id));
          const additions = switches
            .filter((sw) => !existingIds.has(sw.id))
            .map(
              (sw): ModelListEntry => ({
                id: sw.id,
                anchorEventId: sw.anchorEventId,
                profiles: [],
                switchedTo: sw.profileName,
                initiator: sw.initiator,
                reason: sw.reason,
              }),
            );
          if (additions.length === 0) return s;
          return {
            entriesByConversation: {
              ...s.entriesByConversation,
              [conversationId]: [...existing, ...additions],
            },
          };
        }),
      setActiveProfile: (conversationId, profileName) =>
        set((s) => ({
          activeProfileByConversation: {
            ...s.activeProfileByConversation,
            [conversationId]: profileName,
          },
        })),
      clearActiveProfile: (conversationId) =>
        set((s) => {
          if (!(conversationId in s.activeProfileByConversation)) return s;
          const activeProfileByConversation = {
            ...s.activeProfileByConversation,
          };
          delete activeProfileByConversation[conversationId];
          return { activeProfileByConversation };
        }),
      setHomeLlmProfile: (profileName) => set({ homeLlmProfile: profileName }),
      setHomeManualModelPick: (value) => set({ homeManualModelPick: value }),
      clear: (conversationId) =>
        set((s) => {
          const entriesByConversation = { ...s.entriesByConversation };
          delete entriesByConversation[conversationId];
          const activeProfileByConversation = {
            ...s.activeProfileByConversation,
          };
          delete activeProfileByConversation[conversationId];
          return { entriesByConversation, activeProfileByConversation };
        }),
      clearAll: () =>
        set({
          entriesByConversation: {},
          activeProfileByConversation: {},
          homeLlmProfile: null,
          homeManualModelPick: false,
        }),
    }),
    { name: "ModelStore" },
  ),
);
