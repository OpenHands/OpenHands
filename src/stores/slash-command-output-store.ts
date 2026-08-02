import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { v4 as uuidv4 } from "uuid";
import type { LoadedResources, SlashCommandItem } from "#/types/slash-command";

export const HOME_SLASH_COMMAND_SCOPE_ID = "__home__";

interface SkillsCommandOutputBase {
  id: string;
  kind: "skills";
  invocationOrder: number;
  timelineBoundaryEventId: string | null;
  /**
   * Keep a command submitted in the mounted conversation visible while its
   * captured boundary is temporarily outside the loaded event window. The
   * interceptor turns this off when that conversation view unmounts.
   */
  showWhenPlacementUnresolved: boolean;
}

export type SkillsCommandOutput =
  | (SkillsCommandOutputBase & { status: "loading" })
  | (SkillsCommandOutputBase & {
      status: "ready";
      resources: LoadedResources;
    })
  | (SkillsCommandOutputBase & {
      status: "error";
      errorKind: "timeout" | "request";
    });

export interface HelpCommandOutput {
  id: string;
  kind: "help";
  invocationOrder: number;
  timelineBoundaryEventId: string | null;
  commands: SlashCommandItem[];
}

export type SlashCommandOutput = SkillsCommandOutput | HelpCommandOutput;

interface SlashCommandOutputState {
  entriesByScope: Record<string, SlashCommandOutput[]>;
  nextInvocationOrder: number;
  reserveInvocationOrder: () => number;
  beginSkills: (
    scopeId: string,
    timelineBoundaryEventId: string | null,
    invocationOrder?: number,
  ) => string;
  completeSkills: (
    scopeId: string,
    entryId: string,
    resources: LoadedResources,
  ) => void;
  failSkills: (
    scopeId: string,
    entryId: string,
    errorKind: "timeout" | "request",
  ) => void;
  deactivateSkillsPlacementFallback: (scopeId: string) => void;
  showSkills: (
    scopeId: string,
    timelineBoundaryEventId: string | null,
    resources: LoadedResources,
    invocationOrder?: number,
  ) => void;
  showHelp: (
    scopeId: string,
    timelineBoundaryEventId: string | null,
    commands: SlashCommandItem[],
    invocationOrder?: number,
  ) => void;
  clear: (scopeId: string) => void;
  clearAll: () => void;
}

export const useSlashCommandOutputStore = create<SlashCommandOutputState>()(
  devtools(
    (set) => ({
      entriesByScope: {},
      nextInvocationOrder: 0,
      reserveInvocationOrder: () => {
        let reserved = 0;
        set((state) => {
          reserved = state.nextInvocationOrder;
          return { nextInvocationOrder: reserved + 1 };
        });
        return reserved;
      },
      beginSkills: (scopeId, timelineBoundaryEventId, invocationOrder) => {
        const id = uuidv4();
        set((state) => {
          const order = invocationOrder ?? state.nextInvocationOrder;
          const entry: SkillsCommandOutput = {
            id,
            kind: "skills",
            status: "loading",
            invocationOrder: order,
            timelineBoundaryEventId,
            showWhenPlacementUnresolved: true,
          };
          const entries = [
            ...(state.entriesByScope[scopeId] ?? []),
            entry,
          ].sort((left, right) => left.invocationOrder - right.invocationOrder);
          return {
            entriesByScope: {
              ...state.entriesByScope,
              [scopeId]: entries,
            },
            nextInvocationOrder:
              invocationOrder === undefined
                ? state.nextInvocationOrder + 1
                : state.nextInvocationOrder,
          };
        });
        return id;
      },
      completeSkills: (scopeId, entryId, resources) =>
        set((state) => {
          const existing = state.entriesByScope[scopeId];
          if (!existing) return state;

          let changed = false;
          const entries = existing.map((entry): SlashCommandOutput => {
            if (
              entry.id !== entryId ||
              entry.kind !== "skills" ||
              entry.status !== "loading"
            ) {
              return entry;
            }
            changed = true;
            return { ...entry, status: "ready", resources };
          });
          if (!changed) return state;
          return {
            entriesByScope: {
              ...state.entriesByScope,
              [scopeId]: entries,
            },
          };
        }),
      failSkills: (scopeId, entryId, errorKind) =>
        set((state) => {
          const existing = state.entriesByScope[scopeId];
          if (!existing) return state;

          let changed = false;
          const entries = existing.map((entry): SlashCommandOutput => {
            if (
              entry.id !== entryId ||
              entry.kind !== "skills" ||
              entry.status !== "loading"
            ) {
              return entry;
            }
            changed = true;
            return { ...entry, status: "error", errorKind };
          });
          if (!changed) return state;
          return {
            entriesByScope: {
              ...state.entriesByScope,
              [scopeId]: entries,
            },
          };
        }),
      deactivateSkillsPlacementFallback: (scopeId) =>
        set((state) => {
          const existing = state.entriesByScope[scopeId];
          if (!existing) return state;

          let changed = false;
          const entries = existing.map((entry): SlashCommandOutput => {
            if (entry.kind !== "skills" || !entry.showWhenPlacementUnresolved) {
              return entry;
            }
            changed = true;
            return { ...entry, showWhenPlacementUnresolved: false };
          });
          if (!changed) return state;
          return {
            entriesByScope: {
              ...state.entriesByScope,
              [scopeId]: entries,
            },
          };
        }),
      showSkills: (
        scopeId,
        timelineBoundaryEventId,
        resources,
        invocationOrder,
      ) =>
        set((state) => {
          const order = invocationOrder ?? state.nextInvocationOrder;
          const entry = {
            id: uuidv4(),
            kind: "skills" as const,
            status: "ready" as const,
            invocationOrder: order,
            timelineBoundaryEventId,
            showWhenPlacementUnresolved: false,
            resources,
          };
          const entries = [
            ...(state.entriesByScope[scopeId] ?? []),
            entry,
          ].sort((left, right) => left.invocationOrder - right.invocationOrder);
          return {
            entriesByScope: {
              ...state.entriesByScope,
              [scopeId]: entries,
            },
            nextInvocationOrder:
              invocationOrder === undefined
                ? state.nextInvocationOrder + 1
                : state.nextInvocationOrder,
          };
        }),
      showHelp: (scopeId, timelineBoundaryEventId, commands, invocationOrder) =>
        set((state) => {
          const order = invocationOrder ?? state.nextInvocationOrder;
          const entry = {
            id: uuidv4(),
            kind: "help" as const,
            timelineBoundaryEventId,
            commands,
            invocationOrder: order,
          };
          const entries = [
            ...(state.entriesByScope[scopeId] ?? []),
            entry,
          ].sort((left, right) => left.invocationOrder - right.invocationOrder);
          return {
            entriesByScope: {
              ...state.entriesByScope,
              [scopeId]: entries,
            },
            nextInvocationOrder:
              invocationOrder === undefined
                ? state.nextInvocationOrder + 1
                : state.nextInvocationOrder,
          };
        }),
      clear: (scopeId) =>
        set((state) => {
          const entriesByScope = { ...state.entriesByScope };
          delete entriesByScope[scopeId];
          return { entriesByScope };
        }),
      clearAll: () => set({ entriesByScope: {} }),
    }),
    { name: "SlashCommandOutputStore" },
  ),
);
