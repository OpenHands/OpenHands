import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { AppConversationStartTaskStatus } from "#/api/conversation-service/agent-server-conversation-service.types";

const STORAGE_KEY = "openhands-session-activity";
const MAX_CONVERSATIONS = 100;
export interface SessionActivity {
  id: string;
  kind: "initialize" | "resume";
  startedAt: number;
  phase: "pending" | "complete" | "failed";
  steps: AppConversationStartTaskStatus[];
  open: boolean;
  detailsOpen?: boolean;
  afterEventId?: string;
}

export const sessionActivityKey = (
  backendId: string,
  orgId: string | null | undefined,
  conversationId: string,
) => JSON.stringify([backendId, orgId ?? null, conversationId]);

function readEntries(value: unknown): Record<string, SessionActivity[]> {
  if (
    !value ||
    typeof value !== "object" ||
    !("entries" in value) ||
    !value.entries ||
    typeof value.entries !== "object"
  )
    return {};
  return Object.fromEntries(
    Object.entries(value.entries).filter(
      ([, entries]) =>
        Array.isArray(entries) &&
        entries.every(
          (entry) =>
            entry &&
            typeof entry.id === "string" &&
            ["initialize", "resume"].includes(entry.kind) &&
            Number.isFinite(entry.startedAt) &&
            ["pending", "complete", "failed"].includes(entry.phase) &&
            typeof entry.open === "boolean" &&
            Array.isArray(entry.steps) &&
            entry.steps.every((step: unknown) => typeof step === "string"),
        ),
    ),
  );
}

interface SessionActivityState {
  entries: Record<string, SessionActivity[]>;
  begin: (
    key: string,
    kind: SessionActivity["kind"],
    afterEventId?: string,
  ) => void;
  observe: (key: string, status: AppConversationStartTaskStatus) => void;
  transfer: (from: string, to: string) => void;
  setOpen: (key: string, id: string, open: boolean) => void;
  setDetailsOpen: (key: string, id: string, open: boolean) => void;
  remove: (key: string) => void;
}

// Presentation history for this tab only; never sent as agent events.
export const useSessionActivityStore = create<SessionActivityState>()(
  persist(
    (set) => ({
      entries: {},
      begin: (key, kind, afterEventId) =>
        set((state) => {
          const previous = state.entries[key] ?? [];
          if (
            previous.at(-1)?.phase === "pending" ||
            (kind === "initialize" &&
              previous.some((entry) => entry.kind === kind))
          )
            return state;
          const entries = {
            ...state.entries,
            [key]: [
              ...previous,
              {
                id: crypto.randomUUID(),
                kind,
                afterEventId,
                startedAt: Date.now(),
                phase: "pending" as const,
                steps: [],
                open: false,
              },
            ],
          };
          return {
            entries: Object.fromEntries(
              Object.entries(entries).slice(-MAX_CONVERSATIONS),
            ),
          };
        }),
      observe: (key, status) =>
        set((state) => {
          const entries = state.entries[key];
          const last = entries?.at(-1);
          if (!last || last.steps.at(-1) === status) return state;
          const next: SessionActivity = {
            ...last,
            steps: last.steps.includes(status)
              ? last.steps
              : [...last.steps, status],
            phase:
              status === "READY"
                ? "complete"
                : status === "ERROR"
                  ? "failed"
                  : "pending",
          };
          return {
            entries: {
              ...state.entries,
              [key]: [...entries.slice(0, -1), next],
            },
          };
        }),
      transfer: (from, to) =>
        set((state) => {
          if (from === to || !state.entries[from]) return state;
          return { entries: { ...state.entries, [to]: state.entries[from] } };
        }),
      setOpen: (key, id, open) =>
        set((state) => ({
          entries: {
            ...state.entries,
            [key]: (state.entries[key] ?? []).map((entry) =>
              entry.id === id ? { ...entry, open } : entry,
            ),
          },
        })),
      setDetailsOpen: (key, id, detailsOpen) =>
        set((state) => ({
          entries: {
            ...state.entries,
            [key]: (state.entries[key] ?? []).map((entry) =>
              entry.id === id ? { ...entry, detailsOpen } : entry,
            ),
          },
        })),
      remove: (key) =>
        set((state) => {
          const { [key]: removed, ...entries } = state.entries;
          return removed ? { entries } : state;
        }),
    }),
    {
      name: STORAGE_KEY,
      storage: createJSONStorage(() => ({
        getItem: (key) => {
          try {
            return sessionStorage.getItem(key);
          } catch {
            return null;
          }
        },
        setItem: (key, value) => {
          try {
            sessionStorage.setItem(key, value);
          } catch {
            /* Keep this tab's in-memory history when storage is unavailable. */
          }
        },
        removeItem: (key) => {
          try {
            sessionStorage.removeItem(key);
          } catch {
            /* Storage may be disabled by the host. */
          }
        },
      })),
      partialize: (state) => ({ entries: state.entries }),
      merge: (persisted, current) => ({
        ...current,
        entries: readEntries(persisted),
      }),
    },
  ),
);
