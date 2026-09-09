import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type { KanbanCard } from "#/api/kanban-service/kanban-types";
import {
  KANBAN_SYNC_STORAGE_KEY,
  deriveCodeIssues,
  fingerprintKey,
  lanesFromRemoteIssues,
  openReconcileItems,
  reconcileIssues,
  remoteFingerprint,
  type IssueFingerprint,
  type KanbanLane,
  type KanbanMappedSource,
  type KanbanMasterSide,
  type KanbanSourceProvider,
  type MergeRecommendation,
  type RemoteIssue,
} from "#/utils/kanban-sync";

export const KANBAN_SYNC_INITIAL_STATE: KanbanSyncState = {
  sources: [],
  remoteIssues: [],
  lanes: [],
  fingerprints: {},
  conflicts: [],
  collapsedLaneIds: [],
  reconcileOpen: false,
  mappingOpen: false,
};

export interface KanbanSyncState {
  sources: KanbanMappedSource[];
  remoteIssues: RemoteIssue[];
  lanes: KanbanLane[];
  fingerprints: Record<string, IssueFingerprint>;
  conflicts: MergeRecommendation[];
  collapsedLaneIds: string[];
  reconcileOpen: boolean;
  mappingOpen: boolean;
}

interface KanbanSyncActions {
  mapSource: (input: {
    boardId: string;
    provider: KanbanSourceProvider;
    label: string;
    url?: string | null;
    master: KanbanMasterSide;
    issues?: RemoteIssue[];
    localCards?: KanbanCard[];
  }) => KanbanMappedSource;
  ingestIssues: (
    sourceId: string,
    issues: RemoteIssue[],
    localCards: KanbanCard[],
  ) => void;
  reconcileBoard: (
    boardId: string,
    localCards: KanbanCard[],
  ) => MergeRecommendation[];
  resolveConflict: (id: string, fingerprint?: IssueFingerprint) => void;
  setMaster: (sourceId: string, master: KanbanMasterSide) => void;
  addLane: (input: {
    boardId: string;
    name: string;
    kind?: KanbanLane["kind"];
    parentId?: string | null;
  }) => KanbanLane;
  removeLane: (laneId: string) => void;
  toggleLaneCollapsed: (laneId: string) => void;
  setReconcileOpen: (open: boolean) => void;
  setMappingOpen: (open: boolean) => void;
  detectAfterLocalChange: (boardId: string, localCards: KanbanCard[]) => void;
  reset: () => void;
}

type KanbanSyncStore = KanbanSyncState & KanbanSyncActions;

function runReconcile(
  state: KanbanSyncState,
  boardId: string,
  localCards: KanbanCard[],
): MergeRecommendation[] {
  const sources = state.sources.filter((source) => source.boardId === boardId);
  const remotes = state.remoteIssues.filter((issue) =>
    sources.some((source) => source.id === issue.sourceId),
  );
  const master = sources[0]?.master ?? "local";
  return reconcileIssues({
    localCards,
    remotes,
    master,
    fingerprints: state.fingerprints,
    boardId,
  });
}

export const useKanbanSyncStore = create<KanbanSyncStore>()(
  persist(
    (set, get) => ({
      ...KANBAN_SYNC_INITIAL_STATE,

      mapSource: ({
        boardId,
        provider,
        label,
        url = null,
        master,
        issues = [],
        localCards = [],
      }) => {
        const source: KanbanMappedSource = {
          id: crypto.randomUUID(),
          boardId,
          provider,
          label,
          url,
          master,
          createdAt: new Date().toISOString(),
        };
        const derived =
          provider === "code"
            ? deriveCodeIssues(localCards, source.id)
            : issues.map((issue) => ({ ...issue, sourceId: source.id }));
        set((state) => {
          const lanes = lanesFromRemoteIssues(
            boardId,
            source.id,
            derived,
            state.lanes,
          );
          const next: KanbanSyncState = {
            ...state,
            sources: [...state.sources, source],
            remoteIssues: [
              ...state.remoteIssues.filter(
                (issue) => issue.sourceId !== source.id,
              ),
              ...derived,
            ],
            lanes,
            mappingOpen: false,
            reconcileOpen: true,
          };
          const recs = runReconcile(next, boardId, localCards);
          return {
            ...next,
            conflicts: openReconcileItems(recs),
          };
        });
        return source;
      },

      ingestIssues: (sourceId, issues, localCards) => {
        const source = get().sources.find((item) => item.id === sourceId);
        if (!source) return;
        const tagged = issues.map((issue) => ({ ...issue, sourceId }));
        set((state) => {
          const lanes = lanesFromRemoteIssues(
            source.boardId,
            sourceId,
            tagged,
            state.lanes,
          );
          const next: KanbanSyncState = {
            ...state,
            remoteIssues: [
              ...state.remoteIssues.filter(
                (issue) => issue.sourceId !== sourceId,
              ),
              ...tagged,
            ],
            lanes,
            reconcileOpen: true,
          };
          return {
            ...next,
            conflicts: openReconcileItems(
              runReconcile(next, source.boardId, localCards),
            ),
          };
        });
      },

      reconcileBoard: (boardId, localCards) => {
        const recs = runReconcile(get(), boardId, localCards);
        const open = openReconcileItems(recs);
        set({ conflicts: open, reconcileOpen: open.length > 0 });
        return recs;
      },

      resolveConflict: (id, fingerprint) => {
        set((state) => {
          const conflict = state.conflicts.find((item) => item.id === id);
          const fingerprints = { ...state.fingerprints };
          if (fingerprint && conflict?.remote) {
            fingerprints[
              fingerprintKey(
                conflict.remote.sourceId,
                conflict.remote.externalId,
              )
            ] = fingerprint;
          } else if (conflict?.remote) {
            fingerprints[
              fingerprintKey(
                conflict.remote.sourceId,
                conflict.remote.externalId,
              )
            ] = remoteFingerprint(conflict.remote);
          }
          return {
            conflicts: state.conflicts.filter((item) => item.id !== id),
            fingerprints,
          };
        });
      },

      setMaster: (sourceId, master) =>
        set((state) => ({
          sources: state.sources.map((source) =>
            source.id === sourceId ? { ...source, master } : source,
          ),
        })),

      addLane: ({ boardId, name, kind = "manual", parentId = null }) => {
        const lane: KanbanLane = {
          id: crypto.randomUUID(),
          boardId,
          parentId,
          name,
          kind,
          sourceId: null,
          remoteKey: null,
          position: get().lanes.filter((item) => item.boardId === boardId)
            .length,
        };
        set((state) => ({ lanes: [...state.lanes, lane] }));
        return lane;
      },

      removeLane: (laneId) =>
        set((state) => ({
          lanes: state.lanes.filter(
            (lane) => lane.id !== laneId && lane.parentId !== laneId,
          ),
        })),

      toggleLaneCollapsed: (laneId) =>
        set((state) => ({
          collapsedLaneIds: state.collapsedLaneIds.includes(laneId)
            ? state.collapsedLaneIds.filter((id) => id !== laneId)
            : [...state.collapsedLaneIds, laneId],
        })),

      setReconcileOpen: (open) => set({ reconcileOpen: open }),
      setMappingOpen: (open) => set({ mappingOpen: open }),

      detectAfterLocalChange: (boardId, localCards) => {
        const recs = openReconcileItems(
          runReconcile(get(), boardId, localCards),
        );
        if (recs.length === 0) return;
        set({ conflicts: recs, reconcileOpen: true });
      },

      reset: () => set(KANBAN_SYNC_INITIAL_STATE),
    }),
    {
      name: KANBAN_SYNC_STORAGE_KEY,
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        sources: state.sources,
        remoteIssues: state.remoteIssues,
        lanes: state.lanes,
        fingerprints: state.fingerprints,
        collapsedLaneIds: state.collapsedLaneIds,
      }),
    },
  ),
);
