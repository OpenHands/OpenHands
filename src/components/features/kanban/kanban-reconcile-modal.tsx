import { useTranslation } from "react-i18next";
import type {
  CreateCardPayload,
  KanbanBoard,
  KanbanCard,
  UpdateCardPayload,
} from "#/api/kanban-service/kanban-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { I18nKey } from "#/i18n/declaration";
import { useKanbanSyncStore } from "#/stores/kanban-sync-store";
import {
  cardFingerprint,
  columnIdForRemoteStatus,
  patchFromRemote,
  remoteFingerprint,
  type MergeRecommendation,
} from "#/utils/kanban-sync";
import { cn } from "#/utils/utils";

const KIND_KEYS: Record<MergeRecommendation["kind"], I18nKey> = {
  create_local: I18nKey.KANBAN$CREATE_LOCAL,
  create_remote: I18nKey.KANBAN$CREATE_REMOTE,
  extend_local: I18nKey.KANBAN$EXTEND_LOCAL,
  extend_remote: I18nKey.KANBAN$EXTEND_REMOTE,
  conflict: I18nKey.KANBAN$RECONCILE,
  remote_remote: I18nKey.KANBAN$REMOTE_REMOTE,
  matched: I18nKey.KANBAN$RECONCILE_EMPTY,
};

export interface KanbanReconcileModalProps {
  boards: KanbanBoard[];
  localCards: KanbanCard[];
  onUpdateCard: (cardId: string, payload: UpdateCardPayload) => void;
  onCreateCard: (columnId: string, payload: CreateCardPayload) => void;
}

export function KanbanReconcileModal({
  boards,
  localCards,
  onUpdateCard,
  onCreateCard,
}: KanbanReconcileModalProps) {
  const { t } = useTranslation("openhands");
  const reconcileOpen = useKanbanSyncStore((state) => state.reconcileOpen);
  const conflicts = useKanbanSyncStore((state) => state.conflicts);
  const setReconcileOpen = useKanbanSyncStore(
    (state) => state.setReconcileOpen,
  );
  const resolveConflict = useKanbanSyncStore((state) => state.resolveConflict);
  const setMaster = useKanbanSyncStore((state) => state.setMaster);
  const sources = useKanbanSyncStore((state) => state.sources);
  const lanes = useKanbanSyncStore((state) => state.lanes);

  if (!reconcileOpen) return null;

  const cardById = new Map(localCards.map((card) => [card.id, card]));
  const boardById = new Map(boards.map((board) => [board.id, board]));

  const applyRemote = (item: MergeRecommendation) => {
    if (!item.remote) {
      resolveConflict(item.id);
      return;
    }
    const board =
      boardById.get(item.boardId ?? "") ??
      boards.find((entry) => entry.id === item.remote?.sourceId) ??
      boards[0];
    if (item.kind === "create_local" || !item.localCardId) {
      const columnId = columnIdForRemoteStatus(
        board?.columns ?? [],
        item.remote.status,
      );
      if (columnId) {
        const lane = lanes.find(
          (itemLane) =>
            itemLane.boardId === (board?.id ?? item.boardId) &&
            itemLane.sourceId === item.remote?.sourceId &&
            itemLane.remoteKey === item.remote?.groupKey,
        );
        onCreateCard(columnId, {
          title: item.remote.title,
          description: item.remote.description,
          status: item.remote.status,
          assignee: item.remote.assignee,
          origin: "remote",
          external_id: item.remote.externalId,
          source_id: item.remote.sourceId,
          lane_id: lane?.id ?? null,
        });
      }
      resolveConflict(item.id, remoteFingerprint(item.remote));
      return;
    }
    onUpdateCard(item.localCardId, patchFromRemote(item.remote));
    resolveConflict(item.id, remoteFingerprint(item.remote));
  };

  const keepLocal = (item: MergeRecommendation) => {
    const local = item.localCardId ? cardById.get(item.localCardId) : null;
    resolveConflict(item.id, local ? cardFingerprint(local) : undefined);
  };

  const applyRecommended = (item: MergeRecommendation) => {
    if (item.recommended === "remote") applyRemote(item);
    else keepLocal(item);
  };

  return (
    <ModalBackdrop onClose={() => setReconcileOpen(false)}>
      <div
        data-testid="kanban-reconcile-modal"
        className="flex max-h-[min(36rem,calc(100vh-2rem))] w-[min(40rem,calc(100vw-2rem))] flex-col gap-4 overflow-hidden rounded-xl border border-[var(--oh-border)] bg-base-secondary p-4"
      >
        <div className="flex items-start justify-between gap-3">
          <h2 className="text-sm font-medium text-[var(--oh-foreground)]">
            {t(I18nKey.KANBAN$RECONCILE_TITLE)}
          </h2>
          <BrandButton
            type="button"
            variant="tertiary"
            testId="kanban-reconcile-close"
            onClick={() => setReconcileOpen(false)}
          >
            {t(I18nKey.BUTTON$CLOSE)}
          </BrandButton>
        </div>
        {sources[0] ? (
          <div className="flex items-center gap-2 text-xs text-tertiary-light">
            <span>{t(I18nKey.KANBAN$MASTER_COPY)}</span>
            <BrandButton
              type="button"
              variant={sources[0].master === "local" ? "primary" : "secondary"}
              testId="kanban-master-local"
              onClick={() => setMaster(sources[0]!.id, "local")}
            >
              {t(I18nKey.KANBAN$MASTER_LOCAL)}
            </BrandButton>
            <BrandButton
              type="button"
              variant={sources[0].master === "remote" ? "primary" : "secondary"}
              testId="kanban-master-remote"
              onClick={() => setMaster(sources[0]!.id, "remote")}
            >
              {t(I18nKey.KANBAN$MASTER_REMOTE)}
            </BrandButton>
          </div>
        ) : null}
        <div className="min-h-0 flex-1 space-y-3 overflow-y-auto">
          {conflicts.length === 0 ? (
            <p
              data-testid="kanban-reconcile-empty"
              className="text-sm text-tertiary-light"
            >
              {t(I18nKey.KANBAN$RECONCILE_EMPTY)}
            </p>
          ) : (
            conflicts.map((item) => {
              const local = item.localCardId
                ? cardById.get(item.localCardId)
                : null;
              return (
                <article
                  key={item.id}
                  data-testid={`kanban-reconcile-item-${item.id}`}
                  className="rounded-lg border border-[var(--oh-border-subtle)] p-3"
                >
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <p className="text-[13px] font-medium text-[var(--oh-foreground)]">
                      {item.remote?.title ?? local?.title ?? ""}
                    </p>
                    <span className="text-[11px] text-[var(--oh-muted)]">
                      {t(KIND_KEYS[item.kind])}
                    </span>
                  </div>
                  {item.fieldDiffs.length > 0 ? (
                    <div className="mb-3 grid grid-cols-3 gap-2 text-[11px] leading-4">
                      <span className="text-tertiary-light">
                        {t(I18nKey.KANBAN$FIELD_LOCAL)}
                      </span>
                      <span className="text-tertiary-light">
                        {t(I18nKey.KANBAN$FIELD_RECOMMENDED)}
                      </span>
                      <span className="text-tertiary-light">
                        {t(I18nKey.KANBAN$FIELD_REMOTE)}
                      </span>
                      {item.fieldDiffs.map((diff) => (
                        <span key={diff.field} className="contents">
                          <span className="truncate text-[var(--oh-muted)]">
                            {diff.local}
                          </span>
                          <span
                            className={cn(
                              "truncate",
                              item.recommended === "remote"
                                ? "text-[var(--oh-foreground)]"
                                : "text-[var(--oh-muted)]",
                            )}
                          >
                            {item.recommended === "remote"
                              ? diff.remote
                              : diff.local}
                          </span>
                          <span className="truncate text-[var(--oh-muted)]">
                            {diff.remote}
                          </span>
                        </span>
                      ))}
                    </div>
                  ) : null}
                  <div className="flex flex-wrap gap-2">
                    <BrandButton
                      type="button"
                      variant="secondary"
                      testId={`kanban-keep-local-${item.id}`}
                      onClick={() => keepLocal(item)}
                    >
                      {t(I18nKey.KANBAN$KEEP_LOCAL)}
                    </BrandButton>
                    <BrandButton
                      type="button"
                      variant="secondary"
                      testId={`kanban-take-remote-${item.id}`}
                      onClick={() => applyRemote(item)}
                    >
                      {t(I18nKey.KANBAN$TAKE_REMOTE)}
                    </BrandButton>
                    <BrandButton
                      type="button"
                      variant="primary"
                      testId={`kanban-apply-merge-${item.id}`}
                      onClick={() => applyRecommended(item)}
                    >
                      {t(I18nKey.KANBAN$APPLY_MERGE)}
                    </BrandButton>
                  </div>
                </article>
              );
            })
          )}
        </div>
      </div>
    </ModalBackdrop>
  );
}
