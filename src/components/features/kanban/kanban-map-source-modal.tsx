import React from "react";
import { useTranslation } from "react-i18next";
import type {
  KanbanBoard,
  KanbanCard,
} from "#/api/kanban-service/kanban-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SegmentedToggle } from "#/components/features/files-tab/segmented-toggle";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { I18nKey } from "#/i18n/declaration";
import { useKanbanSyncStore } from "#/stores/kanban-sync-store";
import {
  formControlFieldClassName,
  formControlMultilineFieldClassName,
} from "#/utils/form-control-classes";
import {
  KANBAN_SOURCE_PROVIDERS,
  parseRemoteIssuesJson,
  type KanbanMasterSide,
  type KanbanSourceProvider,
} from "#/utils/kanban-sync";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { cn } from "#/utils/utils";

const PROVIDER_KEYS: Record<KanbanSourceProvider, I18nKey> = {
  jira: I18nKey.KANBAN$SOURCE_JIRA,
  linear: I18nKey.KANBAN$SOURCE_LINEAR,
  plane: I18nKey.KANBAN$SOURCE_PLANE,
  github: I18nKey.KANBAN$SOURCE_GITHUB,
  code: I18nKey.KANBAN$SOURCE_CODE,
};

export interface KanbanMapSourceModalProps {
  boards: { board: KanbanBoard; name: string }[];
  localCards: KanbanCard[];
  defaultBoardId: string | null;
}

export function KanbanMapSourceModal({
  boards,
  localCards,
  defaultBoardId,
}: KanbanMapSourceModalProps) {
  const { t } = useTranslation("openhands");
  const mappingOpen = useKanbanSyncStore((state) => state.mappingOpen);
  const setMappingOpen = useKanbanSyncStore((state) => state.setMappingOpen);
  const mapSource = useKanbanSyncStore((state) => state.mapSource);
  const [provider, setProvider] =
    React.useState<KanbanSourceProvider>("linear");
  const [master, setMaster] = React.useState<KanbanMasterSide>("remote");
  const [label, setLabel] = React.useState("");
  const [url, setUrl] = React.useState("");
  const [snapshot, setSnapshot] = React.useState("");
  const [boardId, setBoardId] = React.useState(defaultBoardId ?? "");

  React.useEffect(() => {
    if (mappingOpen) {
      setBoardId(defaultBoardId ?? boards[0]?.board.id ?? "");
      setLabel("");
      setUrl("");
      setSnapshot("");
    }
  }, [mappingOpen, defaultBoardId, boards]);

  if (!mappingOpen) return null;

  const submit = () => {
    const selectedBoardId = boardId || boards[0]?.board.id;
    if (!selectedBoardId) return;
    let issues = [];
    try {
      issues = parseRemoteIssuesJson(snapshot, "pending");
    } catch {
      displayErrorToast(t(I18nKey.ERROR$GENERIC));
      return;
    }
    mapSource({
      boardId: selectedBoardId,
      provider,
      label: label.trim() || t(PROVIDER_KEYS[provider]),
      url: url.trim() || null,
      master,
      issues,
      localCards: localCards.filter(
        (card) => card.board_id === selectedBoardId,
      ),
    });
  };

  return (
    <ModalBackdrop onClose={() => setMappingOpen(false)}>
      <div
        data-testid="kanban-map-source-modal"
        className="flex w-[min(32rem,calc(100vw-2rem))] flex-col gap-4 rounded-xl border border-[var(--oh-border)] bg-base-secondary p-4"
      >
        <h2 className="text-sm font-medium text-[var(--oh-foreground)]">
          {t(I18nKey.KANBAN$MAP_SOURCE_TITLE)}
        </h2>
        {boards.length > 1 ? (
          <label className="block text-xs text-tertiary-light">
            {t(I18nKey.KANBAN$SELECT_BOARD)}
            <select
              data-testid="kanban-map-source-board"
              value={boardId}
              onChange={(event) => setBoardId(event.target.value)}
              className={cn(formControlFieldClassName, "mt-1")}
            >
              {boards.map((item) => (
                <option key={item.board.id} value={item.board.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <div>
          <p className="mb-1.5 text-xs text-tertiary-light">
            {t(I18nKey.KANBAN$SOURCE_PROVIDER)}
          </p>
          <SegmentedToggle
            value={provider}
            onChange={setProvider}
            ariaLabel={t(I18nKey.KANBAN$SOURCE_PROVIDER)}
            testId="kanban-source-provider"
            options={KANBAN_SOURCE_PROVIDERS.map((item) => ({
              value: item,
              label: t(PROVIDER_KEYS[item]),
            }))}
          />
        </div>
        <label className="block text-xs text-tertiary-light">
          {t(I18nKey.KANBAN$SOURCE_LABEL)}
          <input
            data-testid="kanban-source-label"
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            className={cn(formControlFieldClassName, "mt-1")}
          />
        </label>
        <label className="block text-xs text-tertiary-light">
          {t(I18nKey.KANBAN$SOURCE_URL)}
          <input
            data-testid="kanban-source-url"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            className={cn(formControlFieldClassName, "mt-1")}
          />
        </label>
        <div>
          <p className="mb-1.5 text-xs text-tertiary-light">
            {t(I18nKey.KANBAN$MASTER_COPY)}
          </p>
          <SegmentedToggle
            value={master}
            onChange={setMaster}
            ariaLabel={t(I18nKey.KANBAN$MASTER_COPY)}
            testId="kanban-source-master"
            options={[
              { value: "local", label: t(I18nKey.KANBAN$MASTER_LOCAL) },
              { value: "remote", label: t(I18nKey.KANBAN$MASTER_REMOTE) },
            ]}
          />
        </div>
        {provider === "code" ? null : (
          <label className="block text-xs text-tertiary-light">
            {t(I18nKey.KANBAN$SOURCE_ISSUES)}
            <textarea
              data-testid="kanban-source-issues"
              value={snapshot}
              onChange={(event) => setSnapshot(event.target.value)}
              className={cn(
                formControlMultilineFieldClassName,
                "mt-1 min-h-28",
              )}
            />
            <span className="mt-1 block text-[11px] leading-4">
              {t(I18nKey.KANBAN$SOURCE_ISSUES_HINT)}
            </span>
          </label>
        )}
        <div className="flex justify-end gap-2">
          <BrandButton
            type="button"
            variant="secondary"
            testId="kanban-map-source-cancel"
            onClick={() => setMappingOpen(false)}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            type="button"
            variant="primary"
            testId="kanban-map-source-submit"
            onClick={submit}
          >
            {t(I18nKey.KANBAN$SYNC_NOW)}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}
