import React from "react";
import { useTranslation } from "react-i18next";
import type {
  KanbanCard,
  UpdateCardPayload,
} from "#/api/kanban-service/kanban-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { I18nKey } from "#/i18n/declaration";
import { formControlMultilineFieldClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import { formatUsd } from "./kanban-cost";

export interface CardDetailPanelProps {
  card: KanbanCard;
  onClose: () => void;
  onUpdate?: (cardId: string, payload: UpdateCardPayload) => void;
  onDelete?: (cardId: string) => void;
}

export function CardDetailPanel({
  card,
  onClose,
  onUpdate,
  onDelete,
}: CardDetailPanelProps) {
  const { t } = useTranslation("openhands");
  const [description, setDescription] = React.useState(card.description ?? "");
  const [confirmDelete, setConfirmDelete] = React.useState(false);

  React.useEffect(() => {
    setDescription(card.description ?? "");
  }, [card.id, card.description]);

  return (
    <aside
      data-testid="kanban-card-detail"
      className={cn(
        "flex h-[min(50%,24rem)] w-full shrink-0 flex-col overflow-y-auto border-[var(--oh-border)] bg-base-secondary p-4",
        "border-t lg:h-full lg:w-[min(100%,22rem)] lg:border-t-0 lg:border-l",
      )}
    >
      <div className="mb-4 flex items-start justify-between gap-2">
        <h2 className="text-sm font-medium leading-5 text-white">
          {card.title}
        </h2>
        <BrandButton
          type="button"
          variant="tertiary"
          ariaLabel={t(I18nKey.KANBAN$CLOSE_DETAIL)}
          testId="kanban-card-detail-close"
          onClick={onClose}
        >
          {t(I18nKey.BUTTON$CLOSE)}
        </BrandButton>
      </div>

      <label className="mb-4 block text-xs leading-4 text-tertiary-light">
        {t(I18nKey.KANBAN$DESCRIPTION)}
        <textarea
          data-testid="kanban-card-description"
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          onBlur={() =>
            onUpdate?.(card.id, { description: description || null })
          }
          className={cn(formControlMultilineFieldClassName, "mt-1 min-h-24")}
        />
      </label>

      <dl className="space-y-2 text-sm">
        <div className="flex justify-between gap-3">
          <dt className="text-tertiary-light">{t(I18nKey.KANBAN$PRIORITY)}</dt>
          <dd data-testid="kanban-detail-priority">{card.priority}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-tertiary-light">{t(I18nKey.COMMON$STATUS)}</dt>
          <dd>{card.status}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-tertiary-light">{t(I18nKey.KANBAN$ASSIGNEE)}</dt>
          <dd>{card.assignee ?? ""}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-tertiary-light">
            {t(I18nKey.KANBAN$LINKED_BRANCH)}
          </dt>
          <dd data-testid="kanban-detail-branch">{card.linked_branch ?? ""}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-tertiary-light">{t(I18nKey.KANBAN$LINKED_PR)}</dt>
          <dd data-testid="kanban-detail-pr">{card.linked_pr ?? ""}</dd>
        </div>
      </dl>

      <h3 className="mt-6 text-sm font-medium text-white">
        {t(I18nKey.KANBAN$COST_BREAKDOWN)}
      </h3>
      <dl className="mt-2 space-y-2 text-sm">
        <div className="flex justify-between gap-3">
          <dt className="text-tertiary-light">
            {t(I18nKey.KANBAN$COST_ESTIMATE)}
          </dt>
          <dd data-testid="kanban-detail-estimate" className="tabular-nums">
            {formatUsd(card.estimate_cost)}
          </dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-tertiary-light">
            {t(I18nKey.KANBAN$COST_ACTUAL)}
          </dt>
          <dd data-testid="kanban-detail-actual" className="tabular-nums">
            {formatUsd(card.actual_cost)}
          </dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-tertiary-light">{t(I18nKey.KANBAN$TOKENS)}</dt>
          <dd className="tabular-nums">
            {card.actual_tokens ?? card.estimate_tokens ?? 0}
          </dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-tertiary-light">
            {t(I18nKey.KANBAN$TOOL_CALLS)}
          </dt>
          <dd className="tabular-nums">{card.tool_calls ?? 0}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-tertiary-light">
            {t(I18nKey.KANBAN$AGENT_TIME)}
          </dt>
          <dd className="tabular-nums">{card.agent_time ?? 0}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-tertiary-light">{t(I18nKey.KANBAN$MODEL)}</dt>
          <dd>{card.model_used ?? ""}</dd>
        </div>
      </dl>

      <h3 className="mt-6 text-sm font-medium text-white">
        {t(I18nKey.KANBAN$ACTIVITY)}
      </h3>
      {card.activity_log && card.activity_log.length > 0 ? (
        <ul
          data-testid="kanban-activity-log"
          className="mt-2 space-y-1 text-sm text-tertiary-light"
        >
          {card.activity_log.map((item) => (
            <li key={`${item.timestamp}-${item.message}`}>{item.message}</li>
          ))}
        </ul>
      ) : (
        <p className="mt-2 text-sm text-tertiary-light">
          {t(I18nKey.KANBAN$NO_ACTIVITY)}
        </p>
      )}

      {onDelete ? (
        <div className="mt-auto pt-4">
          <BrandButton
            type="button"
            variant="danger"
            testId="kanban-card-delete"
            onClick={() => setConfirmDelete(true)}
          >
            {t(I18nKey.BUTTON$DELETE)}
          </BrandButton>
        </div>
      ) : null}

      {confirmDelete ? (
        <ConfirmationModal
          text={t(I18nKey.KANBAN$DELETE_CARD)}
          onCancel={() => setConfirmDelete(false)}
          onConfirm={() => {
            onDelete?.(card.id);
            setConfirmDelete(false);
          }}
        />
      ) : null}
    </aside>
  );
}
