import { Pencil, RefreshCw, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  settingsListIconActionButtonClassName,
  settingsListRowClassName,
  settingsListTableCellClassName,
  settingsListTableRowClassName,
} from "#/utils/settings-list-classes";
import type { CustomWebhook } from "#/types/webhook";

export function WebhookListItemSkeleton() {
  return (
    <div
      className={cn(
        settingsListRowClassName,
        "justify-between border-t border-[var(--oh-border)] first:border-t-0",
      )}
    >
      <div className="flex min-w-0 flex-1 items-center gap-4">
        <span className="skeleton h-4 w-1/4" />
        <span className="skeleton h-4 w-1/2" />
      </div>
      <div className="flex items-center gap-1">
        <span className="skeleton h-4 w-4" />
        <span className="skeleton h-4 w-4" />
        <span className="skeleton h-4 w-4" />
      </div>
    </div>
  );
}

interface WebhookListItemProps {
  webhook: CustomWebhook;
  onEdit: () => void;
  onRotateSecret: () => void;
  onDelete: () => void;
}

export function WebhookListItem({
  webhook,
  onEdit,
  onRotateSecret,
  onDelete,
}: WebhookListItemProps) {
  const { t } = useTranslation("openhands");

  return (
    <tr data-testid="webhook-item" className={settingsListTableRowClassName}>
      <td
        className={cn(
          settingsListTableCellClassName,
          "text-content-2 truncate",
        )}
        title={webhook.name}
      >
        {webhook.name}
        {!webhook.enabled && (
          <span className="ml-2 rounded-full border border-[var(--oh-border)] px-2 py-0.5 text-xs text-muted">
            {t(I18nKey.AUTOMATIONS$WEBHOOKS$DISABLED)}
          </span>
        )}
      </td>

      <td
        className={cn(
          settingsListTableCellClassName,
          "truncate font-mono text-content-2 opacity-80",
        )}
        title={webhook.source}
      >
        {webhook.source}
      </td>

      <td
        className={cn(
          settingsListTableCellClassName,
          "truncate text-content-2 opacity-80",
        )}
        title={webhook.webhook_url}
      >
        {webhook.webhook_url}
      </td>

      <td className={settingsListTableCellClassName}>
        <div className="flex items-center justify-end gap-0.5">
          <button
            data-testid="rotate-webhook-secret-button"
            type="button"
            onClick={onRotateSecret}
            aria-label={t(I18nKey.AUTOMATIONS$WEBHOOKS$ROTATE_SECRET_FOR, {
              name: webhook.name,
            })}
            className={settingsListIconActionButtonClassName}
          >
            <RefreshCw aria-hidden className="size-4" strokeWidth={2} />
          </button>
          <button
            data-testid="edit-webhook-button"
            type="button"
            onClick={onEdit}
            aria-label={t(I18nKey.AUTOMATIONS$WEBHOOKS$EDIT_FOR, {
              name: webhook.name,
            })}
            className={settingsListIconActionButtonClassName}
          >
            <Pencil aria-hidden className="size-4" strokeWidth={2} />
          </button>
          <button
            data-testid="delete-webhook-button"
            type="button"
            onClick={onDelete}
            aria-label={t(I18nKey.AUTOMATIONS$WEBHOOKS$DELETE_FOR, {
              name: webhook.name,
            })}
            className={settingsListIconActionButtonClassName}
          >
            <Trash2 aria-hidden className="size-4" strokeWidth={2} />
          </button>
        </div>
      </td>
    </tr>
  );
}
