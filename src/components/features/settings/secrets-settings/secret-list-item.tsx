import { Pencil, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  settingsListIconActionButtonClassName,
  settingsListRowClassName,
  settingsListTableCellClassName,
  settingsListTableRowClassName,
} from "#/utils/settings-list-classes";

export function SecretListItemSkeleton() {
  return (
    <div
      className={cn(
        settingsListRowClassName,
        "justify-between border-t border-border first:border-t-0",
      )}
    >
      <div className="flex min-w-0 flex-1 items-center gap-4">
        <span className="skeleton h-4 w-1/4" />
        <span className="skeleton h-4 w-1/2" />
      </div>
      <div className="flex items-center gap-1">
        <span className="skeleton h-4 w-4" />
        <span className="skeleton h-4 w-4" />
      </div>
    </div>
  );
}

interface SecretListItemProps {
  title: string;
  description?: string;
  /**
   * Whether this secret is personal (owned by the current user) or shared
   * across the organization. Organization-scoped secrets are usable by
   * everyone but cannot be edited or deleted through the personal secrets
   * endpoints, so their action buttons are hidden.
   */
  scope?: "personal" | "organization";
  onEdit: () => void;
  onDelete: () => void;
}

export function SecretListItem({
  title,
  description,
  scope = "personal",
  onEdit,
  onDelete,
}: SecretListItemProps) {
  const { t } = useTranslation("openhands");
  const isOrgShared = scope === "organization";

  return (
    <tr data-testid="secret-item" className={settingsListTableRowClassName}>
      <td
        className={cn(
          settingsListTableCellClassName,
          "text-content-2 truncate",
        )}
        title={title}
      >
        <div className="flex items-center gap-2">
          <span className="truncate">{title}</span>
          {isOrgShared && (
            <span
              data-testid="org-shared-badge"
              className="shrink-0 rounded bg-surface-raised px-1.5 py-0.5 text-xs text-muted"
            >
              {t(I18nKey.SECRETS$ORG_SHARED_BADGE)}
            </span>
          )}
        </div>
      </td>

      <td
        className={cn(
          settingsListTableCellClassName,
          "truncate text-content-2 opacity-80",
        )}
        title={description || ""}
      >
        {description || ""}
      </td>

      <td className={settingsListTableCellClassName}>
        {!isOrgShared && (
          <div className="flex items-center justify-end gap-0.5">
            <button
              data-testid="edit-secret-button"
              type="button"
              onClick={onEdit}
              aria-label={`Edit ${title}`}
              className={settingsListIconActionButtonClassName}
            >
              <Pencil aria-hidden className="size-4" strokeWidth={2} />
            </button>
            <button
              data-testid="delete-secret-button"
              type="button"
              onClick={onDelete}
              aria-label={`Delete ${title}`}
              className={settingsListIconActionButtonClassName}
            >
              <Trash2 aria-hidden className="size-4" strokeWidth={2} />
            </button>
          </div>
        )}
      </td>
    </tr>
  );
}
