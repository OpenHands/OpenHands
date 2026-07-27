import { Pencil, Trash2 } from "lucide-react";
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
      </div>
    </div>
  );
}

interface SecretListItemProps {
  title: string;
  description?: string;
  onEdit: () => void;
  onDelete: () => void;
}

export function SecretListItem({
  title,
  description,
  onEdit,
  onDelete,
}: SecretListItemProps) {
  return (
    <tr data-testid="secret-item" className={settingsListTableRowClassName}>
      <td
        className={cn(
          settingsListTableCellClassName,
          "text-content-2 truncate",
        )}
        title={title}
      >
        {title}
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
      </td>
    </tr>
  );
}
