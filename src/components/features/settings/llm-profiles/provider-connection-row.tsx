import { useTranslation } from "react-i18next";
import EditIcon from "#/icons/u-edit.svg?react";
import DeleteIcon from "#/icons/u-delete.svg?react";
import { KeyStatusIcon } from "#/components/features/settings/key-status-icon";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import { cn } from "#/utils/utils";
import {
  settingsListIconActionButtonClassName,
  settingsListRowClassName,
} from "#/utils/settings-list-classes";
import { I18nKey } from "#/i18n/declaration";

interface ProviderConnectionRowProps {
  connection: ProviderConnection;
  /** Number of LLM profiles linked to this connection. */
  linkedProfileCount: number;
  onEdit: (connection: ProviderConnection) => void;
  onDelete: (connection: ProviderConnection) => void;
}

export function ProviderConnectionRow({
  connection,
  linkedProfileCount,
  onEdit,
  onDelete,
}: ProviderConnectionRowProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="provider-connection-row"
      className={cn(settingsListRowClassName, "justify-between gap-3")}
    >
      <div className="flex min-w-0 flex-1 items-center gap-3">
        <span
          className="min-w-0 max-w-full truncate text-sm font-medium text-white"
          title={connection.display_name}
        >
          {connection.display_name}
        </span>
        <span className="min-w-0 max-w-full truncate text-sm text-[var(--oh-muted)]">
          {connection.provider}
        </span>
        <span className="shrink-0 text-sm text-[var(--oh-muted)]">
          {t(I18nKey.SETTINGS$PROVIDER_CONNECTION_MODEL_COUNT, {
            count: linkedProfileCount,
          })}
        </span>
        <KeyStatusIcon isSet={connection.api_key_set} />
      </div>
      <div className="flex shrink-0 items-center gap-1">
        <button
          type="button"
          data-testid="provider-connection-edit"
          aria-label={t(I18nKey.SETTINGS$PROVIDER_CONNECTION_EDIT_TITLE)}
          className={settingsListIconActionButtonClassName}
          onClick={() => onEdit(connection)}
        >
          <EditIcon width={16} height={16} />
        </button>
        <button
          type="button"
          data-testid="provider-connection-delete"
          aria-label={t(I18nKey.SETTINGS$PROVIDER_CONNECTION_DELETE_TITLE)}
          className={settingsListIconActionButtonClassName}
          onClick={() => onDelete(connection)}
        >
          <DeleteIcon width={16} height={16} />
        </button>
      </div>
    </div>
  );
}
