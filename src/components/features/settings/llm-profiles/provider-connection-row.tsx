import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { KeyStatusIcon } from "#/components/features/settings/key-status-icon";
import { EllipsisButton } from "#/components/features/conversation-panel/ellipsis-button";
import { ProviderConnectionActionsMenu } from "./provider-connection-actions-menu";
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
  onAddModels: (connection: ProviderConnection) => void;
  onEdit: (connection: ProviderConnection) => void;
  onDelete: (connection: ProviderConnection) => void;
}

export function ProviderConnectionRow({
  connection,
  linkedProfileCount,
  onAddModels,
  onEdit,
  onDelete,
}: ProviderConnectionRowProps) {
  const { t } = useTranslation("openhands");
  const [menuOpen, setMenuOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);

  return (
    <div
      data-testid="provider-connection-row"
      className={cn(settingsListRowClassName, "justify-between gap-3")}
    >
      <div className="flex min-w-0 flex-1 items-center gap-3">
        <span
          className="min-w-0 max-w-full truncate text-sm font-medium text-contrast"
          title={connection.display_name}
        >
          {connection.display_name}
        </span>
        <span className="min-w-0 max-w-full truncate text-sm text-muted">
          {connection.provider}
        </span>
        <span className="shrink-0 text-sm text-muted">
          {t(I18nKey.SETTINGS$PROVIDER_CONNECTION_MODEL_COUNT, {
            count: linkedProfileCount,
          })}
        </span>
        <KeyStatusIcon isSet={connection.api_key_set} />
      </div>
      <div className="relative shrink-0">
        <EllipsisButton
          ref={triggerRef}
          onClick={() => setMenuOpen((open) => !open)}
          ariaLabel={t(I18nKey.SETTINGS$PROVIDER_CONNECTION_MENU)}
          testId="provider-connection-menu-trigger"
          className={settingsListIconActionButtonClassName}
        />
        {menuOpen && (
          <ProviderConnectionActionsMenu
            anchorRef={triggerRef}
            onAddModels={() => onAddModels(connection)}
            onEdit={() => onEdit(connection)}
            onDelete={() => onDelete(connection)}
            onClose={() => setMenuOpen(false)}
          />
        )}
      </div>
    </div>
  );
}
