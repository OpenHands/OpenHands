import { useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ProviderConnectionRow } from "./provider-connection-row";
import { ProviderConnectionModal } from "./provider-connection-modal";
import { DeleteProviderConnectionModal } from "./delete-provider-connection-modal";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import { cn } from "#/utils/utils";
import {
  settingsListContainerClassName,
  settingsListDividerClassName,
} from "#/utils/settings-list-classes";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import { I18nKey } from "#/i18n/declaration";

interface ProviderConnectionsManagerProps {
  connections: ProviderConnection[];
  /** Number of LLM profiles linked to each connection id. */
  linkedCountById: Record<string, number>;
  isLoading: boolean;
  loadError: Error | null;
}

/**
 * Manages shared provider connections: a shared API key + optional base URL
 * that LLM profiles reference by id. Rendered only for the local agent-server,
 * which is the only backend exposing the endpoints.
 */
export function ProviderConnectionsManager({
  connections,
  linkedCountById,
  isLoading,
  loadError,
}: ProviderConnectionsManagerProps) {
  const { t } = useTranslation("openhands");
  const [isCreating, setIsCreating] = useState(false);
  const [connectionToEdit, setConnectionToEdit] =
    useState<ProviderConnection | null>(null);
  const [connectionToDelete, setConnectionToDelete] =
    useState<ProviderConnection | null>(null);

  const renderBody = () => {
    if (isLoading) return null;

    if (loadError) {
      return (
        <div
          data-testid="provider-connections-load-error"
          className={extensionModuleEmptyStateClassName}
        >
          <p className="text-sm text-red-400">
            {t(I18nKey.SETTINGS$PROVIDER_CONNECTIONS_LOAD_ERROR)}
          </p>
        </div>
      );
    }

    if (connections.length === 0) {
      return (
        <div
          data-testid="provider-connections-empty"
          className={extensionModuleEmptyStateClassName}
        >
          <p className="text-sm text-[var(--oh-muted)]">
            {t(I18nKey.SETTINGS$PROVIDER_CONNECTIONS_EMPTY)}
          </p>
        </div>
      );
    }

    return (
      <div
        className={cn(
          settingsListContainerClassName,
          settingsListDividerClassName,
        )}
      >
        {connections.map((connection) => (
          <ProviderConnectionRow
            key={connection.id}
            connection={connection}
            linkedProfileCount={linkedCountById[connection.id] ?? 0}
            onEdit={setConnectionToEdit}
            onDelete={setConnectionToDelete}
          />
        ))}
      </div>
    );
  };

  return (
    <>
      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-col gap-1">
            <h2 className="text-base font-medium text-white">
              {t(I18nKey.SETTINGS$PROVIDER_CONNECTIONS_TITLE)}
            </h2>
            <p className="text-sm text-[var(--oh-muted)]">
              {t(I18nKey.SETTINGS$PROVIDER_CONNECTIONS_SUBLINE)}
            </p>
          </div>
          <BrandButton
            testId="add-provider-connection"
            type="button"
            variant="secondary"
            className="ml-auto"
            onClick={() => setIsCreating(true)}
          >
            {t(I18nKey.SETTINGS$PROVIDER_CONNECTION_ADD)}
          </BrandButton>
        </div>

        {renderBody()}
      </div>

      {isCreating && (
        <ProviderConnectionModal
          isCreate
          onClose={() => setIsCreating(false)}
        />
      )}
      <ProviderConnectionModal
        isCreate={false}
        connection={connectionToEdit}
        onClose={() => setConnectionToEdit(null)}
      />
      <DeleteProviderConnectionModal
        connection={connectionToDelete}
        onClose={() => setConnectionToDelete(null)}
      />
    </>
  );
}
