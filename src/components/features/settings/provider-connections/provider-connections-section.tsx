import { useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { ConnectProviderWizard } from "./connect-provider-wizard";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { useProviderConnections } from "#/hooks/query/use-provider-connections";
import {
  useDeleteProviderConnection,
  useUpdateProviderConnection,
  useValidateProviderConnection,
} from "#/hooks/mutation/use-provider-connection-mutations";
import type { ProviderConnection } from "#/api/provider-connections-service";
import {
  assertConnectionsSupportedLocally,
  isProviderConnectionsNotOnCloudError,
} from "#/api/provider-connections-service";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { formatRelativeTime } from "#/utils/format-relative-time";
import { I18nKey } from "#/i18n/declaration";
import { useCanManageOrgProfiles } from "#/hooks/use-can-manage-org-profiles";

function ConnectionRow({ connection }: { connection: ProviderConnection }) {
  const { t, i18n } = useTranslation("openhands");
  const deleteConnection = useDeleteProviderConnection();
  const updateConnection = useUpdateProviderConnection();
  const validateConnection = useValidateProviderConnection();
  const canManage = useCanManageOrgProfiles();
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [rotating, setRotating] = useState(false);
  const [newKey, setNewKey] = useState("");

  const handleRefresh = async () => {
    try {
      const result = await validateConnection.mutateAsync(connection.id);
      if (result.ok) {
        displaySuccessToast(
          t(I18nKey.SETTINGS$CONNECTION_VALIDATED, {
            count: result.models.length,
          }),
        );
      } else {
        displayErrorToast(
          t(I18nKey.SETTINGS$CONNECTION_INVALID, {
            error: result.error ?? "",
          }),
        );
      }
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
    }
  };

  const handleRotate = async () => {
    if (!newKey.trim()) return;
    setRotating(true);
    try {
      await updateConnection.mutateAsync({
        id: connection.id,
        request: { key: newKey.trim() },
      });
      displaySuccessToast(t(I18nKey.SETTINGS$CONNECTION_ROTATED));
      setNewKey("");
      setRotating(false);
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
      setRotating(false);
    }
  };

  const handleDelete = async () => {
    try {
      const result = await deleteConnection.mutateAsync(connection.id);
      displaySuccessToast(
        t(I18nKey.SETTINGS$CONNECTION_DELETED, {
          provider: connection.provider,
        }),
      );
      // Warn if profiles were left pointing at the now-deleted key.
      if (result.affectedProfiles.length > 0) {
        displayErrorToast(
          t(I18nKey.SETTINGS$CONNECTION_DELETE_CONFIRMATION, {
            provider: connection.provider,
          }),
        );
      }
      setConfirmingDelete(false);
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
    }
  };

  const lastRefreshed = connection.lastValidatedAt
    ? t(I18nKey.SETTINGS$CONNECTION_LAST_REFRESHED, {
        time: formatRelativeTime(
          new Date(connection.lastValidatedAt * 1000).toISOString(),
          i18n.language,
          t,
        ),
      })
    : null;

  return (
    <li
      data-testid={`connection-row-${connection.id}`}
      className="flex flex-col gap-1 rounded-md border border-tertiary-light/30 p-3"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm font-medium text-white">
          {connection.provider}
          {connection.label ? ` · ${connection.label}` : ""}
          {connection.models.length > 0
            ? ` · ${t(I18nKey.SETTINGS$CONNECTION_MODELS_COUNT, {
                count: connection.models.length,
              })}`
            : ""}
          {lastRefreshed ? ` · ${lastRefreshed}` : ""}
        </span>
        {connection.apiKeySet ? (
          <span
            data-testid={`connection-key-set-${connection.id}`}
            className="text-xs text-tertiary-light"
          >
            {t(I18nKey.SETTINGS$CONNECTION_API_KEY_SET)}
          </span>
        ) : null}
      </div>
      {canManage ? (
        <div className="flex flex-col gap-2 pt-1">
          <div className="flex flex-wrap gap-2">
            <BrandButton
              testId={`connection-refresh-${connection.id}`}
              type="button"
              variant="tertiary"
              onClick={handleRefresh}
              isDisabled={validateConnection.isPending}
            >
              {validateConnection.isPending
                ? t(I18nKey.SETTINGS$CONNECTION_VALIDATING)
                : t(I18nKey.SETTINGS$CONNECTION_REFRESH)}
            </BrandButton>
            <BrandButton
              testId={`connection-rotate-${connection.id}`}
              type="button"
              variant="tertiary"
              onClick={() => setRotating((v) => !v)}
            >
              {t(I18nKey.SETTINGS$CONNECTION_ROTATE_KEY)}
            </BrandButton>
            {confirmingDelete ? (
              <>
                <BrandButton
                  testId={`connection-disconnect-confirm-${connection.id}`}
                  type="button"
                  variant="danger"
                  onClick={handleDelete}
                  isDisabled={deleteConnection.isPending}
                  aria-busy={deleteConnection.isPending}
                >
                  {t(I18nKey.BUTTON$CONFIRM)}
                </BrandButton>
                <BrandButton
                  testId={`connection-disconnect-cancel-${connection.id}`}
                  type="button"
                  variant="secondary"
                  onClick={() => setConfirmingDelete(false)}
                >
                  {t(I18nKey.BUTTON$CANCEL)}
                </BrandButton>
              </>
            ) : (
              <BrandButton
                testId={`connection-disconnect-${connection.id}`}
                type="button"
                variant="ghost-danger"
                onClick={() => setConfirmingDelete(true)}
              >
                {t(I18nKey.SETTINGS$CONNECTION_DISCONNECT)}
              </BrandButton>
            )}
          </div>
          {rotating ? (
            <div className="flex flex-wrap items-end gap-2">
              <SettingsInput
                testId={`connection-rotate-key-input-${connection.id}`}
                name={`connection-rotate-key-${connection.id}`}
                type="password"
                label={t(I18nKey.SETTINGS$CONNECTION_ROTATE_KEY)}
                value={newKey}
                placeholder={t(
                  I18nKey.SETTINGS$CONNECTION_ROTATE_KEY_PLACEHOLDER,
                )}
                onChange={setNewKey}
              />
              <BrandButton
                testId={`connection-rotate-confirm-${connection.id}`}
                type="button"
                variant="primary"
                onClick={handleRotate}
                isDisabled={!newKey.trim() || updateConnection.isPending}
                aria-busy={updateConnection.isPending}
              >
                {t(I18nKey.SETTINGS$CONNECTION_ROTATE_KEY)}
              </BrandButton>
            </div>
          ) : null}
        </div>
      ) : null}
    </li>
  );
}

/**
 * Lists connected providers with Refresh / Rotate key / Disconnect actions and
 * a "Connect a Provider" button that opens the wizard. Rendered above the
 * per-model LLM profile list so a single connection can back many profiles.
 *
 * Local-only in this release. On cloud the section hides itself (the cloud
 * mirror is a follow-up), so it never makes a network call that would throw.
 */
export function ProviderConnectionsSection() {
  const { t } = useTranslation("openhands");
  const canManage = useCanManageOrgProfiles();
  const [wizardOpen, setWizardOpen] = useState(false);

  // Skip the query on cloud: the service throws there, and the section is
  // hidden entirely in that case (see below) so the query never fires.
  const { data, isLoading, error } = useProviderConnections({
    enabled: canManage,
  });

  // Cloud gate: connections are local-first. Detect via the service's guard
  // without an extra network round-trip.
  const cloudUnavailable = (() => {
    try {
      assertConnectionsSupportedLocally();
      return false;
    } catch (e) {
      return isProviderConnectionsNotOnCloudError(e);
    }
  })();

  if (cloudUnavailable) return null;

  return (
    <section
      data-testid="provider-connections-section"
      className="flex flex-col gap-3"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-col gap-1">
          <h2 className="text-base font-medium text-white">
            {t(I18nKey.SETTINGS$PROVIDER_CONNECTIONS)}
          </h2>
          <p className="text-xs leading-4 text-tertiary-light">
            {t(I18nKey.SETTINGS$PROVIDER_CONNECTIONS_HINT)}
          </p>
        </div>
        {canManage ? (
          <BrandButton
            testId="open-connect-provider-wizard"
            type="button"
            variant="secondary"
            className="ml-auto"
            onClick={() => setWizardOpen(true)}
          >
            {t(I18nKey.SETTINGS$CONNECT_PROVIDER)}
          </BrandButton>
        ) : null}
      </div>

      {isLoading ? <LoadingSpinner size="small" /> : null}
      {error ? (
        <p
          data-testid="provider-connections-error"
          className="text-sm leading-5 text-danger"
        >
          {t(I18nKey.ERROR$GENERIC)}
        </p>
      ) : null}
      {!isLoading && (data ?? []).length === 0 ? (
        <p
          data-testid="provider-connections-empty"
          className="text-sm leading-5 text-tertiary-light"
        >
          {t(I18nKey.SETTINGS$CONNECTION_EMPTY)}
        </p>
      ) : null}
      {(data ?? []).length > 0 ? (
        <ul className="flex flex-col gap-2">
          {data!.map((c) => (
            <ConnectionRow key={c.id} connection={c} />
          ))}
        </ul>
      ) : null}

      <ConnectProviderWizard
        isOpen={wizardOpen}
        onClose={() => setWizardOpen(false)}
      />
    </section>
  );
}
