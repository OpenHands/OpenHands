import { useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ConnectProviderWizard } from "./connect-provider-wizard";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { useProviderConnections } from "#/hooks/query/use-provider-connections";
import {
  useDeleteProviderConnection,
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
import { I18nKey } from "#/i18n/declaration";
import { useCanManageOrgProfiles } from "#/hooks/use-can-manage-org-profiles";

function ConnectionRow({ connection }: { connection: ProviderConnection }) {
  const { t } = useTranslation("openhands");
  const deleteConnection = useDeleteProviderConnection();
  const validateConnection = useValidateProviderConnection();
  const canManage = useCanManageOrgProfiles();
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const handleValidate = async () => {
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

  const handleDelete = async () => {
    try {
      await deleteConnection.mutateAsync(connection.id);
      displaySuccessToast(
        t(I18nKey.SETTINGS$CONNECTION_DELETED, {
          provider: connection.provider,
        }),
      );
      setConfirmingDelete(false);
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
    }
  };

  return (
    <li
      data-testid={`connection-row-${connection.id}`}
      className="flex flex-col gap-1 rounded-md border border-tertiary-light/30 p-3"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm font-medium text-white">
          {connection.provider}
          {connection.label ? ` · ${connection.label}` : ""}
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
      {connection.models.length > 0 ? (
        <span
          data-testid={`connection-models-${connection.id}`}
          className="text-xs text-tertiary-light"
        >
          {connection.models.join(", ")}
        </span>
      ) : null}
      {canManage ? (
        <div className="flex flex-wrap gap-2 pt-1">
          <BrandButton
            testId={`connection-revalidate-${connection.id}`}
            type="button"
            variant="tertiary"
            onClick={handleValidate}
            isDisabled={validateConnection.isPending}
          >
            {validateConnection.isPending
              ? t(I18nKey.SETTINGS$CONNECTION_VALIDATING)
              : t(I18nKey.SETTINGS$CONNECTION_REVALIDATE)}
          </BrandButton>
          {confirmingDelete ? (
            <>
              <BrandButton
                testId={`connection-delete-confirm-${connection.id}`}
                type="button"
                variant="danger"
                onClick={handleDelete}
                isDisabled={deleteConnection.isPending}
                aria-busy={deleteConnection.isPending}
              >
                {t(I18nKey.BUTTON$CONFIRM)}
              </BrandButton>
              <BrandButton
                testId={`connection-delete-cancel-${connection.id}`}
                type="button"
                variant="secondary"
                onClick={() => setConfirmingDelete(false)}
              >
                {t(I18nKey.BUTTON$CANCEL)}
              </BrandButton>
            </>
          ) : (
            <BrandButton
              testId={`connection-delete-${connection.id}`}
              type="button"
              variant="ghost-danger"
              onClick={() => setConfirmingDelete(true)}
            >
              {t(I18nKey.SETTINGS$PROFILE_DELETE)}
            </BrandButton>
          )}
        </div>
      ) : null}
    </li>
  );
}

/**
 * Lists connected providers with re-validate / delete actions, and a
 * "Connect a Provider" button that opens the wizard. Rendered above the
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
