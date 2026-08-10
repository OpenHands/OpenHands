import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { useSearchProviders } from "#/hooks/query/use-search-providers";
import {
  useCreateProviderConnection,
  useValidateProviderConnection,
} from "#/hooks/mutation/use-provider-connection-mutations";
import type { ValidateConnectionResponse } from "#/api/provider-connections-service";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";

interface ConnectProviderWizardProps {
  isOpen: boolean;
  onClose: () => void;
  /** Optional preselected provider (e.g., from an empty-state CTA). */
  defaultProvider?: string;
}

/**
 * Connect-a-Provider wizard: pick a vendor → paste one key → validate to see
 * the catalog the key grants → save the connection. The key is stored as a
 * named secret server-side and never returned here; only a boolean is shown.
 *
 * Local-only in this release; the cloud app-server mirror is a follow-up.
 */
export function ConnectProviderWizard({
  isOpen,
  onClose,
  defaultProvider,
}: ConnectProviderWizardProps) {
  const { t } = useTranslation("openhands");
  const { data: providers, isLoading: providersLoading } = useSearchProviders();
  const createConnection = useCreateProviderConnection();
  const validateExisting = useValidateProviderConnection();

  const [provider, setProvider] = useState<string>(defaultProvider ?? "");
  const [key, setKey] = useState("");
  const [label, setLabel] = useState("");
  // Catalog surfaced by a successful create-then-validate round-trip.
  const [validated, setValidated] = useState<ValidateConnectionResponse | null>(
    null,
  );
  const [submitting, setSubmitting] = useState(false);
  const keyInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setProvider(defaultProvider ?? "");
      setKey("");
      setLabel("");
      setValidated(null);
    }
  }, [isOpen, defaultProvider]);

  const providerItems = useMemo(
    () => (providers ?? []).map((p) => ({ key: p.name, label: p.name })),
    [providers],
  );

  const canConnect = Boolean(provider && key.trim()) && !submitting;

  const handleConnect = async () => {
    if (!canConnect) {
      displayErrorToast(t(I18nKey.SETTINGS$CONNECTION_PROVIDER_REQUIRED));
      return;
    }
    setSubmitting(true);
    try {
      const conn = await createConnection.mutateAsync({
        provider,
        key: key.trim(),
        label: label.trim() || undefined,
      });
      // Validate immediately so the user sees the catalog the key grants.
      const result = await validateExisting.mutateAsync(conn.id);
      setValidated(result);
      if (result.ok) {
        displaySuccessToast(
          t(I18nKey.SETTINGS$CONNECTION_CREATED, { provider }),
        );
        onClose();
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
    } finally {
      setSubmitting(false);
    }
  };

  const footer = (
    <>
      <BrandButton
        testId="connect-provider-cancel"
        type="button"
        variant="secondary"
        onClick={onClose}
        isDisabled={submitting}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
      <BrandButton
        testId="connect-provider-submit"
        type="button"
        variant="primary"
        onClick={handleConnect}
        isDisabled={!canConnect}
        aria-busy={submitting}
      >
        {submitting
          ? t(I18nKey.STATUS$SAVING)
          : t(I18nKey.SETTINGS$CONNECTION_SAVE)}
      </BrandButton>
    </>
  );

  return (
    <ApiKeyModalBase
      isOpen={isOpen}
      title={t(I18nKey.SETTINGS$CONNECT_PROVIDER)}
      width="md"
      footer={footer}
      onClose={onClose}
      initialFocusRef={keyInputRef}
    >
      <div className="flex flex-col gap-4">
        <SettingsDropdownInput
          testId="connection-provider-select"
          name="connection-provider"
          label={t(I18nKey.SETTINGS$CONNECTION_PROVIDER_LABEL)}
          items={providerItems}
          selectedKey={provider}
          placeholder={t(I18nKey.SETTINGS$CONNECTION_PROVIDER_PLACEHOLDER)}
          isLoading={providersLoading}
          onSelectionChange={(k) => setProvider(k ? String(k) : "")}
        />
        <SettingsInput
          testId="connection-api-key"
          name="connection-api-key"
          ref={keyInputRef}
          label={t(I18nKey.SETTINGS$CONNECTION_API_KEY_LABEL)}
          type="password"
          value={key}
          placeholder={t(I18nKey.SETTINGS$CONNECTION_API_KEY_PLACEHOLDER)}
          onChange={setKey}
          showRequiredTag
        />
        <SettingsInput
          testId="connection-label"
          name="connection-label"
          label={t(I18nKey.SETTINGS$CONNECTION_LABEL_FIELD)}
          type="text"
          value={label}
          onChange={setLabel}
          showOptionalTag
        />
        {validateExisting.isPending || submitting ? (
          <LoadingSpinner size="small" />
        ) : null}
        {validated?.ok ? (
          <p
            data-testid="connection-validated-summary"
            className="text-sm leading-5 text-success"
          >
            {t(I18nKey.SETTINGS$CONNECTION_VALIDATED, {
              count: validated.models.length,
            })}
          </p>
        ) : null}
        {validated && !validated.ok ? (
          <p
            data-testid="connection-invalid-summary"
            className="text-sm leading-5 text-danger"
          >
            {t(I18nKey.SETTINGS$CONNECTION_INVALID, {
              error: validated.error ?? "",
            })}
          </p>
        ) : null}
      </div>
    </ApiKeyModalBase>
  );
}
