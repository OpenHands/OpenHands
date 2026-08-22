import { useEffect, useRef, useState } from "react";
import {
  Autocomplete,
  AutocompleteItem,
  AutocompleteSection,
} from "@heroui/react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { KeyStatusIcon } from "#/components/features/settings/key-status-icon";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import { useCreateProviderConnection } from "#/hooks/mutation/use-create-provider-connection";
import { useUpdateProviderConnection } from "#/hooks/mutation/use-update-provider-connection";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { I18nKey } from "#/i18n/declaration";
import { useSearchProviders } from "#/hooks/query/use-search-providers";
import { mapProvider } from "#/utils/map-provider";
import { formControlSettingsFieldClassName } from "#/utils/form-control-classes";
import { heroUiAutocompleteSelectorButtonClassName } from "#/ui/combobox-caret";

const DEFAULT_PROVIDER = "custom";

interface ProviderConnectionModalProps {
  /** When `null` the modal is closed; otherwise it edits that connection. */
  connection?: ProviderConnection | null;
  /** When true the modal creates a new connection. */
  isCreate: boolean;
  onClose: () => void;
  /** Called with the saved connection so a caller can select it (create flow). */
  onSaved?: (connection: ProviderConnection) => void;
}

/**
 * Single modal for creating, editing, and rotating a provider connection. Create
 * is a POST; every edit (rename, base_url, key rotation) is a single PATCH. The
 * key field follows the same "empty means unchanged" convention as the profile
 * form, so a blank key on edit is simply omitted from the request — the
 * agent-server rejects `api_key: null`, so we never send it.
 */
export function ProviderConnectionModal({
  connection,
  isCreate,
  onClose,
  onSaved,
}: ProviderConnectionModalProps) {
  const { t } = useTranslation("openhands");
  const createConnection = useCreateProviderConnection();
  const updateConnection = useUpdateProviderConnection();
  const { data: providers = [] } = useSearchProviders();
  const firstFieldRef = useRef<HTMLInputElement>(null);

  const [displayName, setDisplayName] = useState("");
  const [provider, setProvider] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");

  const isOpen = isCreate || Boolean(connection);
  const keyAlreadySet = Boolean(connection?.api_key_set);

  useEffect(() => {
    setDisplayName(connection?.display_name ?? "");
    setProvider(connection?.provider ?? (isCreate ? null : DEFAULT_PROVIDER));
    setBaseUrl(connection?.base_url ?? "");
    setApiKey("");
  }, [connection, isCreate]);

  const isPending = createConnection.isPending || updateConnection.isPending;
  const trimmedName = displayName.trim();
  const trimmedKey = apiKey.trim();
  // On create the key is required; on edit an empty key means "leave unchanged".
  const isValid =
    Boolean(trimmedName) &&
    Boolean(provider?.trim()) &&
    (!isCreate || Boolean(trimmedKey));

  const verifiedProviders = providers.filter((candidate) => candidate.verified);
  const unverifiedProviders = providers.filter(
    (candidate) => !candidate.verified,
  );

  if (!isOpen) return null;

  const handleClose = () => {
    if (!isPending) onClose();
  };

  const handleSubmit = async () => {
    if (!isValid || isPending) return;
    const trimmedBaseUrl = baseUrl.trim();

    try {
      if (isCreate) {
        const created = await createConnection.mutateAsync({
          display_name: trimmedName,
          provider: provider?.trim() || DEFAULT_PROVIDER,
          api_key: trimmedKey,
          base_url: trimmedBaseUrl || null,
        });
        displaySuccessToast(
          t(I18nKey.SETTINGS$PROVIDER_CONNECTION_CREATED, {
            name: created.display_name,
          }),
        );
        onSaved?.(created);
      } else if (connection) {
        const updated = await updateConnection.mutateAsync({
          id: connection.id,
          request: {
            display_name: trimmedName,
            provider: provider?.trim() || DEFAULT_PROVIDER,
            base_url: trimmedBaseUrl || null,
            // Omit the key entirely when left blank so the stored key is kept.
            ...(trimmedKey ? { api_key: trimmedKey } : {}),
          },
        });
        displaySuccessToast(
          t(I18nKey.SETTINGS$PROVIDER_CONNECTION_UPDATED, {
            name: updated.display_name,
          }),
        );
        onSaved?.(updated);
      }
      onClose();
    } catch (error) {
      displayErrorToast(getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)));
    }
  };

  const footer = (
    <>
      <BrandButton
        type="button"
        variant="tertiary"
        onClick={handleClose}
        isDisabled={isPending}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
      <BrandButton
        testId="provider-connection-submit"
        type="button"
        variant="primary"
        onClick={handleSubmit}
        isDisabled={isPending || !isValid}
        aria-busy={isPending}
      >
        {isPending ? <LoadingSpinner size="small" /> : t(I18nKey.BUTTON$SAVE)}
      </BrandButton>
    </>
  );

  return (
    <ApiKeyModalBase
      isOpen
      title={
        isCreate
          ? t(I18nKey.SETTINGS$PROVIDER_CONNECTION_ADD_TITLE)
          : t(I18nKey.SETTINGS$PROVIDER_CONNECTION_EDIT_TITLE)
      }
      footer={footer}
      onClose={handleClose}
      initialFocusRef={firstFieldRef}
    >
      <div
        data-testid="provider-connection-modal"
        className="flex flex-col gap-4"
      >
        <SettingsInput
          ref={firstFieldRef}
          testId="provider-connection-name-input"
          label={t(I18nKey.SETTINGS$NAME)}
          type="text"
          className="w-full"
          value={displayName}
          onChange={setDisplayName}
          required
        />
        {isCreate ? (
          <fieldset className="flex flex-col gap-2.5 w-full">
            <label className="text-sm">
              {t(I18nKey.SETTINGS$PROVIDER_CONNECTION_PROVIDER)}
            </label>
            <Autocomplete
              data-testid="provider-connection-provider-input"
              isRequired
              isVirtualized={false}
              name="provider-connection-provider-input"
              aria-label={t(I18nKey.SETTINGS$PROVIDER_CONNECTION_PROVIDER)}
              isClearable={false}
              selectedKey={provider}
              onSelectionChange={(key) => setProvider(key?.toString() ?? null)}
              classNames={{
                popoverContent:
                  "bg-content1 rounded-xl border border-[var(--oh-border)]",
                selectorButton: heroUiAutocompleteSelectorButtonClassName,
              }}
              selectorButtonProps={{ disableRipple: true }}
              inputProps={{
                classNames: {
                  inputWrapper: formControlSettingsFieldClassName,
                },
              }}
            >
              <AutocompleteSection
                title={t(I18nKey.MODEL_SELECTOR$VERIFIED)}
                classNames={{ heading: "text-[var(--oh-muted)]" }}
              >
                {verifiedProviders.map((candidate) => (
                  <AutocompleteItem
                    data-testid={`provider-item-${candidate.name}`}
                    key={candidate.name}
                    textValue={mapProvider(candidate.name)}
                  >
                    {mapProvider(candidate.name)}
                  </AutocompleteItem>
                ))}
              </AutocompleteSection>
              {unverifiedProviders.length > 0 ? (
                <AutocompleteSection
                  title={t(I18nKey.MODEL_SELECTOR$OTHERS)}
                  classNames={{ heading: "text-[var(--oh-muted)]" }}
                >
                  {unverifiedProviders.map((candidate) => (
                    <AutocompleteItem
                      key={candidate.name}
                      textValue={mapProvider(candidate.name)}
                    >
                      {mapProvider(candidate.name)}
                    </AutocompleteItem>
                  ))}
                </AutocompleteSection>
              ) : null}
            </Autocomplete>
          </fieldset>
        ) : (
          <SettingsInput
            testId="provider-connection-provider-input"
            label={t(I18nKey.SETTINGS$PROVIDER_CONNECTION_PROVIDER)}
            type="text"
            className="w-full"
            value={provider ?? ""}
            onChange={setProvider}
          />
        )}
        <SettingsInput
          testId="provider-connection-api-key-input"
          label={t(I18nKey.SETTINGS_FORM$API_KEY)}
          type="password"
          className="w-full"
          value={apiKey}
          // eslint-disable-next-line i18next/no-literal-string -- masked-key sentinel, not translatable
          placeholder={keyAlreadySet ? "<hidden>" : ""}
          onChange={setApiKey}
          startContent={
            keyAlreadySet ? <KeyStatusIcon isSet={keyAlreadySet} /> : undefined
          }
          hint={
            keyAlreadySet
              ? t(I18nKey.SETTINGS$PROVIDER_CONNECTION_ROTATE_HINT)
              : undefined
          }
        />
        <SettingsInput
          testId="provider-connection-base-url-input"
          label={t(I18nKey.SETTINGS$BASE_URL)}
          type="text"
          className="w-full"
          value={baseUrl}
          // eslint-disable-next-line i18next/no-literal-string -- example value, not translatable
          placeholder="https://api.openai.com"
          onChange={setBaseUrl}
          showOptionalTag
        />
      </div>
    </ApiKeyModalBase>
  );
}
