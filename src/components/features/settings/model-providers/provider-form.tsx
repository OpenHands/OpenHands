import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import {
  useCreateModelProvider,
  useUpdateModelProvider,
} from "#/hooks/mutation/use-model-provider-mutations";
import type { ModelProvider, WireApi } from "#/api/model-providers-service";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";

interface ProviderFormProps {
  isOpen: boolean;
  onClose: () => void;
  /** The provider kind to create (e.g. "openai"); ignored when editing. */
  kind?: string;
  /** Human label for the kind, used in the title when creating. */
  kindLabel?: string;
  /** When set, the form edits this provider instead of creating a new one. */
  provider?: ModelProvider;
}

const WIRE_API_ITEMS = [
  { key: "auto", label: "auto" },
  { key: "chat", label: "completions" },
  { key: "responses", label: "responses" },
];

function parseCustomHeaders(value: string): Record<string, string> | null {
  const trimmed = value.trim();
  if (!trimmed) return {};
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (
      !parsed ||
      typeof parsed !== "object" ||
      Array.isArray(parsed) ||
      Object.values(parsed).some((v) => typeof v !== "string")
    ) {
      return null;
    }
    return parsed as Record<string, string>;
  } catch {
    return null;
  }
}

/**
 * Add / edit a model provider, implementing the issue #15492 wireframe form:
 * Display name, Base URL, Wire API, API key (stored as a named secret), and
 * optional custom headers. The key is held on the provider and shared by every
 * model under it; the server never returns it, so when editing we leave the key
 * field blank and only send it if the user types a new one.
 */
export function ProviderForm({
  isOpen,
  onClose,
  kind,
  kindLabel,
  provider,
}: ProviderFormProps) {
  const { t } = useTranslation("openhands");
  const createProvider = useCreateModelProvider();
  const updateProvider = useUpdateModelProvider();
  const isEditing = Boolean(provider);

  const [displayName, setDisplayName] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [wireApi, setWireApi] = useState<WireApi>("auto");
  const [key, setKey] = useState("");
  const [customHeadersJson, setCustomHeadersJson] = useState("");
  const nameInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!isOpen) return;
    if (provider) {
      setDisplayName(provider.displayName);
      setBaseUrl(provider.baseUrl ?? "");
      setWireApi(provider.wireApi);
      setCustomHeadersJson(
        Object.keys(provider.customHeaders).length > 0
          ? JSON.stringify(provider.customHeaders)
          : "",
      );
    } else {
      setDisplayName(kindLabel ?? "");
      setBaseUrl("");
      setWireApi("auto");
      setCustomHeadersJson("");
    }
    setKey("");
  }, [isOpen, provider, kindLabel]);

  const busy = createProvider.isPending || updateProvider.isPending;

  const handleSubmit = async () => {
    if (!displayName.trim()) {
      displayErrorToast(t(I18nKey.SETTINGS$PROVIDER_NAME_REQUIRED));
      return;
    }
    if (!isEditing && !key.trim()) {
      displayErrorToast(t(I18nKey.SETTINGS$PROVIDER_KEY_REQUIRED));
      return;
    }
    const customHeaders = parseCustomHeaders(customHeadersJson);
    if (customHeaders === null) {
      displayErrorToast(t(I18nKey.SETTINGS$CONNECTION_HEADERS_INVALID));
      return;
    }

    try {
      if (provider) {
        await updateProvider.mutateAsync({
          id: provider.id,
          request: {
            displayName: displayName.trim(),
            baseUrl: baseUrl.trim(),
            wireApi,
            customHeaders,
            // Only rotate the key when the user typed a new one.
            ...(key.trim() ? { key: key.trim() } : {}),
          },
        });
        displaySuccessToast(
          t(I18nKey.SETTINGS$PROVIDER_UPDATED, {
            provider: displayName.trim(),
          }),
        );
      } else {
        await createProvider.mutateAsync({
          kind: kind ?? "custom",
          displayName: displayName.trim(),
          key: key.trim(),
          baseUrl: baseUrl.trim() || undefined,
          wireApi,
          customHeaders,
        });
        displaySuccessToast(
          t(I18nKey.SETTINGS$PROVIDER_CREATED, {
            provider: displayName.trim(),
          }),
        );
      }
      onClose();
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
    }
  };

  const footer = (
    <>
      <BrandButton
        testId="provider-form-cancel"
        type="button"
        variant="secondary"
        onClick={onClose}
        isDisabled={busy}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
      <BrandButton
        testId="provider-form-submit"
        type="button"
        variant="primary"
        onClick={handleSubmit}
        isDisabled={busy}
        aria-busy={busy}
      >
        {busy
          ? t(I18nKey.STATUS$SAVING)
          : isEditing
            ? t(I18nKey.BUTTON$SAVE)
            : t(I18nKey.SETTINGS$CONNECT_PROVIDER)}
      </BrandButton>
    </>
  );

  const title = isEditing
    ? t(I18nKey.SETTINGS$EDIT_PROVIDER_TITLE, {
        provider: provider?.displayName ?? "",
      })
    : t(I18nKey.SETTINGS$ADD_PROVIDER_TITLE, {
        provider: kindLabel ?? kind ?? "",
      });

  return (
    <ApiKeyModalBase
      isOpen={isOpen}
      title={title}
      width="md"
      footer={footer}
      onClose={onClose}
      initialFocusRef={nameInputRef}
    >
      <div className="flex flex-col gap-4" data-testid="provider-form">
        <SettingsInput
          testId="provider-display-name"
          name="provider-display-name"
          ref={nameInputRef}
          label={t(I18nKey.SETTINGS$PROVIDER_DISPLAY_NAME_LABEL)}
          type="text"
          value={displayName}
          onChange={setDisplayName}
          showRequiredTag
        />
        <div className="flex flex-col gap-1">
          <SettingsInput
            testId="provider-base-url"
            name="provider-base-url"
            label={t(I18nKey.SETTINGS$BASE_URL)}
            type="text"
            value={baseUrl}
            placeholder={t(I18nKey.SETTINGS$CONNECTION_BASE_URL_PLACEHOLDER)}
            onChange={setBaseUrl}
            showOptionalTag
          />
          <p className="text-xs leading-4 text-tertiary-light">
            {t(I18nKey.SETTINGS$PROVIDER_BASE_URL_HELPER)}
          </p>
        </div>
        <div className="flex flex-col gap-1">
          <SettingsDropdownInput
            testId="provider-wire-api"
            name="provider-wire-api"
            label={t(I18nKey.SETTINGS$CONNECTION_WIRE_API_LABEL)}
            items={WIRE_API_ITEMS}
            selectedKey={wireApi}
            onSelectionChange={(k) => {
              const next = String(k || "auto");
              setWireApi(
                next === "chat" || next === "responses" ? next : "auto",
              );
            }}
          />
          <p className="text-xs leading-4 text-tertiary-light">
            {t(I18nKey.SETTINGS$PROVIDER_WIRE_API_HELPER)}
          </p>
        </div>
        <div className="flex flex-col gap-1">
          <SettingsInput
            testId="provider-api-key"
            name="provider-api-key"
            label={t(I18nKey.SETTINGS$CONNECTION_API_KEY_LABEL)}
            type="password"
            value={key}
            placeholder={
              isEditing
                ? t(I18nKey.SETTINGS$PROVIDER_KEY_UNCHANGED_PLACEHOLDER)
                : t(I18nKey.SETTINGS$CONNECTION_API_KEY_PLACEHOLDER)
            }
            onChange={setKey}
            showRequiredTag={!isEditing}
            showOptionalTag={isEditing}
          />
          <p className="text-xs leading-4 text-tertiary-light">
            {t(I18nKey.SETTINGS$PROVIDER_API_KEY_HELPER)}
          </p>
        </div>
        <div className="flex flex-col gap-1">
          <SettingsInput
            testId="provider-custom-headers"
            name="provider-custom-headers"
            label={t(I18nKey.SETTINGS$CONNECTION_CUSTOM_HEADERS_LABEL)}
            type="text"
            value={customHeadersJson}
            placeholder={t(
              I18nKey.SETTINGS$CONNECTION_CUSTOM_HEADERS_PLACEHOLDER,
            )}
            onChange={setCustomHeadersJson}
            showOptionalTag
          />
          <p className="text-xs leading-4 text-tertiary-light">
            {t(I18nKey.SETTINGS$CONNECTION_CUSTOM_HEADERS_HELPER)}
          </p>
        </div>
        {busy ? <LoadingSpinner size="small" /> : null}
      </div>
    </ApiKeyModalBase>
  );
}
