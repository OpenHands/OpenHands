import { useEffect, useMemo, useRef, useState } from "react";
import { Autocomplete, AutocompleteItem } from "@heroui/react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { KeyStatusIcon } from "#/components/features/settings/key-status-icon";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { OpenAISubscriptionAuthCard } from "#/components/features/settings/llm-settings/openai-subscription-auth-card";
import { CliSubscriptionAuthCard } from "#/components/features/settings/llm-profiles/cli-subscription-auth-card";
import { HelpLink } from "#/ui/help-link";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import { SecretsService } from "#/api/secrets-service";
import { useCreateProviderConnection } from "#/hooks/mutation/use-create-provider-connection";
import { useUpdateProviderConnection } from "#/hooks/mutation/use-update-provider-connection";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import { useActivateLlmProfile } from "#/hooks/mutation/use-activate-llm-profile";
import { useOpenAISubscriptionStatus } from "#/hooks/query/use-llm-subscription-status";
import { useOpenAISubscriptionModels } from "#/hooks/query/use-llm-subscription-models";
import { useAcpAuthStatus } from "#/hooks/query/use-acp-auth-status";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { profileNameFromDisplayName } from "#/utils/derive-profile-name";
import { linkedLlmProfileFromConnection } from "#/utils/linked-llm-profile-from-connection";
import { I18nKey } from "#/i18n/declaration";
import { mapProvider } from "#/utils/map-provider";
import { formControlSettingsFieldClassName } from "#/utils/form-control-classes";
import { heroUiAutocompleteSelectorButtonClassName } from "#/ui/combobox-caret";
import {
  LLM_AUTH_TYPE_SUBSCRIPTION,
  OPENAI_SUBSCRIPTION_VENDOR,
} from "#/constants/llm-subscription";
import {
  getProviderConnectionOption,
  isCliSubscriptionAuth,
  listProviderConnectionOptions,
  OLLAMA_DEFAULT_BASE_URL,
  OLLAMA_PLACEHOLDER_API_KEY,
  OLLAMA_PROVIDER_ID,
  PROVIDER_CONNECTION_AUTH_API_KEY,
  PROVIDER_CONNECTION_AUTH_ANTHROPIC_SUBSCRIPTION,
  PROVIDER_CONNECTION_AUTH_OPENAI_SUBSCRIPTION,
  resolveProviderConnectionFields,
  type ProviderConnectionAuth,
} from "#/constants/provider-connection-options";

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
 * Create / edit a provider. Options are the providers this form can actually
 * configure (API key, optional URL, or ChatGPT login) — not LiteLLM's full catalog.
 */
export function ProviderConnectionModal({
  connection,
  isCreate,
  onClose,
  onSaved,
}: ProviderConnectionModalProps) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const createConnection = useCreateProviderConnection();
  const updateConnection = useUpdateProviderConnection();
  const saveProfile = useSaveLlmProfile();
  const activateProfile = useActivateLlmProfile();
  const firstFieldRef = useRef<HTMLInputElement>(null);

  const [displayName, setDisplayName] = useState("");
  const [provider, setProvider] = useState<string | null>(null);
  const [auth, setAuth] = useState<ProviderConnectionAuth>(
    PROVIDER_CONNECTION_AUTH_API_KEY,
  );
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [oauthToken, setOauthToken] = useState("");

  const isOpen = isCreate || Boolean(connection);
  const keyAlreadySet = Boolean(connection?.api_key_set);
  const option = getProviderConnectionOption(provider);
  const isOpenAiSubscription =
    auth === PROVIDER_CONNECTION_AUTH_OPENAI_SUBSCRIPTION;
  const isCliAuth = isCliSubscriptionAuth(auth);
  const fields = resolveProviderConnectionFields(provider, auth);
  const { data: subscriptionStatus } = useOpenAISubscriptionStatus({
    enabled: isOpen && isOpenAiSubscription,
  });
  const { data: subscriptionModels } = useOpenAISubscriptionModels({
    enabled: isOpen && isOpenAiSubscription,
  });
  const cliProbe = useAcpAuthStatus(option?.cliProbeKey, {
    enabled: isOpen && isCliAuth && Boolean(option?.cliProbeKey),
  });

  useEffect(() => {
    setDisplayName(connection?.display_name ?? "");
    setProvider(connection?.provider ?? (isCreate ? null : DEFAULT_PROVIDER));
    setBaseUrl(connection?.base_url ?? "");
    setApiKey("");
    setOauthToken("");
    setAuth(PROVIDER_CONNECTION_AUTH_API_KEY);
  }, [connection, isCreate]);

  const isPending =
    createConnection.isPending ||
    updateConnection.isPending ||
    saveProfile.isPending ||
    activateProfile.isPending;
  const trimmedName = displayName.trim();
  const trimmedKey = apiKey.trim();
  const trimmedBaseUrl = baseUrl.trim();
  const trimmedOauth = oauthToken.trim();
  const keyOk =
    fields.apiKey !== "required" || !isCreate || Boolean(trimmedKey);
  const urlOk = fields.baseUrl !== "required" || Boolean(trimmedBaseUrl);
  const subscriptionOk =
    !isOpenAiSubscription || Boolean(subscriptionStatus?.connected);
  const cliOk =
    !isCliAuth ||
    cliProbe.status === "authenticated" ||
    Boolean(trimmedOauth) ||
    (fields.apiKey === "optional" && Boolean(trimmedKey));
  const nameOk = isCliAuth || Boolean(trimmedName);
  const isValid =
    nameOk &&
    Boolean(provider?.trim()) &&
    keyOk &&
    urlOk &&
    subscriptionOk &&
    cliOk;

  const providerOptions = useMemo(() => {
    const listed = listProviderConnectionOptions().map((candidate) => ({
      name: candidate.id,
    }));
    if (provider && !listed.some((candidate) => candidate.name === provider)) {
      listed.push({ name: provider });
    }
    return listed;
  }, [provider]);

  if (!isOpen) return null;

  const handleClose = () => {
    if (!isPending) onClose();
  };

  const handleProviderChange = (key: string | null) => {
    setProvider(key);
    const next = getProviderConnectionOption(key);
    if (!next) return;
    setAuth(next.authModes.includes(auth) ? auth : next.authModes[0]);
    if (key === OLLAMA_PROVIDER_ID && !baseUrl.trim()) {
      setBaseUrl(OLLAMA_DEFAULT_BASE_URL);
    }
  };

  const authItems = (option?.authModes ?? []).flatMap((mode) => {
    if (mode === PROVIDER_CONNECTION_AUTH_API_KEY) {
      return [
        {
          key: PROVIDER_CONNECTION_AUTH_API_KEY,
          label: t(I18nKey.SETTINGS$LLM_AUTH_TYPE_API_KEY),
        },
      ];
    }
    if (mode === PROVIDER_CONNECTION_AUTH_OPENAI_SUBSCRIPTION) {
      return [
        {
          key: PROVIDER_CONNECTION_AUTH_OPENAI_SUBSCRIPTION,
          label: t(I18nKey.SETTINGS$LLM_AUTH_TYPE_SUBSCRIPTION),
        },
      ];
    }
    if (mode === PROVIDER_CONNECTION_AUTH_ANTHROPIC_SUBSCRIPTION) {
      return [
        {
          key: PROVIDER_CONNECTION_AUTH_ANTHROPIC_SUBSCRIPTION,
          label: t(I18nKey.SETTINGS$LLM_AUTH_TYPE_CLAUDE_SUBSCRIPTION),
        },
      ];
    }
    return [];
  });

  const handleSubmit = async () => {
    if (!isValid || isPending) return;

    try {
      if (isCreate && isOpenAiSubscription) {
        const model = subscriptionModels?.[0];
        if (!model) {
          displayErrorToast(t(I18nKey.ERROR$GENERIC));
          return;
        }
        const name = profileNameFromDisplayName(trimmedName);
        await saveProfile.mutateAsync({
          name,
          request: {
            llm: {
              auth_type: LLM_AUTH_TYPE_SUBSCRIPTION,
              subscription_vendor: OPENAI_SUBSCRIPTION_VENDOR,
              model,
            },
          },
        });
        await activateProfile.mutateAsync(name);
        displaySuccessToast(t(I18nKey.SETTINGS$PROFILE_CREATED, { name }));
        onClose();
        return;
      }

      if (isCreate && isCliAuth) {
        const secretName = option?.oauthSecretName;
        const secretValue =
          trimmedOauth || (fields.apiKey === "optional" ? trimmedKey : "");
        if (secretName && secretValue) {
          await SecretsService.createSecret(secretName, secretValue);
        }
        await queryClient.invalidateQueries({
          queryKey: ["acp-auth-status"],
        });
        displaySuccessToast(t(I18nKey.SETTINGS$SUBSCRIPTION_CONNECTED_TOAST));
        onClose();
        return;
      }

      if (isCreate) {
        const created = await createConnection.mutateAsync({
          display_name: trimmedName,
          provider: provider?.trim() || DEFAULT_PROVIDER,
          api_key:
            trimmedKey ||
            (fields.apiKey === "optional" ? OLLAMA_PLACEHOLDER_API_KEY : ""),
          base_url: trimmedBaseUrl || null,
        });
        const linked = linkedLlmProfileFromConnection(created);
        await saveProfile.mutateAsync(linked);
        await activateProfile.mutateAsync(linked.name);
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
        {!isCliAuth ? (
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
        ) : null}
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
            onSelectionChange={(key) =>
              handleProviderChange(key?.toString() ?? null)
            }
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
            {providerOptions.map((candidate) => (
              <AutocompleteItem
                data-testid={`provider-item-${candidate.name}`}
                key={candidate.name}
                textValue={mapProvider(candidate.name)}
              >
                {mapProvider(candidate.name)}
              </AutocompleteItem>
            ))}
          </Autocomplete>
        </fieldset>
        {authItems.length > 1 && isCreate ? (
          <SettingsDropdownInput
            testId="provider-connection-auth-input"
            name="provider-connection-auth"
            label={t(I18nKey.SETTINGS$LLM_AUTH_TYPE)}
            items={authItems}
            selectedKey={auth}
            isClearable={false}
            required
            onSelectionChange={(selectedKey) => {
              if (
                selectedKey === PROVIDER_CONNECTION_AUTH_OPENAI_SUBSCRIPTION
              ) {
                setAuth(PROVIDER_CONNECTION_AUTH_OPENAI_SUBSCRIPTION);
                return;
              }
              if (
                selectedKey === PROVIDER_CONNECTION_AUTH_ANTHROPIC_SUBSCRIPTION
              ) {
                setAuth(PROVIDER_CONNECTION_AUTH_ANTHROPIC_SUBSCRIPTION);
                return;
              }
              setAuth(PROVIDER_CONNECTION_AUTH_API_KEY);
            }}
          />
        ) : null}
        {isOpenAiSubscription ? <OpenAISubscriptionAuthCard /> : null}
        {isCliAuth && option?.cliProbeKey && option.loginCommand ? (
          <CliSubscriptionAuthCard
            probeKey={option.cliProbeKey}
            providerName={mapProvider(option.id)}
            loginCommand={option.loginCommand}
          />
        ) : null}
        {isCliAuth &&
        auth === PROVIDER_CONNECTION_AUTH_ANTHROPIC_SUBSCRIPTION ? (
          <SettingsInput
            testId="provider-connection-oauth-input"
            label={t(I18nKey.SETTINGS$CLI_SUBSCRIPTION_OAUTH_TOKEN)}
            type="password"
            className="w-full"
            value={oauthToken}
            onChange={setOauthToken}
            showOptionalTag
            hint={t(I18nKey.ONBOARDING$ACP_SECRET_OAUTH_TOKEN_HINT)}
          />
        ) : null}
        {fields.apiKey !== "hidden" ? (
          <SettingsInput
            testId="provider-connection-api-key-input"
            label={t(I18nKey.SETTINGS_FORM$API_KEY)}
            type="password"
            className="w-full"
            value={apiKey}
            // eslint-disable-next-line i18next/no-literal-string -- masked-key sentinel, not translatable
            placeholder={keyAlreadySet ? "<hidden>" : ""}
            onChange={setApiKey}
            showOptionalTag={fields.apiKey === "optional"}
            required={fields.apiKey === "required" && isCreate}
            startContent={
              keyAlreadySet ? (
                <KeyStatusIcon isSet={keyAlreadySet} />
              ) : undefined
            }
            hint={
              keyAlreadySet
                ? t(I18nKey.SETTINGS$PROVIDER_CONNECTION_ROTATE_HINT)
                : undefined
            }
          />
        ) : null}
        {fields.showOpenHandsHelp ? (
          <HelpLink
            testId="provider-connection-openhands-help"
            text={t(I18nKey.SETTINGS$OPENHANDS_API_KEY_HELP_TEXT)}
            linkText={t(I18nKey.SETTINGS$OPENHANDS_API_KEY_HELP_LINK)}
            href="https://app.all-hands.dev/settings/api-keys"
            suffix={` ${t(I18nKey.SETTINGS$OPENHANDS_API_KEY_HELP_SUFFIX)}`}
            suffixLinkText={t(I18nKey.SETTINGS$SEE_HERE_FOR_MORE_DETAILS)}
            suffixLinkHref="https://docs.openhands.dev/usage/local-setup#getting-an-api-key"
            trailing="."
          />
        ) : null}
        {fields.baseUrl !== "hidden" ? (
          <SettingsInput
            testId="provider-connection-base-url-input"
            label={t(I18nKey.SETTINGS$BASE_URL)}
            type="text"
            className="w-full"
            value={baseUrl}
            placeholder={
              provider === OLLAMA_PROVIDER_ID
                ? OLLAMA_DEFAULT_BASE_URL
                : undefined
            }
            hint={
              provider === OLLAMA_PROVIDER_ID
                ? t(I18nKey.SCHEMA$LLM$OLLAMA_BASE_URL$DESCRIPTION)
                : undefined
            }
            onChange={setBaseUrl}
            showOptionalTag={fields.baseUrl === "optional"}
            required={fields.baseUrl === "required"}
          />
        ) : null}
      </div>
    </ApiKeyModalBase>
  );
}
