import React from "react";
import { useSearchParams } from "react-router";
import { useTranslation } from "react-i18next";
import { FaChevronLeft } from "react-icons/fa6";
import { ModelSelector } from "#/components/shared/modals/settings/model-selector";
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { requireOrgDefaultsRedirect } from "#/utils/org/saas-redirect-to-org-defaults-guard";
import { useAgentSettingsSchema } from "#/hooks/query/use-agent-settings-schema";
import { useSettings } from "#/hooks/query/use-settings";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { HelpLink } from "#/ui/help-link";
import { useConfig } from "#/hooks/query/use-config";
import { KeyStatusIcon } from "#/components/features/settings/key-status-icon";
import { OpenHandsApiKeyHelp } from "#/components/features/settings/openhands-api-key-help";
import {
  SdkSectionHeaderProps,
  SdkSectionPage,
} from "#/components/features/settings/sdk-settings/sdk-section-page";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { Settings, SettingsSchema, SettingsScope } from "#/types/settings";
import { extractModelAndProvider } from "#/utils/extract-model-and-provider";
import {
  inferInitialView,
  type SettingsView,
} from "#/utils/sdk-settings-schema";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import { useActivateLlmProfile } from "#/hooks/mutation/use-activate-llm-profile";
import { useRenameLlmProfile } from "#/hooks/mutation/use-rename-llm-profile";
import {
  deriveProfileNameFromModel,
  PROFILE_NAME_PATTERN,
} from "#/utils/derive-profile-name";
import { LlmProfilesManager } from "#/components/features/settings/llm-profiles-manager";
import { ProfileNameInput } from "#/components/features/settings/profile-name-input";
import { Typography } from "#/ui/typography";

const LLM_EXCLUDED_KEYS = new Set(["llm.model", "llm.api_key", "llm.base_url"]);

const buildModelId = (provider: string | null, model: string | null) => {
  if (!provider || !model) return null;
  return `${provider}/${model}`;
};

const getSchemaFieldDefaultValue = (
  schema: SettingsSchema | null | undefined,
  fieldKey: string,
) =>
  schema?.sections
    .flatMap((section) => section.fields)
    .find((field) => field.key === fieldKey)?.default ?? null;

const KNOWN_PROVIDER_DEFAULT_BASE_URLS: Partial<Record<string, Set<string>>> = {
  openai: new Set(["https://api.openai.com", "https://api.openai.com/v1"]),
  openhands: new Set([
    "https://llm-proxy.app.all-hands.dev",
    "https://llm-proxy.app.all-hands.dev/v1",
  ]),
  litellm_proxy: new Set([
    "https://llm-proxy.app.all-hands.dev",
    "https://llm-proxy.app.all-hands.dev/v1",
  ]),
};

const normalizeBaseUrl = (baseUrl: string) => {
  try {
    const parsedUrl = new URL(baseUrl);
    const normalizedPath = parsedUrl.pathname.replace(/\/+$/, "") || "";
    return `${parsedUrl.origin}${normalizedPath}`;
  } catch {
    return baseUrl.trim().replace(/\/+$/, "");
  }
};

const isProviderDefaultBaseUrl = (model: string, baseUrl: string) => {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl);
  const { provider } = extractModelAndProvider(model);

  if (provider) {
    const knownDefaults = KNOWN_PROVIDER_DEFAULT_BASE_URLS[provider];
    if (knownDefaults) {
      return knownDefaults.has(normalizedBaseUrl);
    }
  }

  return Object.values(KNOWN_PROVIDER_DEFAULT_BASE_URLS).some((knownDefaults) =>
    knownDefaults?.has(normalizedBaseUrl),
  );
};

export function LlmSettingsScreen({
  scope = "personal",
}: {
  scope?: SettingsScope;
}) {
  const { t } = useTranslation();
  const [searchParams, setSearchParams] = useSearchParams();

  const { data: settings } = useSettings(scope);
  const { data: schema } = useAgentSettingsSchema(
    settings?.agent_settings_schema,
  );
  const { data: config } = useConfig();

  const [selectedProvider, setSelectedProvider] = React.useState<string | null>(
    null,
  );
  const hasHydratedInitialPersonalSaasViewRef = React.useRef(false);
  // Captured during buildPayload so onSaveSuccess can derive a profile name
  // from the exact model that was just persisted.
  const lastSavedModelRef = React.useRef<string | null>(null);
  const saveProfile = useSaveLlmProfile();
  const activateProfile = useActivateLlmProfile();
  const renameProfile = useRenameLlmProfile();

  // Controls whether the LLM form or the Profiles list is shown. Flipping
  // this unmounts the inactive branch, so the SdkSectionPage re-hydrates
  // its view from ``initialViewHint`` when coming back from profiles.
  // Personal scope lands on the Available Models list first; org-scope
  // defaults (which don't have profiles) always open straight into the form.
  const [showProfiles, setShowProfiles] = React.useState(scope === "personal");
  // User-supplied profile name. Empty → fall back to deriveProfileNameFromModel
  // in handleSaveSuccess. Reset on every form open so a stale name from the
  // previous Add doesn't leak in.
  const [profileName, setProfileName] = React.useState("");
  // Snapshotted on form open so we can flag the form dirty when the user
  // edits *only* the name — the SDK section page tracks the LLM fields but
  // not the profile-name input that lives outside its schema.
  const [initialProfileName, setInitialProfileName] = React.useState("");
  // When the user clicks Basic / Advanced / All from inside the profiles
  // view, we want the LLM form to open on *that* tier — not whatever the
  // schema happened to infer. We stash the choice here and consume it in
  // getInitialView below.
  const [initialViewHint, setInitialViewHint] =
    React.useState<SettingsView | null>(null);
  const isProfilesView = scope === "personal" && showProfiles;

  const defaultModel = String(
    (DEFAULT_SETTINGS.agent_settings?.llm as Record<string, unknown>)?.model ??
      "",
  );

  const isSaasMode = config?.app_mode === "saas";

  React.useEffect(() => {
    if (settings?.llm_model) {
      const { provider } = extractModelAndProvider(settings.llm_model);
      setSelectedProvider(provider || null);
    }
  }, [settings?.llm_model]);

  React.useEffect(() => {
    const checkout = searchParams.get("checkout");

    if (checkout === "success") {
      displaySuccessToast(t(I18nKey.SUBSCRIPTION$SUCCESS));
      setSearchParams({});
    } else if (checkout === "cancel") {
      displayErrorToast(t(I18nKey.SUBSCRIPTION$FAILURE));
      setSearchParams({});
    }
  }, [searchParams, setSearchParams, t]);

  const infoMessageKey = React.useMemo((): I18nKey | null => {
    if (!isSaasMode) return null;
    return scope === "org"
      ? I18nKey.SETTINGS$ORG_DEFAULTS_INFO
      : I18nKey.SETTINGS$PERSONAL_AGENT_INFO;
  }, [isSaasMode, scope]);

  const getInitialView = React.useCallback(
    (
      currentSettings: Settings,
      filteredSchema: SettingsSchema,
    ): SettingsView => {
      // A hint set by the Profiles mirror-strip beats every other rule —
      // the user explicitly asked for this tier when leaving profiles.
      if (initialViewHint) {
        return initialViewHint;
      }

      // Personal SaaS users now land on Available Models first; the form
      // is mounted on-demand (Add / Edit). The first form mount per session
      // should still default to basic so users aren't dropped straight into
      // advanced/all even if the active profile has complex fields.
      if (
        isSaasMode &&
        scope !== "org" &&
        !hasHydratedInitialPersonalSaasViewRef.current
      ) {
        hasHydratedInitialPersonalSaasViewRef.current = true;
        return "basic";
      }

      const schemaView = inferInitialView(currentSettings, filteredSchema);
      if (schemaView !== "basic") {
        return schemaView;
      }

      const currentModel = currentSettings.llm_model ?? "";
      const trimmedBaseUrl = currentSettings.llm_base_url?.trim() ?? "";
      const hasCustomBaseUrl =
        trimmedBaseUrl.length > 0 &&
        !isProviderDefaultBaseUrl(currentModel, trimmedBaseUrl);

      return hasCustomBaseUrl ? "all" : "basic";
    },
    [initialViewHint, isSaasMode, scope],
  );

  const buildHeader = React.useCallback(
    ({ values, isDisabled, view, onChange }: SdkSectionHeaderProps) => {
      const modelValue =
        typeof values["llm.model"] === "string" ? values["llm.model"] : "";
      const baseUrlValue =
        typeof values["llm.base_url"] === "string"
          ? values["llm.base_url"]
          : "";
      const derivedProvider = modelValue
        ? extractModelAndProvider(modelValue).provider || null
        : null;
      const activeProvider =
        view === "basic"
          ? (selectedProvider ?? derivedProvider)
          : derivedProvider;
      const shouldUseOpenHandsKey =
        isSaasMode && activeProvider === "openhands";
      const showOpenHandsApiKeyHelp = modelValue.startsWith("openhands/");

      const renderApiKeyInput = (testId: string, helpTestId: string) => {
        if (shouldUseOpenHandsKey) {
          return null;
        }

        return (
          <>
            <SettingsInput
              testId={testId}
              label={t(I18nKey.SETTINGS_FORM$API_KEY)}
              type="password"
              className="w-full"
              value={
                typeof values["llm.api_key"] === "string"
                  ? values["llm.api_key"]
                  : ""
              }
              placeholder={settings?.llm_api_key_set ? "<hidden>" : ""}
              onChange={(value) => onChange("llm.api_key", value)}
              isDisabled={isDisabled}
              startContent={
                settings?.llm_api_key_set ? (
                  <KeyStatusIcon isSet={settings.llm_api_key_set} />
                ) : undefined
              }
            />

            <HelpLink
              testId={helpTestId}
              text={t(I18nKey.SETTINGS$DONT_KNOW_API_KEY)}
              linkText={t(I18nKey.SETTINGS$CLICK_FOR_INSTRUCTIONS)}
              href="https://docs.openhands.dev/usage/local-setup#getting-an-api-key"
            />
          </>
        );
      };

      const profileNamePlaceholder =
        deriveProfileNameFromModel(modelValue) ?? "";

      return (
        <div className="flex flex-col gap-6">
          {infoMessageKey ? (
            <Typography.Paragraph
              testId="llm-settings-info-message"
              className="text-sm text-tertiary-alt"
            >
              {t(infoMessageKey)}
            </Typography.Paragraph>
          ) : null}

          {scope === "personal" ? (
            <ProfileNameInput
              testId="llm-profile-name-input"
              ruleTestId="llm-profile-name-rule"
              value={profileName}
              placeholder={profileNamePlaceholder}
              onChange={setProfileName}
              isDisabled={isDisabled}
              isOptional
            />
          ) : null}

          {view === "basic" ? (
            <div
              className="flex flex-col gap-6"
              data-testid="llm-settings-form-basic"
            >
              <ModelSelector
                currentModel={modelValue || undefined}
                onChange={(provider, model) => {
                  setSelectedProvider(provider);
                  const nextModel = buildModelId(provider, model);
                  if (nextModel) {
                    onChange("llm.model", nextModel);
                  }
                }}
                wrapperClassName="!flex-col !gap-6"
                isDisabled={isDisabled}
              />

              {showOpenHandsApiKeyHelp ? (
                <OpenHandsApiKeyHelp testId="openhands-api-key-help" />
              ) : null}

              {renderApiKeyInput(
                "llm-api-key-input",
                "llm-api-key-help-anchor",
              )}
            </div>
          ) : (
            <div
              className="flex flex-col gap-6"
              data-testid="llm-settings-form-advanced"
            >
              <SettingsInput
                testId="llm-custom-model-input"
                label={t(I18nKey.SETTINGS$CUSTOM_MODEL)}
                type="text"
                className="w-full"
                value={modelValue}
                placeholder={defaultModel}
                onChange={(value) => onChange("llm.model", value)}
                isDisabled={isDisabled}
              />

              {showOpenHandsApiKeyHelp ? (
                <OpenHandsApiKeyHelp testId="openhands-api-key-help-2" />
              ) : null}

              <SettingsInput
                testId="base-url-input"
                label={t(I18nKey.SETTINGS$BASE_URL)}
                type="text"
                className="w-full"
                value={baseUrlValue}
                placeholder="https://api.openai.com"
                onChange={(value) => onChange("llm.base_url", value)}
                isDisabled={isDisabled}
              />

              {renderApiKeyInput(
                "llm-api-key-input",
                "llm-api-key-help-anchor-advanced",
              )}
            </div>
          )}
        </div>
      );
    },
    [
      infoMessageKey,
      isSaasMode,
      defaultModel,
      profileName,
      scope,
      selectedProvider,
      settings?.llm_api_key_set,
      t,
    ],
  );

  const buildPayload = React.useCallback(
    (
      basePayload: Record<string, unknown>,
      context: {
        values: Record<string, string | boolean>;
        view: SettingsView;
      },
    ) => {
      // basePayload is a nested dict (e.g. {llm: {model: "gpt-4"}})
      const agentSettings = structuredClone(basePayload);

      const modelValue =
        typeof context.values["llm.model"] === "string"
          ? context.values["llm.model"]
          : "";
      const derivedProvider = modelValue
        ? extractModelAndProvider(modelValue).provider || null
        : null;
      const activeProvider =
        context.view === "basic"
          ? (selectedProvider ?? derivedProvider)
          : derivedProvider;
      const shouldUseOpenHandsKey =
        isSaasMode && activeProvider === "openhands";

      const llm = (agentSettings.llm ?? {}) as Record<string, unknown>;
      if (shouldUseOpenHandsKey && llm.model !== undefined) {
        llm.api_key = "";
        agentSettings.llm = llm;
      }

      if (context.view === "basic") {
        llm.base_url = getSchemaFieldDefaultValue(schema, "llm.base_url");
        agentSettings.llm = llm;
      }

      // Remember the model currently shown in the form — this is what the
      // user is saving regardless of whether `llm.model` was toggled dirty
      // this turn. ``basePayload`` only includes dirty fields, so falling
      // back to ``context.values`` makes the profile auto-creation fire on
      // same-value re-saves (e.g. save → delete profile → save again).
      lastSavedModelRef.current = modelValue || null;

      return { agent_settings_diff: agentSettings };
    },
    [isSaasMode, schema, selectedProvider],
  );

  const handleSaveSuccess = React.useCallback(async () => {
    const savedModel = lastSavedModelRef.current;
    const trimmedUserName = profileName.trim();
    // Use the user-supplied name only if it matches the backend regex —
    // otherwise silently fall back to the model-derived default (the helper
    // text under the input has already warned them their name was invalid).
    const userName = PROFILE_NAME_PATTERN.test(trimmedUserName)
      ? trimmedUserName
      : null;
    const derivedName = savedModel
      ? deriveProfileNameFromModel(savedModel)
      : null;
    const name = userName ?? derivedName;

    // Auto-saved profiles are a personal-scope feature — organization default
    // LLM settings reuse this screen but shouldn't spawn per-user profiles.
    if (scope !== "org" && name) {
      try {
        // Editing an existing profile and renaming it via the form should
        // rename the record in place rather than spawning a new one and
        // leaving the original orphaned.
        if (initialProfileName && initialProfileName !== name) {
          await renameProfile.mutateAsync({
            name: initialProfileName,
            newName: name,
          });
        }
        // Omit `llm` → backend snapshots the just-saved agent_settings.llm
        // (api_key and all). Saves us from having to hand-reconstruct the
        // config and risk mangling the secret placeholder handling.
        await saveProfile.mutateAsync({
          name,
          request: { include_secrets: true },
        });
        await activateProfile.mutateAsync(name);
      } catch {
        // Best-effort: the settings save already succeeded. Profile cap
        // (HTTP 409) and transient errors are surfaced on the Profiles page.
      }
    }

    setProfileName("");
    setInitialProfileName("");
    setInitialViewHint(null);
    setShowProfiles(true);
  }, [
    activateProfile,
    initialProfileName,
    profileName,
    renameProfile,
    saveProfile,
    scope,
  ]);

  const openForm = (view: SettingsView | null, name = "") => {
    setProfileName(name);
    setInitialProfileName(name);
    setInitialViewHint(view);
    setShowProfiles(false);
  };

  if (isProfilesView) {
    return (
      <LlmProfilesManager
        onAddProfile={() => openForm(null)}
        onEditProfile={(profile) => openForm(null, profile.name)}
      />
    );
  }

  // Sub-page back affordance (personal scope only — org-defaults has no
  // parent list to return to). Replaces the previous "Profiles" trailing
  // action so the form view follows the second-level settings pattern.
  const backToProfiles =
    scope === "personal" ? (
      <button
        data-testid="llm-back-to-profiles"
        type="button"
        onClick={() => {
          setInitialViewHint(null);
          setShowProfiles(true);
        }}
        className="flex items-center gap-2 self-start text-sm text-gray-300 hover:text-white cursor-pointer"
      >
        <FaChevronLeft size={12} aria-hidden="true" />
        {t(I18nKey.SETTINGS$BACK_TO_LLM_LIST)}
      </button>
    ) : null;

  return (
    <div className="flex flex-col gap-4">
      {backToProfiles}
      <SdkSectionPage
        scope={scope}
        sectionKeys={["llm"]}
        excludeKeys={LLM_EXCLUDED_KEYS}
        header={buildHeader}
        buildPayload={buildPayload}
        extraDirty={profileName.trim() !== initialProfileName.trim()}
        onSaveSuccess={handleSaveSuccess}
        getInitialView={getInitialView}
        forceShowAdvancedView
        allowAllView={!isSaasMode}
        testId="llm-settings-screen"
      />
    </div>
  );
}

const orgDefaultsRedirectGuard = requireOrgDefaultsRedirect(
  "/settings/org-defaults",
);
const llmPermissionGuard = createPermissionGuard("view_llm_settings");

export const clientLoader = async (args: { request: Request }) => {
  const blocked = await orgDefaultsRedirectGuard(args);
  if (blocked) return blocked;
  return llmPermissionGuard(args);
};

export default LlmSettingsScreen;
