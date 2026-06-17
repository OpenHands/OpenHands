import React from "react";
import { useQueryClient } from "@tanstack/react-query";
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
import { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import { useActivateLlmProfile } from "#/hooks/mutation/use-activate-llm-profile";
import { useRenameLlmProfile } from "#/hooks/mutation/use-rename-llm-profile";
import {
  useSaveOrgLlmProfile,
  useActivateOrgLlmProfile,
  useRenameOrgLlmProfile,
} from "#/hooks/mutation/use-org-llm-profile-mutations";
import {
  deriveProfileNameFromModel,
  PROFILE_NAME_PATTERN,
} from "#/utils/derive-profile-name";
import { LlmProfilesManager } from "#/components/features/settings/llm-profiles-manager";
import { OrgLlmProfilesManager } from "#/components/features/settings/org-llm-profiles-manager";
import { ProfileNameInput } from "#/components/features/settings/profile-name-input";
import { Typography } from "#/ui/typography";
import { providerModelsQueryOptions } from "#/hooks/query/use-provider-models";
import { useOrgTypeAndAccess } from "#/hooks/use-org-type-and-access";
import { useAppMode } from "#/hooks/use-app-mode";
import { useMe } from "#/hooks/query/use-me";
import { usePermission } from "#/hooks/organizations/use-permissions";

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
  // For openhands/* models, base_url is server-owned (auto-filled on save) —
  // never treat it as user customization that warrants the advanced view.
  // Accepted edge case: a deliberately customized base_url on an openhands/*
  // model also lands on basic; values are preserved, Advanced is a click away.
  if (model.startsWith("openhands/")) {
    return true;
  }

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

type ProfileFormMode = "create" | "edit";

export function LlmSettingsScreen({
  scope = "personal",
}: {
  scope?: SettingsScope;
}) {
  const { t } = useTranslation();
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();

  const { data: settings } = useSettings(scope);
  const { data: schema } = useAgentSettingsSchema(
    settings?.agent_settings_schema,
  );
  const { data: config } = useConfig();
  const { organizationId } = useOrgTypeAndAccess();
  const { data: me } = useMe();
  const { hasPermission } = usePermission(me?.role ?? "member");

  const [selectedProvider, setSelectedProvider] = React.useState<string | null>(
    null,
  );
  const hasHydratedInitialPersonalSaasViewRef = React.useRef(false);
  // Captured during buildPayload so onSaveSuccess can derive a profile name
  // from the exact model that was just persisted.
  const lastSavedModelRef = React.useRef<string | null>(null);
  // Whether the user typed a new API key this save — drives the backend's
  // preserve_existing_api_key so a blank key field keeps the profile's key.
  const lastSavedApiKeyTypedRef = React.useRef(false);

  // Personal profile hooks (for OSS mode)
  const saveProfile = useSaveLlmProfile();
  const activateProfile = useActivateLlmProfile();
  const renameProfile = useRenameLlmProfile();

  // Org profile hooks (for org defaults)
  const saveOrgProfile = useSaveOrgLlmProfile(organizationId);
  const activateOrgProfile = useActivateOrgLlmProfile(organizationId);
  const renameOrgProfile = useRenameOrgLlmProfile(organizationId);

  // Controls whether the LLM form or the Profiles list is shown. Flipping
  // this unmounts the inactive branch, so the SdkSectionPage re-hydrates
  // its view from ``initialViewHint`` when coming back from profiles.
  // Enable profiles for personal settings and org defaults. Org members can
  // view org profiles, but only admins/owners can create or manage them.
  const shouldShowProfilesForScope = scope === "personal" || scope === "org";
  const canManageProfilesForScope =
    scope === "personal" || hasPermission("edit_llm_settings");
  const [showProfiles, setShowProfiles] = React.useState(
    shouldShowProfilesForScope,
  );
  // User-supplied profile name. Empty → fall back to deriveProfileNameFromModel
  // in handleSaveSuccess. Reset on every form open so a stale name from the
  // previous Add doesn't leak in.
  const [profileName, setProfileName] = React.useState("");
  // Distinguishes Add Profile (blank create form, opens on basic) from Edit
  // (form hydrated from the clicked profile). Null when no form is open.
  const [profileFormMode, setProfileFormMode] =
    React.useState<ProfileFormMode | null>(null);
  // The profile summary the edit form was opened with — the source of truth
  // for the form's initial model/base_url instead of the active settings.
  const [editingProfile, setEditingProfile] =
    React.useState<LlmProfileSummary | null>(null);
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

  const isProfilesView = shouldShowProfilesForScope && showProfiles;
  const isOrgProfileMode = scope === "org";

  const defaultModel = String(
    (DEFAULT_SETTINGS.agent_settings?.llm as Record<string, unknown>)?.model ??
      "",
  );

  const isSaasMode = config?.app_mode === "saas";
  const { isEnterpriseCloud } = useAppMode();

  // OHE installs can disable BYOK via OH_ALLOW_USER_LLM_CONFIGURATION; absent
  // (SaaS / existing installs) means allowed. This hides the *editing* UI for
  // custom model / base URL / API key only — saved BYOK settings keep working
  // at runtime, and the managed model dropdown stays fully functional.
  const allowUserLlmConfiguration =
    config?.feature_flags?.allow_user_llm_configuration !== false;

  React.useEffect(() => {
    // An open profile form owns the provider selection (blank for create,
    // profile-derived for edit) — don't let the active settings override it.
    if (profileFormMode !== null && !showProfiles) {
      return;
    }
    if (settings?.llm_model) {
      const { provider } = extractModelAndProvider(settings.llm_model);
      setSelectedProvider(provider || null);
    }
  }, [profileFormMode, settings?.llm_model, showProfiles]);

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
      // Without BYOK there is nothing for the advanced/all tiers to reveal in
      // the custom header, so always land on the managed model dropdown.
      if (!allowUserLlmConfiguration) {
        return "basic";
      }

      // A hint set by the Profiles mirror-strip beats every other rule —
      // the user explicitly asked for this tier when leaving profiles.
      if (initialViewHint) {
        return initialViewHint;
      }

      // Edit opens on the tier the clicked profile needs — its values, not
      // the active settings, are what the form is hydrated with.
      if (profileFormMode === "edit" && editingProfile) {
        const profileModel = editingProfile.model ?? "";
        const profileBaseUrl = editingProfile.base_url?.trim() ?? "";
        const hasCustomProfileBaseUrl =
          profileBaseUrl.length > 0 &&
          !isProviderDefaultBaseUrl(profileModel, profileBaseUrl);
        return hasCustomProfileBaseUrl ? "advanced" : "basic";
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

      // Create always starts on basic — the inference below reads the
      // *active* settings and would otherwise open advanced for users
      // whose current config has a custom model or base URL.
      if (profileFormMode === "create") {
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
    [
      allowUserLlmConfiguration,
      editingProfile,
      initialViewHint,
      isSaasMode,
      profileFormMode,
      scope,
    ],
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
      // The OpenHands key help links to OpenHands Cloud keys/pricing — only
      // meaningful on enterprise cloud, not self-hosted managed installs.
      const showOpenHandsApiKeyHelp =
        isEnterpriseCloud &&
        modelValue.startsWith("openhands/") &&
        allowUserLlmConfiguration;
      // While editing, the set-but-unfetchable key indicator must reflect
      // the clicked profile, not whichever profile is currently active.
      const apiKeySet =
        profileFormMode === "edit" && editingProfile
          ? editingProfile.api_key_set
          : settings?.llm_api_key_set;

      const renderApiKeyInput = (testId: string, helpTestId: string) => {
        // Managed installs: the admin owns provider keys on the bundled
        // proxy; users never enter one.
        if (!allowUserLlmConfiguration) {
          return null;
        }

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
              placeholder={apiKeySet ? "<hidden>" : ""}
              onChange={(value) => onChange("llm.api_key", value)}
              isDisabled={isDisabled}
              startContent={
                apiKeySet ? <KeyStatusIcon isSet={apiKeySet} /> : undefined
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

          {canManageProfilesForScope ? (
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

          {view === "basic" || !allowUserLlmConfiguration ? (
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

              {/* A blank create form has no provider yet — rendering the key
                  input only to remove it when a managed provider is picked is
                  jarring, so wait until a provider is actually selected. */}
              {activeProvider
                ? renderApiKeyInput(
                    "llm-api-key-input",
                    "llm-api-key-help-anchor",
                  )
                : null}
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
      allowUserLlmConfiguration,
      editingProfile,
      infoMessageKey,
      isSaasMode,
      defaultModel,
      profileFormMode,
      profileName,
      scope,
      selectedProvider,
      settings?.llm_api_key_set,
      canManageProfilesForScope,
      t,
    ],
  );

  const buildPayload = React.useCallback(
    (
      defaultPayload: Record<string, unknown>,
      context: {
        values: Record<string, string | boolean>;
        view: SettingsView;
      },
    ) => {
      // defaultPayload is the wrapped diff (e.g.
      // `{ agent_settings_diff: { llm: { model: "gpt-4" } } }`); we only
      // mutate the inner `llm` object below.
      const agentSettings = structuredClone(
        (defaultPayload.agent_settings_diff as Record<string, unknown>) ?? {},
      );

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

      if (context.view === "basic" && llm.model !== undefined) {
        llm.base_url = getSchemaFieldDefaultValue(schema, "llm.base_url");
        agentSettings.llm = llm;
      }

      // Edit hydrates from the profile without marking fields dirty, so a
      // diff-only save would snapshot the *active* settings over the profile.
      // Persist the form's model/base_url explicitly when they aren't dirty.
      if (profileFormMode === "edit") {
        if (llm.model === undefined && modelValue) {
          llm.model = modelValue;
        }
        if (llm.base_url === undefined) {
          const baseUrlValue =
            typeof context.values["llm.base_url"] === "string"
              ? context.values["llm.base_url"].trim()
              : "";
          llm.base_url = baseUrlValue || null;
        }
        if (shouldUseOpenHandsKey && llm.api_key === undefined) {
          llm.api_key = "";
        }
        agentSettings.llm = llm;
      }

      // Remember the model currently shown in the form — this is what the
      // user is saving regardless of whether `llm.model` was toggled dirty
      // this turn. ``defaultPayload`` only includes dirty fields, so
      // falling back to ``context.values`` makes the profile auto-creation
      // fire on same-value re-saves (e.g. save → delete profile → save
      // again).
      lastSavedModelRef.current = modelValue || null;
      // OpenHands-managed saves force api_key to "" above — that's not a
      // user-typed key, so it must not defeat key preservation.
      const typedApiKey =
        typeof context.values["llm.api_key"] === "string"
          ? context.values["llm.api_key"].trim()
          : "";
      lastSavedApiKeyTypedRef.current =
        !shouldUseOpenHandsKey && typedApiKey.length > 0;

      return { agent_settings_diff: agentSettings };
    },
    [isSaasMode, profileFormMode, schema, scope, selectedProvider],
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

    const shouldSaveProfile =
      canManageProfilesForScope &&
      (scope === "personal" || (scope === "org" && organizationId)) &&
      name;

    if (shouldSaveProfile) {
      try {
        const useOrgHooks = scope === "org";

        // Editing an existing profile and renaming it via the form should
        // rename the record in place rather than spawning a new one and
        // leaving the original orphaned.
        if (initialProfileName && initialProfileName !== name) {
          if (useOrgHooks) {
            await renameOrgProfile.mutateAsync({
              name: initialProfileName,
              newName: name,
            });
          } else {
            await renameProfile.mutateAsync({
              name: initialProfileName,
              newName: name,
            });
          }
        }
        // Omit `llm` → backend snapshots the just-saved agent_settings.llm.
        // When the user typed no key, the snapshot would carry the *active*
        // settings' key — preserve_existing_api_key keeps the profile's own.
        const request = {
          include_secrets: true,
          preserve_existing_api_key: !lastSavedApiKeyTypedRef.current,
        };
        if (useOrgHooks) {
          await saveOrgProfile.mutateAsync({ name, request });
          await activateOrgProfile.mutateAsync(name);
        } else {
          await saveProfile.mutateAsync({ name, request });
          await activateProfile.mutateAsync(name);
        }
      } catch {
        // Best-effort: the settings save already succeeded. Profile cap
        // (HTTP 409) and transient errors are surfaced on the Profiles page.
      }
    }

    setProfileName("");
    setInitialProfileName("");
    setInitialViewHint(null);
    setProfileFormMode(null);
    setEditingProfile(null);
    setShowProfiles(true);
  }, [
    activateProfile,
    activateOrgProfile,
    canManageProfilesForScope,
    initialProfileName,
    organizationId,
    profileName,
    renameProfile,
    renameOrgProfile,
    saveProfile,
    saveOrgProfile,
    scope,
  ]);

  // Legacy profiles can store a hidden catalog alias (e.g. a pre-rename
  // proxy route). When the alias maps to a visible model, hydrate the edit
  // form with that canonical id so the dropdown shows a real selection.
  const resolveCanonicalModel = async (model: string): Promise<string> => {
    const { provider, model: modelName } = extractModelAndProvider(model);
    if (!provider) return model;
    try {
      const models = await queryClient.fetchQuery(
        providerModelsQueryOptions(provider),
      );
      const match = models.find((m) => m.name === modelName);
      if (match?.hidden && match.canonical) {
        return `${provider}/${match.canonical}`;
      }
    } catch {
      // Catalog unavailable — open with the stored model, exactly as today.
    }
    return model;
  };

  const openForm = async (
    view: SettingsView | null,
    profile: LlmProfileSummary | null = null,
  ) => {
    // The profiles list passes the profile only when editing; Add Profile
    // opens a blank create form.
    let editProfile = profile;
    if (editProfile?.model) {
      // Resolved *before* the form opens: SdkSectionPage re-hydrates (and
      // resets dirty state) whenever the initial-value overrides change, so
      // a post-mount translation could clobber in-progress user edits.
      const canonicalModel = await resolveCanonicalModel(editProfile.model);
      if (canonicalModel !== editProfile.model) {
        editProfile = { ...editProfile, model: canonicalModel };
      }
    }
    const isEdit = editProfile !== null;
    setProfileFormMode(isEdit ? "edit" : "create");
    setEditingProfile(editProfile);
    setProfileName(editProfile?.name ?? "");
    setInitialProfileName(editProfile?.name ?? "");
    setInitialViewHint(view);
    // Blank for create; the edited profile's provider for edit.
    setSelectedProvider(
      editProfile?.model
        ? extractModelAndProvider(editProfile.model).provider || null
        : null,
    );
    setShowProfiles(false);
  };

  // Create starts from a clean slate; edit hydrates from the clicked
  // profile rather than from the active settings.
  const profileFormInitialValueOverrides = React.useMemo(() => {
    if (profileFormMode === "create") {
      return {
        agent_settings: {
          "llm.model": "",
          "llm.api_key": "",
          "llm.base_url": "",
        },
      };
    }
    if (profileFormMode === "edit" && editingProfile) {
      const fieldDefault = (fieldKey: string) =>
        String(getSchemaFieldDefaultValue(schema, fieldKey) ?? "");
      return {
        agent_settings: {
          "llm.model": editingProfile.model ?? fieldDefault("llm.model"),
          // The stored key can't be fetched; the profile's api_key_set
          // drives the <hidden> placeholder instead.
          "llm.api_key": "",
          "llm.base_url":
            editingProfile.base_url ?? fieldDefault("llm.base_url"),
        },
      };
    }
    return undefined;
  }, [editingProfile, profileFormMode, schema]);

  if (isProfilesView) {
    if (isOrgProfileMode) {
      if (!organizationId) {
        return null;
      }
      return (
        <OrgLlmProfilesManager
          orgId={organizationId}
          canManage={canManageProfilesForScope}
          onAddProfile={
            canManageProfilesForScope ? () => openForm(null) : undefined
          }
          onEditProfile={
            canManageProfilesForScope
              ? (profile) => openForm(null, profile)
              : undefined
          }
        />
      );
    }
    // Use personal profiles manager for OSS mode
    return (
      <LlmProfilesManager
        onAddProfile={() => openForm(null)}
        onEditProfile={(profile) => openForm(null, profile)}
      />
    );
  }

  // Sub-page back affordance when profiles are enabled. Replaces the previous
  // "Profiles" trailing action so the form view follows the second-level
  // settings pattern.
  const backToProfiles = shouldShowProfilesForScope ? (
    <button
      data-testid="llm-back-to-profiles"
      type="button"
      onClick={() => {
        setInitialViewHint(null);
        setProfileFormMode(null);
        setEditingProfile(null);
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
        settingsSources={[
          {
            settingsSource: "agent_settings",
            sectionKeys: ["llm"],
            excludeKeys: LLM_EXCLUDED_KEYS,
          },
        ]}
        header={buildHeader}
        buildPayload={buildPayload}
        // The profile form can always be saved: it snapshots the current LLM
        // config as a profile, and the name is optional — it falls back to a
        // model-derived default in handleSaveSuccess. So don't gate Save on the
        // settings fields being dirty. This matters in SaaS managed mode, where
        // the model is fixed and there's no editable API key, leaving the form
        // pristine and Save stuck disabled.
        extraDirty={canManageProfilesForScope}
        onSaveSuccess={handleSaveSuccess}
        getInitialView={getInitialView}
        initialValueOverrides={profileFormInitialValueOverrides}
        // A new profile starts blank and can't be saved until a model is
        // chosen — saving a model-less create form would only churn the
        // active settings without producing a profile.
        isSaveDisabled={({ values }) =>
          profileFormMode === "create" &&
          !(
            typeof values["llm.model"] === "string" &&
            values["llm.model"].trim().length > 0
          )
        }
        // Advanced (custom model + base URL) only exists under BYOK; gate the
        // toggle on it so Basic/Advanced aren't identical when BYOK is off.
        forceShowAdvancedView={allowUserLlmConfiguration}
        allowAdvancedView={allowUserLlmConfiguration}
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
