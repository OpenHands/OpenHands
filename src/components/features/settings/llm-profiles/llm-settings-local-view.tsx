import React, {
  useState,
  useCallback,
  useMemo,
  useEffect,
  useRef,
} from "react";
import { useTranslation } from "react-i18next";
import { LlmProfilesManager } from "./llm-profiles-manager";
import { ProfileNameInput } from "./profile-name-input";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LlmSettingsScreen } from "#/routes/llm-settings";
import { LLM_PROVIDER_CONNECTION_KEY } from "#/constants/llm-provider-connection";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import { useActivateLlmProfile } from "#/hooks/mutation/use-activate-llm-profile";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useSettings } from "#/hooks/query/use-settings";
import { useAgentSettingsSchema } from "#/hooks/query/use-agent-settings-schema";
import {
  useDefaultModel,
  useDefaultModelReady,
} from "#/hooks/query/use-free-models";
import { LlmSettingsInputsSkeleton } from "#/components/features/settings/llm-settings/llm-settings-inputs-skeleton";
import { DEFAULT_SETTINGS } from "#/services/settings";
import ProfilesService, {
  ProfileInfo,
  type SaveProfileRequest,
} from "#/api/profiles-service/profiles-service.api";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { getApiErrorMessage } from "#/utils/api-error-message";
import {
  deriveProfileNameFromModel,
  isProfileNameValid,
} from "#/utils/derive-profile-name";
import { SdkSectionSaveControl } from "../sdk-settings/sdk-section-page";
import {
  buildProfileLlmConfig,
  isBareProviderPrefix,
  prefillToFormValues,
  profileConfigToFormValues,
  type NewProfilePrefill,
} from "./llm-profile-form";
import {
  LLM_AUTH_TYPE_API_KEY,
  LLM_AUTH_TYPE_KEY,
  LLM_SUBSCRIPTION_VENDOR_KEY,
  OPENAI_SUBSCRIPTION_VENDOR,
} from "#/constants/llm-subscription";
import {
  buildSdkSettingsPayload,
  getAgentSettingValue,
  SettingsFormValues,
} from "#/utils/sdk-settings-schema";
import { BackNavButton } from "#/components/shared/buttons/back-nav-button";
import { Typography } from "#/ui/typography";
import { useSettingsSectionHeader } from "#/contexts/settings-section-header-context";

type ViewMode = "list" | "create" | "edit";

interface EditingProfile {
  profile: ProfileInfo;
  initialValues: SettingsFormValues;
  /**
   * The profile's full LLM config (flat keys; `api_key` is the encrypted
   * token). Used as the merge base on save so fields the user did not touch —
   * including ones hidden in the current tab or absent from the schema — are
   * preserved instead of being reset to LLM defaults by the full-replace save.
   */
  baseConfig: Record<string, unknown>;
}

export function shouldReapplyProfileAfterSave({
  activeProfileName,
  originalName,
  savedName,
}: {
  activeProfileName: string | null | undefined;
  originalName: string | null | undefined;
  savedName: string;
}): boolean {
  if (!activeProfileName) return false;
  if (originalName) return activeProfileName === originalName;
  return activeProfileName === savedName;
}

/**
 * LlmSettingsLocalView provides an integrated view for managing LLM profiles
 * in local agent-server mode. It supports listing, creating, and editing profiles.
 *
 * Note: This component manages multiple responsibilities (view state, validation,
 * form coordination, save logic). A future refactoring could extract these into
 * separate hooks (e.g., useProfileForm, useProfileSave) for better testability.
 * See PR review feedback for details.
 */

export function LlmSettingsLocalView() {
  const { t } = useTranslation("openhands");
  const { setHideSectionHeader } = useSettingsSectionHeader();
  const saveProfile = useSaveLlmProfile();
  const activateProfile = useActivateLlmProfile();
  const { data: profilesData } = useLlmProfiles();
  const { data: settings } = useSettings();
  const { data: agentSchema } = useAgentSettingsSchema(
    settings?.agent_settings_schema,
  );
  const createProfileDefaultModel =
    useDefaultModel() ?? DEFAULT_SETTINGS.llm_model;
  // Gate new-profile creation on the DB default query settling, mirroring
  // onboarding's `useDefaultModelReady`. Without this, clicking Add before
  // the hydrator finishes mounts the keyed form with the static fallback
  // model; SdkSectionPage ignores later initial-value changes once an
  // embedded form hydrates, so the eventual DB default never replaces it and
  // the user can save the wrong model.
  const isDefaultModelReady = useDefaultModelReady();
  // @spec LLP-001 — A fresh profile never inherits schema-declared credentials.
  // Set when a new profile starts from another flow (Add models' hint): the
  // form then opens on Advanced with the provider and connection seeded.
  const [createPrefill, setCreatePrefill] = useState<NewProfilePrefill | null>(
    null,
  );
  const createProfileInitialValues = useMemo(
    () => ({
      // A fresh profile may inherit options, but no existing credentials.
      ...Object.fromEntries(
        (agentSchema?.sections ?? []).flatMap((section) =>
          section.fields
            .filter((field) => field.secret)
            .map((field) => [field.key, ""]),
        ),
      ),
      "llm.model": createProfileDefaultModel,
      "llm.api_key": "",
      "llm.base_url": "",
      [LLM_PROVIDER_CONNECTION_KEY]: "",
      [LLM_AUTH_TYPE_KEY]: LLM_AUTH_TYPE_API_KEY,
      [LLM_SUBSCRIPTION_VENDOR_KEY]: OPENAI_SUBSCRIPTION_VENDOR,
      ...(createPrefill ? prefillToFormValues(createPrefill) : {}),
    }),
    [agentSchema, createProfileDefaultModel, createPrefill],
  );

  // Always hold the freshest schema. `handleEditProfile` awaits a network
  // round-trip before seeding the form, so reading the schema from a ref
  // (rather than the value captured in the callback's closure) picks up a
  // schema that finished loading during that await. This closes the brief
  // first-load window where the schema is still pending when Edit is clicked
  // and would otherwise seed only the three hard-coded keys.
  const agentSchemaRef = useRef(agentSchema);
  agentSchemaRef.current = agentSchema;
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  // Provider connections are available on the local agent-server and on cloud
  // when an org is bound (the org-scoped CRUD routes). A cloud backend without
  // an org (legacy API keys) cannot address them.
  const { backend, orgId } = useActiveBackend();
  const supportsConnections =
    backend.kind === "local" || (backend.kind === "cloud" && !!orgId);

  const [viewMode, setViewMode] = useState<ViewMode>("list");
  const [profileName, setProfileName] = useState("");
  // Whether the user typed the current create-mode name. Until they do (or
  // after they clear it), the name is auto-derived from the selected model.
  const [isProfileNameUserEdited, setIsProfileNameUserEdited] = useState(false);
  const [editingProfile, setEditingProfile] = useState<EditingProfile | null>(
    null,
  );
  const [saveControl, setSaveControl] = useState<SdkSectionSaveControl | null>(
    null,
  );
  const automaticCreateBooleanKeys = useRef<Set<string> | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [isValidating, setIsValidating] = useState(false);

  useEffect(() => {
    setHideSectionHeader(viewMode !== "list");
    return () => setHideSectionHeader(false);
  }, [viewMode, setHideSectionHeader]);

  // Get existing profile names for validation
  const existingNames = useMemo(
    () => new Set(profilesData?.profiles.map((p) => p.name) ?? []),
    [profilesData],
  );

  // Validate profile name. The shared validator rejects any whitespace, so
  // the duplicate check below compares the raw value directly.
  const isNameFormatValid = isProfileNameValid(profileName, {
    isRequired: true,
  });
  // In edit mode, name can match current profile name
  const isDuplicateName =
    isNameFormatValid &&
    existingNames.has(profileName) &&
    (viewMode === "create" ||
      (viewMode === "edit" && profileName !== editingProfile?.profile.name));
  const isNameValid = isNameFormatValid && !isDuplicateName;

  const handleProfileNameChange = useCallback((value: string) => {
    setProfileName(value);
    setIsProfileNameUserEdited(value !== "");
  }, []);

  const handleAddProfile = useCallback((prefill?: NewProfilePrefill) => {
    automaticCreateBooleanKeys.current = null;
    setCreatePrefill(prefill ?? null);
    setProfileName("");
    setIsProfileNameUserEdited(false);
    setEditingProfile(null);
    setViewMode("create");
  }, []);

  const handleEditProfile = useCallback(
    async (profile: ProfileInfo) => {
      try {
        // Fetch profile details with encrypted secrets to preserve API key
        const detail = await ProfilesService.getProfile(
          profile.name,
          "encrypted",
        );

        // Profile config contains llm settings directly at the top level
        // The structure is: { model, api_key, base_url, ... }
        // (NOT nested under a "llm" key)
        const config = (detail.config ?? {}) as Record<string, unknown>;

        // Seed every llm.* form field from the profile config so all tabs —
        // including "All" — reflect the profile's real values rather than the
        // active settings. Fields absent from the schema are still preserved
        // on save via `baseConfig`. Read the schema from the ref so a schema
        // that loaded during the `getProfile` await above is used.
        const initialValues = profileConfigToFormValues(
          agentSchemaRef.current,
          config,
        );

        setEditingProfile({ profile, initialValues, baseConfig: config });
        setProfileName(profile.name);
        setViewMode("edit");
      } catch (error) {
        console.error("Failed to fetch profile details:", error);
        displayErrorToast(t(I18nKey.ERROR$GENERIC));
      }
    },
    [t],
  );

  const handleBackToList = useCallback(() => {
    automaticCreateBooleanKeys.current = null;
    setCreatePrefill(null);
    setViewMode("list");
    setEditingProfile(null);
    setProfileName("");
    setSaveControl(null);
  }, []);

  const handleSaveControlChange = useCallback(
    (control: SdkSectionSaveControl) => {
      setSaveControl(control);
      const currentSettings = settingsRef.current;
      const currentSchema = agentSchemaRef.current;
      // Capture the draft's raw origin once it hydrates. Null-default booleans
      // render unchecked, but null preserves provider auto-detection. A later
      // settings refresh must not change that meaning for this open draft.
      if (
        viewMode === "create" &&
        automaticCreateBooleanKeys.current === null &&
        currentSettings &&
        currentSchema &&
        "llm.model" in control.values
      ) {
        automaticCreateBooleanKeys.current = new Set(
          currentSchema.sections.flatMap((section) =>
            section.fields
              .filter(
                (field) =>
                  field.value_type === "boolean" &&
                  field.default === null &&
                  getAgentSettingValue(currentSettings, field.key) === null,
              )
              .map((field) => field.key),
          ),
        );
      }

      // Auto-derive profile name from model in create mode, and keep it
      // following later model changes until the user types their own name.
      // Note: The uniqueness check uses existingNames from state, which is derived
      // from profilesData at component render time. If another client creates a
      // profile with the same derived name while this form is open, the client-side
      // check would pass but the server save would fail with a conflict error.
      // This is acceptable for the current use case; the server error is handled
      // gracefully in handleSave.
      if (viewMode === "create" && !isProfileNameUserEdited) {
        const modelValue = control.values["llm.model"];
        if (typeof modelValue === "string" && modelValue) {
          const derived = deriveProfileNameFromModel(modelValue);
          // Never auto-fill an existing profile's name; clear the previous
          // model's name instead so it cannot be saved under the new model.
          setProfileName(existingNames.has(derived) ? "" : derived);
        }
      }
    },
    [viewMode, isProfileNameUserEdited, existingNames],
  );

  // @spec LLP-001 — New profiles preserve options and credential boundaries.
  const handleSave = useCallback(async () => {
    if (!saveControl || !isNameValid) return;

    // Coerced, dirty-only changes from the embedded form. Merging these over
    // the profile's existing full config preserves fields the user did not
    // touch (incl. ones hidden in the current tab or absent from the schema);
    // the backend replaces the whole LLM object, so a partial payload would
    // otherwise reset everything else to defaults.
    let dirtyLlm: Record<string, unknown>;
    try {
      // A new profile has no saved config to merge over, so persist every
      // non-secret value its form displays, using the same coercion as edits.
      // Explicit credential edits are included by the dirty payload below.
      const initialPayload =
        viewMode === "create" && agentSchema
          ? buildSdkSettingsPayload(
              agentSchema,
              saveControl.values,
              Object.fromEntries(
                agentSchema.sections.flatMap((section) =>
                  section.fields
                    .filter((field) => field.key in saveControl.values)
                    .map((field) => [
                      field.key,
                      !field.secret &&
                        !automaticCreateBooleanKeys.current?.has(field.key),
                    ]),
                ),
              ),
            )
          : {};
      dirtyLlm = {
        ...((initialPayload.llm ?? {}) as Record<string, unknown>),
        ...((saveControl.getDirtyPayload().llm ?? {}) as Record<
          string,
          unknown
        >),
      };
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
      return;
    }

    const { llmConfig, connectionId } = buildProfileLlmConfig({
      baseConfig:
        viewMode === "edit" && editingProfile?.baseConfig
          ? editingProfile.baseConfig
          : {},
      dirtyLlm,
      values: saveControl.values,
      view: saveControl.view,
      supportsConnections,
      isCloud: backend.kind === "cloud",
    });

    const model = typeof llmConfig.model === "string" ? llmConfig.model : "";
    // A bare "provider/" prefix (a prefilled draft left as-is) names no model.
    if (!model || isBareProviderPrefix(model)) {
      displayErrorToast(t(I18nKey.SETTINGS$MODEL_REQUIRED));
      return;
    }

    const trimmedName = profileName.trim();
    const originalName = editingProfile?.profile.name;
    const isRename =
      viewMode === "edit" && originalName && originalName !== trimmedName;
    const shouldReapplyActiveProfile = shouldReapplyProfileAfterSave({
      activeProfileName: profilesData?.active_profile,
      originalName,
      savedName: trimmedName,
    });

    setIsSaving(true);
    setIsValidating(true);
    try {
      // Pre-flight validation fires a minimal completion to catch a
      // misconfigured profile before saving it. Skip it for connection-linked
      // profiles: their credential lives on the provider connection, not
      // inline, so there is nothing on this profile to pre-flight here.
      if (!connectionId) {
        const preflight = await ProfilesService.validateProfile(trimmedName, {
          llm: llmConfig as SaveProfileRequest["llm"],
          include_secrets: true,
        });
        if (preflight && !preflight.valid) {
          const errorMsg = preflight.error?.message ?? t(I18nKey.ERROR$GENERIC);
          displayErrorToast(errorMsg);
          return;
        }
      }

      setIsValidating(false);

      // If editing and name changed, rename the profile first
      if (isRename) {
        await ProfilesService.renameProfile(originalName, trimmedName);
      }

      await saveProfile.mutateAsync({
        name: trimmedName,
        request: {
          llm: llmConfig as SaveProfileRequest["llm"],
          include_secrets: true,
        },
      });

      // Activation copies the profile into agent_settings.llm, which the
      // planner falls back to. Re-applying an active profile keeps that copy
      // in sync after editing or recreating a profile with its name.
      if (shouldReapplyActiveProfile) {
        await activateProfile.mutateAsync(trimmedName);
      }

      displaySuccessToast(
        viewMode === "create"
          ? t(I18nKey.SETTINGS$PROFILE_CREATED, { name: trimmedName })
          : t(I18nKey.SETTINGS$PROFILE_UPDATED, { name: trimmedName }),
      );
      handleBackToList();
    } catch (error) {
      console.error("Failed to save profile:", error);
      displayErrorToast(getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)));
    } finally {
      setIsValidating(false);
      setIsSaving(false);
    }
  }, [
    saveControl,
    agentSchema,
    isNameValid,
    supportsConnections,
    profileName,
    viewMode,
    editingProfile,
    backend.kind,
    profilesData?.active_profile,
    saveProfile,
    activateProfile,
    t,
    handleBackToList,
  ]);

  // List view: show profiles manager
  if (viewMode === "list") {
    return (
      <LlmProfilesManager
        onAddProfile={handleAddProfile}
        onEditProfile={handleEditProfile}
      />
    );
  }

  const profileEditorTitle =
    viewMode === "edit"
      ? t(I18nKey.SETTINGS$EDIT_LLM_PROFILE)
      : t(I18nKey.SETTINGS$ADD_LLM_PROFILE);
  const profileEditorDescription =
    viewMode === "edit" && editingProfile
      ? t(I18nKey.SETTINGS$PROFILE_LOADED, {
          name: editingProfile.profile.name,
        })
      : t(I18nKey.SETTINGS$PROFILE_SAVE_HINT);

  // Create/Edit view: show form with profile name input
  return (
    <div className="flex flex-col gap-6">
      {/* Header with back button */}
      <div className="flex flex-col gap-2">
        <BackNavButton testId="back-to-profiles" onClick={handleBackToList}>
          {t(I18nKey.BUTTON$BACK)}
        </BackNavButton>
        <Typography.H2 testId="profile-editor-title">
          {profileEditorTitle}
        </Typography.H2>
        <p
          data-testid="profile-editor-description"
          className="text-sm leading-5 text-tertiary-light"
        >
          {profileEditorDescription}
        </p>
      </div>

      {/* Profile name input */}
      <ProfileNameInput
        testId="profile-name-input"
        value={profileName}
        onChange={handleProfileNameChange}
        isRequired
        error={
          isDuplicateName
            ? t(I18nKey.SETTINGS$LLM_PROFILE_NAME_TAKEN)
            : undefined
        }
      />

      {/* Profile form - key ensures form remounts when switching profiles.
          In create mode, wait for the DB default query to settle before
          mounting the keyed form; otherwise SdkSectionPage would hydrate with
          the static fallback and ignore the eventual DB default. Edit mode is
          seeded from the existing profile, so it does not need the gate. */}
      {viewMode === "create" && !isDefaultModelReady ? (
        <LlmSettingsInputsSkeleton />
      ) : (
        <LlmSettingsScreen
          key={
            viewMode === "edit"
              ? `edit-${editingProfile?.profile.name}`
              : "new-profile"
          }
          embedded
          hideSaveButton
          markInitialOverridesDirty={false}
          initialValueOverrides={
            viewMode === "edit" && editingProfile?.initialValues
              ? // Edit mode: use the existing profile values
                editingProfile.initialValues
              : createProfileInitialValues
          }
          showProviderConnection={supportsConnections}
          initialView={
            viewMode === "create"
              ? createPrefill
                ? "advanced"
                : "basic"
              : undefined
          }
          onSaveControlChange={handleSaveControlChange}
        />
      )}

      {/* Action buttons */}
      <div className="flex justify-start gap-3 pt-4">
        <BrandButton
          testId="cancel-profile-btn"
          type="button"
          variant="secondary"
          onClick={handleBackToList}
        >
          {t(I18nKey.BUTTON$CANCEL)}
        </BrandButton>
        <BrandButton
          testId="save-profile-btn"
          type="button"
          variant="primary"
          onClick={handleSave}
          isDisabled={
            !isNameValid ||
            isSaving ||
            isValidating ||
            !saveControl ||
            !(
              viewMode === "create" ||
              saveControl.isDirty ||
              profileName !== editingProfile?.profile.name
            )
          }
          aria-busy={isSaving || isValidating}
        >
          {isValidating
            ? t(I18nKey.STATUS$VALIDATING)
            : isSaving
              ? t(I18nKey.STATUS$SAVING)
              : t(I18nKey.BUTTON$SAVE)}
        </BrandButton>
      </div>
    </div>
  );
}
