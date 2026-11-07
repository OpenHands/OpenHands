import { useCallback, useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { ensureDefaultAgentProfileMigrated } from "#/hooks/use-migrate-default-agent-profile";
import { useTranslation } from "react-i18next";
import { AgentProfilesManager } from "./agent-profiles-manager";
import { mergeAgentProfileSaveInput } from "./merge-agent-profile-save-input";
import { newProfileSeed } from "./new-profile-seed";
import { ProfileNameInput } from "#/components/features/settings/llm-profiles/profile-name-input";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import {
  AgentSettingsScreen,
  type AgentSettingsSaveControl,
} from "#/routes/agent-settings";
import AgentProfilesService, {
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
  type AgentProfile,
  type AgentProfileSummary,
  type AgentProfileSaveInput,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import { useSaveAgentProfile } from "#/hooks/mutation/use-save-agent-profile";
import { useRenameAgentProfile } from "#/hooks/mutation/use-rename-agent-profile";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useSettings } from "#/hooks/query/use-settings";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { isProfileNameValid } from "#/utils/derive-profile-name";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { SettingsValue } from "#/types/settings";
import { BackNavButton } from "#/components/shared/buttons/back-nav-button";
import { Typography } from "#/ui/typography";
import { useSettingsSectionHeader } from "#/contexts/settings-section-header-context";
import { parseCommand } from "#/utils/acp-command";
import { agentProfileSupportsSkills } from "#/api/agent-profiles-service/profile-field-support";
import {
  defaultSkillEnablement,
  toProfileSkillFields,
} from "#/utils/skill-enablement";
import {
  applyDefaultAgentProfileToGlobal,
  enabledMcpServerRefs,
  sharedDefaultProfileFields,
} from "#/hooks/mutation/sync-agent-profiles";
import type { Settings } from "#/types/settings";

type ViewMode = "list" | "create" | "edit";

/**
 * Build the `agent_settings`-shaped seed the embedded {@link AgentSettingsScreen}
 * consumes for an existing profile. The stored ACP command is a shell string;
 * the form's init logic expects a token array, so split it here.
 */
function toAgentSettingsOverride(
  profile: AgentProfile,
): Record<string, SettingsValue> {
  const secretRefs = profile.secret_refs ?? null;
  if (profile.agent_kind === "acp") {
    return {
      agent_kind: "acp",
      mcp_server_refs: profile.mcp_server_refs ?? null,
      secret_refs: secretRefs,
      acp_server: profile.acp_server,
      acp_command: profile.acp_command ? parseCommand(profile.acp_command) : [],
      acp_args: profile.acp_args ?? [],
      acp_model: profile.acp_model ?? "",
    };
  }
  const stored = profile as typeof profile & {
    enable_classify_and_switch_llm_tool?: boolean;
    meta_profile_ref?: string | null;
  };
  return {
    agent_kind: "openhands",
    mcp_server_refs: profile.mcp_server_refs ?? null,
    enable_classify_and_switch_llm_tool:
      stored.enable_classify_and_switch_llm_tool ?? false,
    meta_profile_ref: stored.meta_profile_ref ?? null,
    tools: (profile.tools as SettingsValue) ?? null,
    tool_concurrency_limit: profile.tool_concurrency_limit,
    secret_refs: secretRefs,
    public_skills:
      (profile as { public_skills?: string[] | null }).public_skills ?? null,
    disabled_skills:
      (profile as { disabled_skills?: string[] }).disabled_skills ?? [],
    persona: profile.persona ?? null,
    system_message_suffix: profile.system_message_suffix ?? null,
    condenser: (profile.condenser as SettingsValue) ?? null,
    verification: (profile.verification as unknown as SettingsValue) ?? null,
  };
}

/** An OpenHands `default` opens on the MCP servers switched on and every secret. */
function defaultProfileOverride(
  profile: AgentProfile,
  settings: Settings | undefined,
): Record<string, SettingsValue> {
  return {
    ...toAgentSettingsOverride(profile),
    mcp_server_refs: enabledMcpServerRefs(settings),
    secret_refs: null,
  };
}

/**
 * AgentProfilesLocalView mirrors {@link LlmSettingsLocalView}: a list of the
 * user's Agent profiles, with a create/edit view that reuses the existing Agent
 * settings form (embedded) plus a profile name and — for OpenHands profiles —
 * an LLM-profile picker. Available on local backends only.
 */
export function AgentProfilesLocalView() {
  const { t } = useTranslation("openhands");
  const { setHideSectionHeader } = useSettingsSectionHeader();
  const saveProfile = useSaveAgentProfile();
  const renameProfile = useRenameAgentProfile();
  const { data: profilesData } = useAgentProfiles();
  const { data: llmProfilesData, isLoading: isLlmProfilesLoading } =
    useLlmProfiles();
  const { data: settings } = useSettings();
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();
  // Edits must build on, and land after, the one-time `default` migration.
  const waitForMigration = useCallback(
    () =>
      backend.kind === "local"
        ? ensureDefaultAgentProfileMigrated(queryClient, backend.id)
        : Promise.resolve(true as const),
    [backend.kind, backend.id, queryClient],
  );
  // On cloud, `default` is an ordinary profile name.
  const isSharedDefault = useCallback(
    (name: string | undefined) =>
      backend.kind === "local" &&
      name === WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
    [backend.kind],
  );

  const [viewMode, setViewMode] = useState<ViewMode>("list");
  const [profileName, setProfileName] = useState("");
  const [nameTouched, setNameTouched] = useState(false);
  const [editingProfile, setEditingProfile] = useState<AgentProfile | null>(
    null,
  );
  const [seedProfile, setSeedProfile] = useState<AgentProfile | null>(null);
  const [override, setOverride] = useState<Record<
    string,
    SettingsValue
  > | null>(null);
  const [llmProfileRef, setLlmProfileRef] = useState("");
  const [saveControl, setSaveControl] =
    useState<AgentSettingsSaveControl | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    setHideSectionHeader(viewMode !== "list");
    return () => setHideSectionHeader(false);
  }, [viewMode, setHideSectionHeader]);

  const llmProfiles = llmProfilesData?.profiles ?? [];
  const defaultLlmProfileRef =
    llmProfilesData?.active_profile ?? llmProfiles[0]?.name ?? "";

  const existingNames = useMemo(
    () => new Set(profilesData?.profiles?.map((p) => p.name) ?? []),
    [profilesData],
  );

  // The shared validator rejects whitespace, so the duplicate check compares raw.
  const isNameFormatValid = isProfileNameValid(profileName, {
    isRequired: true,
  });
  const isDuplicateName =
    isNameFormatValid &&
    existingNames.has(profileName) &&
    (viewMode === "create" ||
      (viewMode === "edit" && profileName !== editingProfile?.name));
  const isNameValid = isNameFormatValid && !isDuplicateName;

  const handleAddProfile = useCallback(async () => {
    let seed: AgentProfile | null = null;
    if (backend.kind === "local") {
      try {
        await waitForMigration();
        const { profile } = await AgentProfilesService.getProfile(
          WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
        );
        seed = newProfileSeed(profile);
      } catch {
        // Without a readable `default`, the editor starts from its own values.
      }
    }
    setProfileName("");
    setNameTouched(false);
    setEditingProfile(null);
    setSeedProfile(seed);
    // A new profile copies an OpenHands `default`, skills included.
    setOverride({
      ...(seed
        ? toAgentSettingsOverride(seed)
        : agentProfileSupportsSkills()
          ? toProfileSkillFields(defaultSkillEnablement(settings ?? {}))
          : {}),
      agent_kind: "openhands",
    });
    setLlmProfileRef(defaultLlmProfileRef);
    setSaveControl(null);
    setViewMode("create");
  }, [backend.kind, waitForMigration, defaultLlmProfileRef, settings]);

  const handleEditProfile = useCallback(
    async (summary: AgentProfileSummary) => {
      try {
        await waitForMigration();
        // Fetch with encrypted secret exposure so any `skills[].mcp_tools`
        // values arrive as Fernet tokens rather than masks — the save below
        // round-trips the stored profile, and posting back a mask would
        // persist it literally (the local save only decrypts tokens).
        const detail = await AgentProfilesService.getProfile(
          summary.name,
          "encrypted",
        );
        const profile = detail.profile;
        const isDefault = isSharedDefault(profile.name);
        setEditingProfile(profile);
        setOverride(
          isDefault && profile.agent_kind === "openhands"
            ? defaultProfileOverride(profile, settings)
            : toAgentSettingsOverride(profile),
        );
        if (isDefault) {
          setLlmProfileRef(
            defaultLlmProfileRef ||
              (profile.agent_kind === "openhands"
                ? profile.llm_profile_ref
                : ""),
          );
        } else if (profile.agent_kind === "openhands") {
          // A profile can reference an LLM profile that's since been deleted
          // (or renamed): the dropdown would render nothing selected while
          // `llmProfileRef` still holds the stale name and saves it straight
          // back. Validate against the live list and fall back to the default
          // so a dangling ref self-heals on load (#1571 review). Skip the
          // check while the list is still loading rather than treating an
          // unloaded cache as "not found".
          const refIsLive =
            isLlmProfilesLoading ||
            llmProfiles.some((p) => p.name === profile.llm_profile_ref);
          setLlmProfileRef(
            refIsLive ? profile.llm_profile_ref : defaultLlmProfileRef,
          );
        } else {
          setLlmProfileRef("");
        }
        setProfileName(profile.name);
        setSaveControl(null);
        setViewMode("edit");
      } catch (error) {
        console.error("Failed to fetch agent profile:", error);
        displayErrorToast(t(I18nKey.ERROR$GENERIC));
      }
    },
    [
      t,
      llmProfiles,
      isLlmProfilesLoading,
      defaultLlmProfileRef,
      waitForMigration,
      isSharedDefault,
      settings,
    ],
  );

  const handleBackToList = useCallback(() => {
    setViewMode("list");
    setEditingProfile(null);
    setSeedProfile(null);
    setProfileName("");
    setNameTouched(false);
    setOverride(null);
    setSaveControl(null);
  }, []);

  const handleSave = useCallback(async () => {
    if (!saveControl || !isNameValid) return;

    // Build the variant-specific fields from the embedded form (may throw on
    // invalid input, e.g. a bad concurrency value).
    let input: AgentProfileSaveInput;
    try {
      const fields = saveControl.buildAgentProfileFields();
      if (fields.agent_kind === "openhands") {
        if (!llmProfileRef) {
          displayErrorToast(t(I18nKey.SETTINGS$AGENT_PROFILE_LLM_REQUIRED));
          return;
        }
        input = {
          ...fields,
          llm_profile_ref: llmProfileRef,
        } as AgentProfileSaveInput;
      } else {
        input = fields as AgentProfileSaveInput;
      }
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
      return;
    }

    const trimmedName = profileName.trim();
    const originalName = editingProfile?.name;
    const isRename =
      viewMode === "edit" && originalName && originalName !== trimmedName;
    const savesDefault =
      viewMode === "edit" &&
      isSharedDefault(originalName) &&
      trimmedName === WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME &&
      input.agent_kind === "openhands";

    setIsSaving(true);
    try {
      await waitForMigration();
      // Persist ACP credentials (global secrets) before the profile spec.
      if (saveControl.credentials.isDirty) {
        const ok = await saveControl.credentials.save({ silent: true });
        if (!ok) return;
        saveControl.credentials.reset();
      }

      // Rename first (preserves the profile's stable id / active pointer);
      // saving to a new name would otherwise mint a fresh profile.
      if (isRename) {
        await renameProfile.mutateAsync({
          name: originalName,
          newName: trimmedName,
        });
      }

      // Save is a whole-profile overwrite: spread the stored profile under
      // the edited fields so the fields this editor doesn't model survive an
      // edit (kind-aware — a kind switch stays a clean variant replacement).
      const profile = mergeAgentProfileSaveInput(
        viewMode === "edit" ? editingProfile : seedProfile,
        savesDefault ? sharedDefaultProfileFields(input) : input,
      );
      await saveProfile.mutateAsync({ name: trimmedName, profile });
      if (savesDefault)
        await applyDefaultAgentProfileToGlobal(queryClient, input);

      displaySuccessToast(
        viewMode === "create"
          ? t(I18nKey.SETTINGS$PROFILE_CREATED, { name: trimmedName })
          : t(I18nKey.SETTINGS$PROFILE_UPDATED, { name: trimmedName }),
      );
      handleBackToList();
    } catch (error) {
      const message = retrieveAxiosErrorMessage(error as never);
      displayErrorToast(message || t(I18nKey.ERROR$GENERIC));
    } finally {
      setIsSaving(false);
    }
  }, [
    saveControl,
    isNameValid,
    llmProfileRef,
    profileName,
    viewMode,
    editingProfile,
    seedProfile,
    saveProfile,
    renameProfile,
    t,
    handleBackToList,
    waitForMigration,
    isSharedDefault,
    queryClient,
  ]);

  if (viewMode === "list") {
    return (
      <div className="flex flex-col gap-4">
        <AgentProfilesManager
          onAddProfile={handleAddProfile}
          onEditProfile={handleEditProfile}
        />
      </div>
    );
  }

  const editorTitle =
    viewMode === "edit"
      ? t(I18nKey.SETTINGS$EDIT_AGENT_PROFILE)
      : t(I18nKey.SETTINGS$ADD_AGENT_PROFILE);
  const editorDescription =
    viewMode === "edit" && editingProfile
      ? t(I18nKey.SETTINGS$PROFILE_LOADED, { name: editingProfile.name })
      : t(I18nKey.SETTINGS$AGENT_PROFILE_SAVE_HINT);
  const isOpenHands = saveControl?.agentType !== "acp";
  const isDefaultLocked =
    viewMode === "edit" && isSharedDefault(editingProfile?.name);
  const isEditingDefault =
    isDefaultLocked &&
    profileName.trim() === WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME;
  const nameDirty = viewMode === "edit" && profileName !== editingProfile?.name;
  const loadedLlmRef =
    editingProfile?.agent_kind === "openhands"
      ? editingProfile.llm_profile_ref
      : "";
  const llmRefDirty =
    viewMode === "edit" && isOpenHands && llmProfileRef !== loadedLlmRef;
  const hasUnsavedChanges =
    viewMode === "create" ||
    Boolean(saveControl?.isDirty) ||
    nameDirty ||
    llmRefDirty;

  // @spec FF-001 — Profile validation feedback follows the host error role
  // OpenHands profiles reference an LLM profile (required).
  const llmPicker =
    llmProfiles.length > 0 ? (
      <div className="flex flex-col gap-2">
        <SettingsDropdownInput
          testId="agent-profile-llm-selector"
          name="agent-profile-llm"
          label={t(I18nKey.SETTINGS$AGENT_PROFILE_LLM_LABEL)}
          items={llmProfiles.map((p) => ({
            key: p.name,
            label: p.model ? `${p.name} (${p.model})` : p.name,
          }))}
          selectedKey={llmProfileRef}
          onSelectionChange={(key) => key && setLlmProfileRef(String(key))}
        />
        {isEditingDefault ? (
          <p
            data-testid="agent-profile-llm-default-note"
            className="text-xs text-tertiary-alt"
          >
            {t(I18nKey.SETTINGS$AGENT_PROFILE_DEFAULT_LLM_NOTE)}
          </p>
        ) : null}
      </div>
    ) : (
      <p
        data-testid="agent-profile-no-llm"
        className="text-sm text-feedback-error"
      >
        {t(I18nKey.SETTINGS$AGENT_PROFILE_NO_LLM)}
      </p>
    );

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <BackNavButton
          testId="back-to-agent-profiles"
          onClick={handleBackToList}
        >
          {t(I18nKey.BUTTON$BACK)}
        </BackNavButton>
        <Typography.H2 testId="agent-profile-editor-title">
          {editorTitle}
        </Typography.H2>
        <p
          data-testid="agent-profile-editor-description"
          className="text-sm leading-5 text-tertiary-light"
        >
          {editorDescription}
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <ProfileNameInput
          testId="agent-profile-name-input"
          value={profileName}
          onChange={(name) => {
            setNameTouched(true);
            setProfileName(name);
          }}
          // A new profile's empty name isn't an error until the user types.
          isRequired={viewMode === "edit" || nameTouched}
          isDisabled={isDefaultLocked}
          error={
            isDuplicateName
              ? t(I18nKey.SETTINGS$AGENT_PROFILE_NAME_TAKEN)
              : undefined
          }
        />
        {isDefaultLocked ? (
          <p
            data-testid="agent-profile-default-locked-note"
            className="text-xs text-tertiary-alt"
          >
            {t(I18nKey.SETTINGS$AGENT_PROFILE_DEFAULT_LOCKED)}
          </p>
        ) : null}
      </div>

      {/* Reuse the existing Agent settings form to define the agent. */}
      <AgentSettingsScreen
        key={viewMode === "edit" ? `edit-${editingProfile?.id}` : "new-profile"}
        agentSettingsOverride={override}
        onSaveControlChange={setSaveControl}
        isDefaultProfile={isEditingDefault}
        llmPicker={llmPicker}
      />

      <div className="flex justify-start gap-3 pt-4">
        <BrandButton
          testId="cancel-agent-profile-btn"
          type="button"
          variant="secondary"
          onClick={handleBackToList}
        >
          {t(I18nKey.BUTTON$CANCEL)}
        </BrandButton>
        <BrandButton
          testId="save-agent-profile-btn"
          type="button"
          variant="primary"
          onClick={handleSave}
          isDisabled={
            !isNameValid ||
            isSaving ||
            !saveControl?.isValid ||
            !hasUnsavedChanges
          }
          aria-busy={isSaving}
        >
          {isSaving ? t(I18nKey.SETTINGS$SAVING) : t(I18nKey.BUTTON$SAVE)}
        </BrandButton>
      </div>
    </div>
  );
}
