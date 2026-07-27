import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { AgentProfilesManager } from "./agent-profiles-manager";
import { mergeAgentProfileSaveInput } from "./merge-agent-profile-save-input";
import { ProfileNameInput } from "#/components/features/settings/llm-profiles/profile-name-input";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import {
  AgentSettingsScreen,
  type AgentSettingsSaveControl,
} from "#/routes/agent-settings";
import AgentProfilesService, {
  type AgentProfile,
  type AgentProfileSummary,
  type AgentProfileSaveInput,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import { useSaveAgentProfile } from "#/hooks/mutation/use-save-agent-profile";
import { useRenameAgentProfile } from "#/hooks/mutation/use-rename-agent-profile";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
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

type ViewMode = "list" | "create" | "edit";

/**
 * Build the `agent_settings`-shaped seed the embedded {@link AgentSettingsScreen}
 * consumes for an existing profile. The stored ACP command is a shell string;
 * the form's init logic expects a token array, so split it here.
 */
function toAgentSettingsOverride(
  profile: AgentProfile,
): Record<string, SettingsValue> {
  if (profile.agent_kind === "acp") {
    return {
      agent_kind: "acp",
      acp_server: profile.acp_server,
      acp_command: profile.acp_command ? parseCommand(profile.acp_command) : [],
      acp_args: profile.acp_args ?? [],
      acp_model: profile.acp_model ?? "",
    };
  }
  return {
    agent_kind: "openhands",
    enable_sub_agents: profile.enable_sub_agents,
    tool_concurrency_limit: profile.tool_concurrency_limit,
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

  const [viewMode, setViewMode] = useState<ViewMode>("list");
  const [profileName, setProfileName] = useState("");
  const [editingProfile, setEditingProfile] = useState<AgentProfile | null>(
    null,
  );
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

  // The shared validator rejects whitespace, so duplicate checks compare raw.
  const isNameValid = useMemo(() => {
    if (!isProfileNameValid(profileName, { isRequired: true })) return false;
    if (viewMode === "create" && existingNames.has(profileName)) return false;
    if (
      viewMode === "edit" &&
      profileName !== editingProfile?.name &&
      existingNames.has(profileName)
    ) {
      return false;
    }
    return true;
  }, [profileName, viewMode, existingNames, editingProfile?.name]);

  const handleAddProfile = useCallback(() => {
    setProfileName("");
    setEditingProfile(null);
    setOverride({ agent_kind: "openhands" });
    setLlmProfileRef(defaultLlmProfileRef);
    setSaveControl(null);
    setViewMode("create");
  }, [defaultLlmProfileRef]);

  const handleEditProfile = useCallback(
    async (summary: AgentProfileSummary) => {
      try {
        // Fetch with encrypted secret exposure so any `skills[].mcp_tools`
        // values arrive as Fernet tokens rather than masks — the save below
        // round-trips the stored profile, and posting back a mask would
        // persist it literally (the local save only decrypts tokens).
        const detail = await AgentProfilesService.getProfile(
          summary.name,
          "encrypted",
        );
        const profile = detail.profile;
        setEditingProfile(profile);
        setOverride(toAgentSettingsOverride(profile));
        if (profile.agent_kind === "openhands") {
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
    [t, llmProfiles, isLlmProfilesLoading, defaultLlmProfileRef],
  );

  const handleBackToList = useCallback(() => {
    setViewMode("list");
    setEditingProfile(null);
    setProfileName("");
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
        input = { ...fields, llm_profile_ref: llmProfileRef };
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

    setIsSaving(true);
    try {
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
        viewMode === "edit" ? editingProfile : null,
        input,
      );
      await saveProfile.mutateAsync({ name: trimmedName, profile });

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
    saveProfile,
    renameProfile,
    t,
    handleBackToList,
  ]);

  if (viewMode === "list") {
    return (
      <AgentProfilesManager
        onAddProfile={handleAddProfile}
        onEditProfile={handleEditProfile}
      />
    );
  }

  const editorTitle =
    viewMode === "edit"
      ? t(I18nKey.SETTINGS$EDIT_AGENT_PROFILE)
      : t(I18nKey.SETTINGS$ADD_AGENT_PROFILE);
  const editorDescription =
    viewMode === "edit" && editingProfile
      ? t(I18nKey.SETTINGS$PROFILE_LOADED, { name: editingProfile.name })
      : t(I18nKey.SETTINGS$PROFILE_SAVE_HINT);
  const isOpenHands = saveControl?.agentType !== "acp";

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

      <ProfileNameInput
        testId="agent-profile-name-input"
        value={profileName}
        onChange={setProfileName}
        isRequired
      />

      {/* Reuse the existing Agent settings form to define the agent. */}
      <AgentSettingsScreen
        key={viewMode === "edit" ? `edit-${editingProfile?.id}` : "new-profile"}
        embedded
        agentSettingsOverride={override}
        onSaveControlChange={setSaveControl}
      />

      {/* OpenHands profiles reference an LLM profile (required). */}
      {isOpenHands &&
        (llmProfiles.length > 0 ? (
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
        ) : (
          <p
            data-testid="agent-profile-no-llm"
            className="text-sm text-red-400"
          >
            {t(I18nKey.SETTINGS$AGENT_PROFILE_NO_LLM)}
          </p>
        ))}

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
          isDisabled={!isNameValid || isSaving || !saveControl?.isValid}
          aria-busy={isSaving}
        >
          {isSaving ? t(I18nKey.SETTINGS$SAVING) : t(I18nKey.BUTTON$SAVE)}
        </BrandButton>
      </div>
    </div>
  );
}
