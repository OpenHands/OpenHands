import { useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { AgentProfilesBody } from "./agent-profiles-body";
import { DeleteAgentProfileModal } from "./delete-agent-profile-modal";
import AgentProfilesService, {
  type AgentProfileSaveInput,
  type AgentProfileSummary,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import { useActivateAgentProfile } from "#/hooks/mutation/use-activate-agent-profile";
import { useSaveAgentProfile } from "#/hooks/mutation/use-save-agent-profile";
import { useCanManageOrgProfiles } from "#/hooks/use-can-manage-org-profiles";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";

interface AgentProfilesManagerProps {
  onAddProfile?: () => void;
  onEditProfile?: (profile: AgentProfileSummary) => void;
}

export function AgentProfilesManager({
  onAddProfile,
  onEditProfile,
}: AgentProfilesManagerProps) {
  const { t } = useTranslation("openhands");
  const { data, isLoading, error } = useAgentProfiles();
  const activateProfile = useActivateAgentProfile();
  const saveProfile = useSaveAgentProfile();
  // Cloud members are view-only; only owners/admins (and all local users) may
  // manage agent profiles, as with LLM profiles.
  const canManage = useCanManageOrgProfiles();
  const [profileToDelete, setProfileToDelete] =
    useState<AgentProfileSummary | null>(null);

  const profiles = data?.profiles ?? [];
  const activeId = data?.active_agent_profile_id ?? null;

  const handleActivate = async (profile: AgentProfileSummary) => {
    if (!profile.id) return;
    try {
      await activateProfile.mutateAsync(profile.id);
      displaySuccessToast(
        t(I18nKey.SETTINGS$PROFILE_ACTIVATED, { name: profile.name }),
      );
    } catch (error) {
      // The global mutation error toast reports the failure; just log here.
      console.error("Failed to activate agent profile:", error);
    }
  };

  const handleDuplicate = async (profile: AgentProfileSummary) => {
    const existingNames = new Set(profiles.map(({ name }) => name));
    let suffix = "-copy";
    let copyName = `${profile.name.slice(0, 64 - suffix.length)}${suffix}`;
    for (let index = 1; existingNames.has(copyName); index += 1) {
      suffix = `-copy-${index}`;
      copyName = `${profile.name.slice(0, 64 - suffix.length)}${suffix}`;
    }

    try {
      const { profile: source } = await AgentProfilesService.getProfile(
        profile.name,
        "encrypted",
      );
      const {
        id: _id,
        name: _name,
        revision: _revision,
        schema_version: _schemaVersion,
        ...fields
      } = source;
      await saveProfile.mutateAsync({
        name: copyName,
        profile: fields as AgentProfileSaveInput,
      });
      displaySuccessToast(
        t(I18nKey.SETTINGS$PROFILE_DUPLICATED, { name: copyName }),
      );
    } catch {
      displayErrorToast(t(I18nKey.ERROR$GENERIC));
    }
  };

  return (
    <>
      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-base font-medium text-contrast">
            {t(I18nKey.SETTINGS$AVAILABLE_PROFILES)}
          </h2>
          {onAddProfile && canManage ? (
            <BrandButton
              testId="add-agent-profile"
              type="button"
              variant="secondary"
              className="ml-auto"
              onClick={onAddProfile}
            >
              {t(I18nKey.SETTINGS$ADD_AGENT_PROFILE)}
            </BrandButton>
          ) : null}
        </div>

        <AgentProfilesBody
          isLoading={isLoading}
          loadError={error ?? null}
          profiles={profiles}
          activeId={activeId}
          canManage={canManage}
          onActivate={handleActivate}
          onEdit={(profile) => onEditProfile?.(profile)}
          onDuplicate={handleDuplicate}
          onDelete={setProfileToDelete}
          isActivating={activateProfile.isPending}
        />
      </div>

      <DeleteAgentProfileModal
        profile={profileToDelete}
        onClose={() => setProfileToDelete(null)}
      />
    </>
  );
}
