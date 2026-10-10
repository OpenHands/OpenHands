import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { AgentProfilesBody } from "./agent-profiles-body";
import { DeleteAgentProfileModal } from "./delete-agent-profile-modal";
import {
  allowsAgentSettingsLaunch,
  getAgentProfileLlmDrift,
} from "./agent-profile-llm-drift";
import AgentProfilesService, {
  type AgentProfileSummary,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import {
  AGENT_PROFILES_QUERY_KEYS,
  AGENT_PROFILES_RETRY_OPTIONS,
} from "#/hooks/query/query-keys";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useActivateAgentProfile } from "#/hooks/mutation/use-activate-agent-profile";
import { useCanManageOrgProfiles } from "#/hooks/use-can-manage-org-profiles";
import { displaySuccessToast } from "#/utils/custom-toast-handlers";
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
  const { backend, orgId } = useActiveBackend();
  // A home launch only ignores the active profile's pinned `llm_profile_ref`
  // on local backends (#16193/#16539), so only local rows can drift — skip the
  // fetch entirely on cloud, where the ref is what launches.
  const isLocal = backend.kind === "local";
  const { data: llmProfilesData } = useLlmProfiles({ enabled: isLocal });
  const activateProfile = useActivateAgentProfile();
  // Cloud members are view-only; only owners/admins (and all local users) may
  // add, edit, delete, or activate agent profiles (org-scoped, same permission
  // as LLM profiles). Mirrors LlmProfilesManager.
  const canManage = useCanManageOrgProfiles();
  const [profileToDelete, setProfileToDelete] =
    useState<AgentProfileSummary | null>(null);

  const profiles = data?.profiles ?? [];
  const activeId = data?.active_agent_profile_id ?? null;
  const activeProfile =
    profiles.find((profile) => !!profile.id && profile.id === activeId) ?? null;
  const localActiveLlmProfile = isLocal
    ? (llmProfilesData?.active_profile ?? null)
    : null;
  // The launch re-reads the profile before downgrading and keeps a
  // secret-scoped profile on its pinned ref, so the summary alone cannot tell
  // whether the active profile drifts. Read the detail only when it might.
  const activeProfileMayDrift =
    !!activeProfile &&
    getAgentProfileLlmDrift(activeProfile, true, localActiveLlmProfile) !==
      null;
  const { data: activeProfileDetail } = useQuery({
    queryKey: AGENT_PROFILES_QUERY_KEYS.detail(
      backend.id,
      orgId,
      activeProfile?.name ?? "",
    ),
    queryFn: () => AgentProfilesService.getProfile(activeProfile!.name),
    ...AGENT_PROFILES_RETRY_OPTIONS,
    enabled: activeProfileMayDrift,
    meta: { disableToast: true },
  });
  const activeLlmProfile = allowsAgentSettingsLaunch(
    activeProfileDetail?.profile,
  )
    ? localActiveLlmProfile
    : null;

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
          activeLlmProfile={activeLlmProfile}
          canManage={canManage}
          onActivate={handleActivate}
          onEdit={(profile) => onEditProfile?.(profile)}
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
