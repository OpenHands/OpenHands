import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { RenameProfileModal } from "./rename-profile-modal";
import { DeleteProfileModal } from "./delete-profile-modal";
import { ProfilesBody } from "./profiles-body";
import { ProviderConnectionsManager } from "./provider-connections-manager";
import ProfilesService, {
  ProfileInfo,
  type SaveProfileRequest,
} from "#/api/profiles-service/profiles-service.api";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useProviderConnections } from "#/hooks/query/use-provider-connections";
import { useActivateLlmProfile } from "#/hooks/mutation/use-activate-llm-profile";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import { useCanManageOrgProfiles } from "#/hooks/use-can-manage-org-profiles";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";

interface LlmProfilesManagerProps {
  onAddProfile?: () => void;
  onEditProfile?: (profile: ProfileInfo) => void;
}

export function LlmProfilesManager({
  onAddProfile,
  onEditProfile,
}: LlmProfilesManagerProps) {
  const { t } = useTranslation("openhands");
  const { data, isLoading, error } = useLlmProfiles();
  const activateProfile = useActivateLlmProfile();
  const saveProfile = useSaveLlmProfile();
  // Cloud members are view-only; only owners/admins (and all local users) may
  // add, edit, rename, duplicate, delete, or activate profiles.
  const canManage = useCanManageOrgProfiles();
  // Provider connections exist only on the local agent-server.
  const { backend } = useActiveBackend();
  const isLocal = backend.kind === "local";
  const {
    data: connections,
    isLoading: isLoadingConnections,
    error: connectionsError,
  } = useProviderConnections();
  const [profileToRename, setProfileToRename] = useState<ProfileInfo | null>(
    null,
  );
  const [profileToDelete, setProfileToDelete] = useState<ProfileInfo | null>(
    null,
  );

  const profiles = data?.profiles ?? [];
  const active = data?.active_profile ?? null;
  const connectionList = useMemo(() => connections ?? [], [connections]);

  const connectionNamesById = useMemo(
    () => Object.fromEntries(connectionList.map((c) => [c.id, c.display_name])),
    [connectionList],
  );
  const linkedCountById = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const profile of profiles) {
      const id = profile.provider_connection_id;
      if (id) counts[id] = (counts[id] ?? 0) + 1;
    }
    return counts;
  }, [profiles]);

  const handleActivate = async (name: string) => {
    try {
      await activateProfile.mutateAsync(name);
      displaySuccessToast(t(I18nKey.SETTINGS$PROFILE_ACTIVATED, { name }));
    } catch (error) {
      console.error("Failed to activate profile:", error);
      displayErrorToast(t(I18nKey.ERROR$GENERIC));
    }
  };

  const handleEdit = (profile: ProfileInfo) => {
    onEditProfile?.(profile);
  };

  const handleDuplicate = async (profile: ProfileInfo) => {
    try {
      // Fetch the full config with encrypted secrets so the API key is
      // preserved on the duplicate (same approach as the edit flow).
      const detail = await ProfilesService.getProfile(
        profile.name,
        "encrypted",
      );

      // Find an available name: "{name}-copy", then "{name}-copy-1", etc.
      const existingNames = new Set(profiles.map((p) => p.name));
      let newName = `${profile.name}-copy`;
      let counter = 1;
      while (existingNames.has(newName)) {
        newName = `${profile.name}-copy-${counter}`;
        counter += 1;
      }

      await saveProfile.mutateAsync({
        name: newName,
        request: {
          llm: detail.config as SaveProfileRequest["llm"],
          include_secrets: true,
        },
      });

      displaySuccessToast(
        t(I18nKey.SETTINGS$PROFILE_DUPLICATED, { name: newName }),
      );
    } catch (err) {
      console.error("Failed to duplicate profile:", err);
      displayErrorToast(t(I18nKey.ERROR$GENERIC));
    }
  };

  return (
    <>
      <div className="flex flex-col gap-8">
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-base font-medium text-white">
              {t(I18nKey.SETTINGS$AVAILABLE_PROFILES)}
            </h2>
            {onAddProfile && canManage ? (
              <BrandButton
                testId="add-llm-profile"
                type="button"
                variant="secondary"
                className="ml-auto"
                onClick={onAddProfile}
              >
                {t(I18nKey.SETTINGS$ADD_LLM_PROFILE)}
              </BrandButton>
            ) : null}
          </div>

          <ProfilesBody
            isLoading={isLoading}
            loadError={error ?? null}
            profiles={profiles}
            active={active}
            canManage={canManage}
            connectionNamesById={connectionNamesById}
            onActivate={handleActivate}
            onEdit={handleEdit}
            onRename={setProfileToRename}
            onDuplicate={handleDuplicate}
            onDelete={setProfileToDelete}
            isActivating={activateProfile.isPending}
          />
        </div>

        {isLocal && canManage ? (
          <ProviderConnectionsManager
            connections={connectionList}
            linkedCountById={linkedCountById}
            isLoading={isLoadingConnections}
            loadError={connectionsError ?? null}
          />
        ) : null}
      </div>

      <RenameProfileModal
        profile={profileToRename}
        onClose={() => setProfileToRename(null)}
      />
      <DeleteProfileModal
        profile={profileToDelete}
        onClose={() => setProfileToDelete(null)}
      />
    </>
  );
}
