import { useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { RenameProfileModal } from "#/components/features/settings/rename-profile-modal";
import { DeleteProfileModal } from "#/components/features/settings/delete-profile-modal";
import { ProfilesBody } from "#/components/features/settings/profiles-body";
import { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useActivateLlmProfile } from "#/hooks/mutation/use-activate-llm-profile";
import { mutateWithToast } from "#/utils/mutate-with-toast";
import { extractErrorMessage } from "#/utils/extract-error-message";
import { I18nKey } from "#/i18n/declaration";

interface LlmProfilesManagerProps {
  onAddProfile?: () => void;
  onEditProfile?: (profile: LlmProfileSummary) => void;
}

export function LlmProfilesManager({
  onAddProfile,
  onEditProfile,
}: LlmProfilesManagerProps = {}) {
  const { t } = useTranslation();
  const { data, isLoading, error } = useLlmProfiles();
  const activateProfile = useActivateLlmProfile();
  const [profileToRename, setProfileToRename] =
    useState<LlmProfileSummary | null>(null);
  const [profileToDelete, setProfileToDelete] =
    useState<LlmProfileSummary | null>(null);

  const profiles = data?.profiles ?? [];
  const active = data?.active_profile ?? null;

  const handleActivate = async (name: string) => {
    await mutateWithToast(activateProfile, name, {
      success: t(I18nKey.SETTINGS$PROFILE_ACTIVATED, { name }),
      error: (err) => extractErrorMessage(err, t(I18nKey.ERROR$GENERIC)),
    }).catch(() => null);
  };

  const handleEdit = (profile: LlmProfileSummary) => {
    onEditProfile?.(profile);
  };

  return (
    <>
      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-base font-semibold text-white">
            {t(I18nKey.SETTINGS$AVAILABLE_PROFILES)}
          </h2>
          {onAddProfile ? (
            <BrandButton
              testId="add-llm-profile"
              type="button"
              variant="primary"
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
          onActivate={handleActivate}
          onEdit={handleEdit}
          onRename={setProfileToRename}
          onDelete={setProfileToDelete}
          isActivating={activateProfile.isPending}
        />
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
