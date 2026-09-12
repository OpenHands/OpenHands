import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ProfileNameInput } from "./profile-name-input";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import { useRenameLlmProfile } from "#/hooks/mutation/use-rename-llm-profile";
import { useActivateLlmProfile } from "#/hooks/mutation/use-activate-llm-profile";
import { shouldReapplyProfileAfterSave } from "./llm-settings-local-view";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { I18nKey } from "#/i18n/declaration";
import { isProfileNameValid } from "#/utils/derive-profile-name";

interface RenameProfileModalProps {
  profile: ProfileInfo | null;
  /** Name of the currently active profile, if any — used to decide whether
   * renaming this profile also needs to reapply it (see handleSubmit). */
  activeProfileName?: string | null;
  onClose: () => void;
}

export function RenameProfileModal({
  profile,
  activeProfileName = null,
  onClose,
}: RenameProfileModalProps) {
  const { t } = useTranslation("openhands");
  const [newName, setNewName] = useState("");
  const renameProfile = useRenameLlmProfile();
  const activateProfile = useActivateLlmProfile();
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setNewName(profile?.name ?? "");
  }, [profile?.name]);

  if (!profile) return null;

  const isPending = renameProfile.isPending || activateProfile.isPending;
  const isValid = isProfileNameValid(newName, { isRequired: true });
  const isUnchanged = newName === profile.name;

  const handleSubmit = async () => {
    if (!isValid) {
      displayErrorToast(t(I18nKey.SETTINGS$PROFILE_NAME_RULE));
      return;
    }
    if (isUnchanged) {
      onClose();
      return;
    }

    try {
      await renameProfile.mutateAsync({ name: profile.name, newName });

      // Conversation start uses agent_settings.llm, not the profile row
      // directly — renaming the active profile leaves that pointer stale
      // (still the old name) until it's reapplied under the new one. Mirrors
      // the same guard the Edit flow already applies after a rename there.
      if (
        shouldReapplyProfileAfterSave({
          activeProfileName,
          originalName: profile.name,
          savedName: newName,
        })
      ) {
        await activateProfile.mutateAsync(newName);
      }

      displaySuccessToast(
        t(I18nKey.SETTINGS$PROFILE_RENAMED, { name: newName }),
      );
      onClose();
    } catch (error) {
      displayErrorToast(getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)));
    }
  };

  // Handle close only if not pending to prevent inconsistent state
  const handleClose = () => {
    if (!isPending) {
      onClose();
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
        testId="rename-profile-submit"
        type="button"
        variant="primary"
        onClick={handleSubmit}
        isDisabled={isPending || !isValid}
      >
        {isPending ? <LoadingSpinner size="small" /> : t(I18nKey.BUTTON$RENAME)}
      </BrandButton>
    </>
  );

  return (
    <ApiKeyModalBase
      isOpen
      title={t(I18nKey.SETTINGS$PROFILE_RENAME_TITLE)}
      footer={footer}
      onClose={handleClose}
      initialFocusRef={inputRef}
    >
      <div data-testid="rename-profile-modal" className="flex flex-col gap-3">
        <ProfileNameInput
          ref={inputRef}
          testId="rename-profile-input"
          ruleTestId="rename-profile-rule"
          value={newName}
          onChange={setNewName}
          isRequired
          onKeyDown={(e) => {
            if (e.key === "Enter" && !isPending && isValid) {
              handleSubmit();
            }
          }}
        />
      </div>
    </ApiKeyModalBase>
  );
}
