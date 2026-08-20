import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ProfileNameInput } from "./profile-name-input";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import { useRenameLlmProfile } from "#/hooks/mutation/use-rename-llm-profile";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { isProfileNameValid } from "#/utils/derive-profile-name";

interface RenameProfileModalProps {
  profile: ProfileInfo | null;
  onClose: () => void;
}

export function RenameProfileModal({
  profile,
  onClose,
}: RenameProfileModalProps) {
  const { t } = useTranslation("openhands");
  const [newName, setNewName] = useState("");
  const renameProfile = useRenameLlmProfile();
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setNewName(profile?.name ?? "");
  }, [profile?.name]);

  if (!profile) return null;

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
      displaySuccessToast(
        t(I18nKey.SETTINGS$PROFILE_RENAMED, { name: newName }),
      );
      onClose();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC);
      displayErrorToast(message);
    }
  };

  // Handle close only if not pending to prevent inconsistent state
  const handleClose = () => {
    if (!renameProfile.isPending) {
      onClose();
    }
  };

  const footer = (
    <>
      <BrandButton
        type="button"
        variant="tertiary"
        onClick={handleClose}
        isDisabled={renameProfile.isPending}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
      <BrandButton
        testId="rename-profile-submit"
        type="button"
        variant="primary"
        onClick={handleSubmit}
        isDisabled={renameProfile.isPending || !isValid}
      >
        {renameProfile.isPending ? (
          <LoadingSpinner size="small" />
        ) : (
          t(I18nKey.BUTTON$RENAME)
        )}
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
            if (e.key === "Enter" && !renameProfile.isPending && isValid) {
              handleSubmit();
            }
          }}
        />
      </div>
    </ApiKeyModalBase>
  );
}
