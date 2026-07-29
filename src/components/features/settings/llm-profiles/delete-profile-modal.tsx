import { useRef } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import { useDeleteLlmProfile } from "#/hooks/mutation/use-delete-llm-profile";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";

interface DeleteProfileModalProps {
  profile: ProfileInfo | null;
  onClose: () => void;
}

export function DeleteProfileModal({
  profile,
  onClose,
}: DeleteProfileModalProps) {
  const { t } = useTranslation("openhands");
  const deleteProfile = useDeleteLlmProfile();
  const cancelButtonRef = useRef<HTMLButtonElement>(null);

  if (!profile) return null;

  const handleDelete = async () => {
    try {
      await deleteProfile.mutateAsync(profile.name);
      displaySuccessToast(
        t(I18nKey.SETTINGS$PROFILE_DELETED, { name: profile.name }),
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
    if (!deleteProfile.isPending) {
      onClose();
    }
  };

  const footer = (
    <>
      <BrandButton
        ref={cancelButtonRef}
        type="button"
        variant="tertiary"
        onClick={handleClose}
        isDisabled={deleteProfile.isPending}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
      <BrandButton
        testId="delete-profile-confirm"
        type="button"
        variant="danger"
        onClick={handleDelete}
        isDisabled={deleteProfile.isPending}
        aria-busy={deleteProfile.isPending}
      >
        {deleteProfile.isPending ? (
          <>
            <LoadingSpinner size="small" />
            <span className="sr-only">{t(I18nKey.BUTTON$DELETE)}</span>
          </>
        ) : (
          t(I18nKey.BUTTON$DELETE)
        )}
      </BrandButton>
    </>
  );

  return (
    <ApiKeyModalBase
      isOpen
      title={t(I18nKey.SETTINGS$PROFILE_DELETE_TITLE)}
      footer={footer}
      onClose={handleClose}
      initialFocusRef={cancelButtonRef}
    >
      <p className="text-sm break-all">
        {t(I18nKey.SETTINGS$PROFILE_DELETE_CONFIRMATION, {
          name: profile.name,
        })}
      </p>
    </ApiKeyModalBase>
  );
}
