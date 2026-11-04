import { useRef } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { useDeleteMetaProfile } from "#/hooks/mutation/use-delete-meta-profile";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";

interface DeleteMetaProfileModalProps {
  name: string | null;
  onClose: () => void;
}

export function DeleteMetaProfileModal({
  name,
  onClose,
}: DeleteMetaProfileModalProps) {
  const { t } = useTranslation("openhands");
  const deleteMetaProfile = useDeleteMetaProfile();
  const cancelButtonRef = useRef<HTMLButtonElement>(null);

  if (!name) return null;

  const handleDelete = async () => {
    try {
      await deleteMetaProfile.mutateAsync(name);
      displaySuccessToast(t(I18nKey.SETTINGS$META_PROFILE_DELETED, { name }));
      onClose();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC);
      displayErrorToast(message);
    }
  };

  const handleClose = () => {
    if (!deleteMetaProfile.isPending) {
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
        isDisabled={deleteMetaProfile.isPending}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
      <BrandButton
        testId="delete-meta-profile-confirm"
        type="button"
        variant="danger"
        onClick={handleDelete}
        isDisabled={deleteMetaProfile.isPending}
        aria-busy={deleteMetaProfile.isPending}
      >
        {deleteMetaProfile.isPending ? (
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
      title={t(I18nKey.SETTINGS$META_PROFILE_DELETE_TITLE)}
      footer={footer}
      onClose={handleClose}
      initialFocusRef={cancelButtonRef}
    >
      <p className="text-sm break-all">
        {t(I18nKey.SETTINGS$META_PROFILE_DELETE_CONFIRMATION, { name })}
      </p>
    </ApiKeyModalBase>
  );
}
