import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";
import { useDeleteLlmProfile } from "#/hooks/mutation/use-delete-llm-profile";
import { mutateWithToast } from "#/utils/mutate-with-toast";
import { extractErrorMessage } from "#/utils/extract-error-message";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

interface DeleteProfileModalProps {
  profile: LlmProfileSummary | null;
  onClose: () => void;
}

export function DeleteProfileModal({
  profile,
  onClose,
}: DeleteProfileModalProps) {
  const { t } = useTranslation();
  const deleteProfile = useDeleteLlmProfile();

  if (!profile) return null;

  const handleDelete = async () => {
    const ok = await mutateWithToast(deleteProfile, profile.name, {
      success: t(I18nKey.SETTINGS$PROFILE_DELETED, { name: profile.name }),
      error: (err) => extractErrorMessage(err, t(I18nKey.ERROR$GENERIC)),
    }).catch(() => null);
    if (ok !== null) onClose();
  };

  const footer = (
    <>
      <BrandButton
        testId="delete-profile-confirm"
        type="button"
        variant="danger"
        className="grow"
        onClick={handleDelete}
        isDisabled={deleteProfile.isPending}
      >
        {deleteProfile.isPending ? (
          <LoadingSpinner size="small" />
        ) : (
          t(I18nKey.BUTTON$DELETE)
        )}
      </BrandButton>
      <BrandButton
        type="button"
        variant="secondary"
        className="grow"
        onClick={onClose}
        isDisabled={deleteProfile.isPending}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
    </>
  );

  return (
    <ApiKeyModalBase
      isOpen
      title={t(I18nKey.SETTINGS$PROFILE_DELETE_TITLE)}
      footer={footer}
    >
      <Typography.Paragraph className="text-sm break-all">
        {t(I18nKey.SETTINGS$PROFILE_DELETE_CONFIRMATION, {
          name: profile.name,
        })}
      </Typography.Paragraph>
    </ApiKeyModalBase>
  );
}
