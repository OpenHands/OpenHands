import { useRef } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { type AgentProfileSummary } from "#/api/agent-profiles-service/agent-profiles-service.api";
import { useDeleteAgentProfile } from "#/hooks/mutation/use-delete-agent-profile";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";

interface DeleteAgentProfileModalProps {
  profile: AgentProfileSummary | null;
  onClose: () => void;
}

export function DeleteAgentProfileModal({
  profile,
  onClose,
}: DeleteAgentProfileModalProps) {
  const { t } = useTranslation("openhands");
  const deleteProfile = useDeleteAgentProfile();
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
        testId="delete-agent-profile-confirm"
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
