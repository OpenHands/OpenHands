import { AxiosError } from "axios";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import {
  useAcceptInvitation,
  getInvitationErrorCode,
} from "#/hooks/mutation/use-accept-invitation";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

interface InvitationAcceptModalProps {
  token: string;
  onClose: () => void;
  onSuccess: (payload: {
    orgId: string;
    orgName: string;
    isPersonal: boolean;
  }) => void;
}

export function InvitationAcceptModal({
  token,
  onClose,
  onSuccess,
}: InvitationAcceptModalProps) {
  const { t } = useTranslation();
  const { mutate: acceptInvitation, isPending } = useAcceptInvitation();

  const getErrorMessage = (errorCode: string | null): string => {
    switch (errorCode) {
      case "invitation_expired":
        return t(I18nKey.ORG$INVITATION_EXPIRED);
      case "invitation_invalid":
        return t(I18nKey.ORG$INVITATION_INVALID);
      case "already_member":
        return t(I18nKey.ORG$ALREADY_MEMBER);
      case "email_mismatch":
        return t(I18nKey.ORG$INVITATION_EMAIL_MISMATCH);
      default:
        return t(I18nKey.ORG$INVITATION_ACCEPT_ERROR);
    }
  };

  const handleAccept = () => {
    acceptInvitation(
      { token },
      {
        onSuccess: (data) => {
          displaySuccessToast(
            t(I18nKey.ORG$INVITATION_ACCEPTED_SUCCESS, {
              orgName: data.org_name,
            }),
          );
          onSuccess({
            orgId: data.org_id,
            orgName: data.org_name,
            isPersonal: false,
          });
        },
        onError: (error) => {
          const errorCode = getInvitationErrorCode(
            error as AxiosError<{ detail: string }>,
          );
          displayErrorToast(getErrorMessage(errorCode));
          onClose();
        },
      },
    );
  };

  return (
    <ModalBackdrop onClose={onClose} aria-label="Accept invitation">
      <div
        data-testid="invitation-accept-modal"
        className="bg-base-secondary p-6 rounded-xl flex flex-col gap-4 border border-tertiary"
        style={{ width: "500px" }}
      >
        <h3 className="text-xl font-bold">
          {t(I18nKey.ORG$INVITATION_ACCEPT_TITLE)}
        </h3>
        <p className="text-sm text-gray-300">
          {t(I18nKey.ORG$INVITATION_ACCEPT_DESCRIPTION)}
        </p>
        <div className="w-full flex gap-2 mt-2">
          <BrandButton
            testId="accept-invitation-button"
            type="button"
            variant="primary"
            className="grow flex items-center justify-center"
            onClick={handleAccept}
            isDisabled={isPending}
          >
            {isPending ? (
              <LoadingSpinner size="small" />
            ) : (
              t(I18nKey.BUTTON$CONFIRM)
            )}
          </BrandButton>
          <BrandButton
            testId="cancel-invitation-button"
            type="button"
            variant="secondary"
            className="grow"
            onClick={onClose}
            isDisabled={isPending}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}
