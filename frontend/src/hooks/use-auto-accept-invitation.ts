import React from "react";
import { AxiosError } from "axios";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useIsAuthed } from "#/hooks/query/use-is-authed";
import { useIsOnIntermediatePage } from "#/hooks/use-is-on-intermediate-page";
import { useInvitation } from "#/hooks/use-invitation";
import {
  useAcceptInvitation,
  getInvitationErrorCode,
} from "#/hooks/mutation/use-accept-invitation";
import { useSwitchOrganization } from "#/hooks/mutation/use-switch-organization";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

/**
 * Accept a pending invitation token automatically once the user is
 * authenticated.
 *
 * Clicking the invitation link is the user's consent — an extra
 * confirm/cancel dialog added no security (the token is bound to the
 * invited email) and its cancel didn't actually decline anything, so the
 * token is simply submitted as soon as a session exists and the outcome is
 * reported via a toast.
 */
export function useAutoAcceptInvitation() {
  const { t } = useTranslation();
  const { data: isAuthed } = useIsAuthed();
  const isOnIntermediatePage = useIsOnIntermediatePage();
  const { invitationToken, clearInvitation } = useInvitation();
  const { mutate: acceptInvitation } = useAcceptInvitation();
  const { mutate: switchOrganization } = useSwitchOrganization();
  const attemptedTokenRef = React.useRef<string | null>(null);

  React.useEffect(() => {
    if (!isAuthed || !invitationToken || isOnIntermediatePage) return;
    if (attemptedTokenRef.current === invitationToken) return;
    attemptedTokenRef.current = invitationToken;

    acceptInvitation(
      { token: invitationToken },
      {
        onSuccess: (data) => {
          displaySuccessToast(
            t(I18nKey.ORG$INVITATION_ACCEPTED_SUCCESS, {
              orgName: data.org_name,
            }),
          );
          switchOrganization({
            orgId: data.org_id,
            orgName: data.org_name,
            isPersonal: false,
          });
        },
        onError: (error) => {
          const errorCode = getInvitationErrorCode(
            error as AxiosError<{ detail: string }>,
          );
          switch (errorCode) {
            case "already_member":
              // The invitation was already applied (e.g. accepted on the
              // user's behalf at sign-in) — success from their perspective.
              displaySuccessToast(t(I18nKey.ORG$ALREADY_MEMBER));
              break;
            case "invitation_expired":
              displayErrorToast(t(I18nKey.ORG$INVITATION_EXPIRED));
              break;
            case "email_mismatch":
              displayErrorToast(t(I18nKey.ORG$INVITATION_EMAIL_MISMATCH));
              break;
            case "invitation_invalid":
              displayErrorToast(t(I18nKey.ORG$INVITATION_INVALID));
              break;
            default:
              displayErrorToast(t(I18nKey.ORG$INVITATION_ACCEPT_ERROR));
          }
        },
        onSettled: () => {
          clearInvitation();
        },
      },
    );
  }, [
    isAuthed,
    invitationToken,
    isOnIntermediatePage,
    acceptInvitation,
    switchOrganization,
    clearInvitation,
    t,
  ]);
}
