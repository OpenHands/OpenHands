import React from "react";
import { useSearchParams } from "react-router";
import { AxiosError } from "axios";
import { useMutation } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { emailService } from "#/api/email-service/email-service.api";
import {
  displaySuccessToast,
  displayErrorToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { ResendEmailVerificationParams } from "#/api/email-service/email.types";

/**
 * Hook to handle email verification logic from URL query parameters.
 * Manages the email verification modal state and email verified state
 * based on query parameters in the URL.
 * Also provides functionality to resend email verification.
 *
 * @returns An object containing:
 *   - emailVerificationModalOpen: boolean state for modal visibility
 *   - setEmailVerificationModalOpen: function to control modal visibility
 *   - emailVerified: boolean state for email verification status
 *   - setEmailVerified: function to control email verification status
 *   - hasDuplicatedEmail: boolean state for duplicate email error status
 *   - userId: string | null for the user ID from the redirect URL
 *   - resendEmailVerification: function to resend verification email
 *   - isResendingVerification: boolean indicating if resend is in progress
 *   - isCooldownActive: boolean indicating if cooldown is currently active
 *   - cooldownRemaining: number of milliseconds remaining in cooldown
 *   - formattedCooldownTime: string formatted as "M:SS" for display
 */
export function useEmailVerification() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [emailVerificationModalOpen, setEmailVerificationModalOpen] =
    React.useState(false);
  const [emailVerified, setEmailVerified] = React.useState(false);
  const [hasDuplicatedEmail, setHasDuplicatedEmail] = React.useState(false);
  const [userId, setUserId] = React.useState<string | null>(null);
  const [lastSentTimestamp, setLastSentTimestamp] = React.useState<
    number | null
  >(null);
  const [cooldownRemaining, setCooldownRemaining] = React.useState<number>(0);
  const { t } = useTranslation();

  const COOLDOWN_DURATION_MS = 2 * 60 * 1000; // 2 minutes

  const formatCooldownTime = (ms: number): string => {
    const seconds = Math.ceil(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
  };

  const resendEmailVerificationMutation = useMutation({
    mutationFn: (params: ResendEmailVerificationParams) =>
      emailService.resendEmailVerification(params),
    onSuccess: () => {
      setLastSentTimestamp(Date.now());
      displaySuccessToast(t(I18nKey.SETTINGS$VERIFICATION_EMAIL_SENT));
    },
    onError: (error: AxiosError) => {
      // Check if it's a rate limit error (429)
      if (error.response?.status === 429) {
        // FastAPI returns errors in { detail: "..." } format
        const errorData = error.response.data as
          | { detail?: string }
          | undefined;

        const rateLimitMessage =
          errorData?.detail ||
          retrieveAxiosErrorMessage(error) ||
          t(I18nKey.SETTINGS$FAILED_TO_RESEND_VERIFICATION);

        displayErrorToast(rateLimitMessage);
      } else {
        // For other errors, show the generic error message
        displayErrorToast(t(I18nKey.SETTINGS$FAILED_TO_RESEND_VERIFICATION));
      }
    },
  });

  // Check for email verification query parameters
  React.useEffect(() => {
    const emailVerificationRequired = searchParams.get(
      "email_verification_required",
    );
    const emailVerifiedParam = searchParams.get("email_verified");
    const duplicatedEmailParam = searchParams.get("duplicated_email");
    const userIdParam = searchParams.get("user_id");
    let shouldUpdate = false;

    if (emailVerificationRequired === "true") {
      setEmailVerificationModalOpen(true);
      searchParams.delete("email_verification_required");
      shouldUpdate = true;
    }

    if (emailVerifiedParam === "true") {
      setEmailVerified(true);
      searchParams.delete("email_verified");
      shouldUpdate = true;
    }

    if (duplicatedEmailParam === "true") {
      setHasDuplicatedEmail(true);
      searchParams.delete("duplicated_email");
      shouldUpdate = true;
    }

    if (userIdParam) {
      setUserId(userIdParam);
      searchParams.delete("user_id");
      shouldUpdate = true;
    }

    // Clean up the URL by removing parameters if any were found
    if (shouldUpdate) {
      setSearchParams(searchParams, { replace: true });
    }
  }, [searchParams, setSearchParams]);

  // Update cooldown remaining time
  React.useEffect(() => {
    if (lastSentTimestamp === null) {
      setCooldownRemaining(0);
      return undefined;
    }

    let timeoutId: NodeJS.Timeout | null = null;

    const updateCooldown = () => {
      const elapsed = Date.now() - lastSentTimestamp!;
      const remaining = Math.max(0, COOLDOWN_DURATION_MS - elapsed);
      setCooldownRemaining(remaining);

      if (remaining > 0) {
        // Update every second while cooldown is active
        timeoutId = setTimeout(updateCooldown, 1000);
      }
    };

    updateCooldown();

    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [lastSentTimestamp, COOLDOWN_DURATION_MS]);

  const isCooldownActive = cooldownRemaining > 0;
  const formattedCooldownTime = formatCooldownTime(cooldownRemaining);

  return {
    emailVerificationModalOpen,
    setEmailVerificationModalOpen,
    emailVerified,
    setEmailVerified,
    hasDuplicatedEmail,
    userId,
    resendEmailVerification: resendEmailVerificationMutation.mutate,
    isResendingVerification: resendEmailVerificationMutation.isPending,
    isCooldownActive,
    cooldownRemaining,
    formattedCooldownTime,
  };
}
