import React from "react";
import { useSearchParams } from "react-router";
import { useResendEmailVerification } from "#/hooks/mutation/use-resend-email-verification";

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
 *   - recaptchaBlocked: boolean state for reCAPTCHA blocked error status
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
  const [recaptchaBlocked, setRecaptchaBlocked] = React.useState(false);
  const [userId, setUserId] = React.useState<string | null>(null);
  const [lastSentTimestamp, setLastSentTimestamp] = React.useState<
    number | null
  >(null);
  const [cooldownRemaining, setCooldownRemaining] = React.useState<number>(0);

  const COOLDOWN_DURATION_MS = 30 * 1000; // 30 seconds

  const formatCooldownTime = (ms: number): string => {
    const seconds = Math.ceil(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
  };

  const resendEmailVerificationMutation = useResendEmailVerification({
    onSuccess: () => {
      setLastSentTimestamp(Date.now());
    },
  });

  // Check for email verification and reCAPTCHA query parameters
  React.useEffect(() => {
    const emailVerificationRequired = searchParams.get(
      "email_verification_required",
    );
    const emailVerifiedParam = searchParams.get("email_verified");
    const duplicatedEmailParam = searchParams.get("duplicated_email");
    const recaptchaBlockedParam = searchParams.get("recaptcha_blocked");
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

    if (recaptchaBlockedParam === "true") {
      setRecaptchaBlocked(true);
      searchParams.delete("recaptcha_blocked");
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
    recaptchaBlocked,
    userId,
    resendEmailVerification: resendEmailVerificationMutation.mutate,
    isResendingVerification: resendEmailVerificationMutation.isPending,
    isCooldownActive,
    cooldownRemaining,
    formattedCooldownTime,
  };
}
