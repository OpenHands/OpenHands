import React from "react";
import { useSearchParams } from "react-router";

/**
 * Hook to handle email verification logic from URL query parameters.
 * Manages the email verification modal state and email verified state
 * based on query parameters in the URL.
 *
 * @returns An object containing:
 *   - emailVerificationModalOpen: boolean state for modal visibility
 *   - setEmailVerificationModalOpen: function to control modal visibility
 *   - emailVerified: boolean state for email verification status
 *   - setEmailVerified: function to control email verification status
 *   - hasDuplicatedEmail: boolean state for duplicate email error status
 */
export function useEmailVerification() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [emailVerificationModalOpen, setEmailVerificationModalOpen] =
    React.useState(false);
  const [emailVerified, setEmailVerified] = React.useState(false);
  const [hasDuplicatedEmail, setHasDuplicatedEmail] = React.useState(false);

  // Check for email verification query parameters
  React.useEffect(() => {
    const emailVerificationRequired = searchParams.get(
      "email_verification_required",
    );
    const emailVerifiedParam = searchParams.get("email_verified");
    const duplicatedEmailParam = searchParams.get("duplicated_email");
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

    // Clean up the URL by removing parameters if any were found
    if (shouldUpdate) {
      setSearchParams(searchParams, { replace: true });
    }
  }, [searchParams, setSearchParams]);

  return {
    emailVerificationModalOpen,
    setEmailVerificationModalOpen,
    emailVerified,
    setEmailVerified,
    hasDuplicatedEmail,
  };
}
