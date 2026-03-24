import React from "react";
import { useSearchParams } from "react-router";

const INVITATION_TOKEN_KEY = "openhands_invitation_token";

interface UseInvitationReturn {
  /** The invitation token, if present */
  invitationToken: string | null;
  /** Whether there is an active invitation */
  hasInvitation: boolean;
  /** Clear the stored invitation token */
  clearInvitation: () => void;
  /** Build OAuth state data including invitation token if present */
  buildOAuthStateData: (
    baseStateData: Record<string, string>,
  ) => Record<string, string>;
}

/**
 * Hook to manage organization invitation tokens during the login flow.
 *
 * This hook:
 * 1. Reads invitation_token from URL query params on mount
 * 2. Persists the token in localStorage (survives page refresh and works across tabs)
 * 3. Provides the token for inclusion in OAuth state
 * 4. Provides cleanup method after successful authentication
 *
 * The invitation token flow:
 * 1. User clicks invitation link → /api/invitations/accept?token=xxx
 * 2. Backend redirects to /login?invitation_token=xxx
 * 3. This hook captures token and stores in localStorage
 * 4. When user clicks login button, token is included in OAuth state
 * 5. After auth callback processes invitation, frontend clears the token
 *
 * Note: localStorage is used instead of sessionStorage to support scenarios where
 * the user opens the email verification link in a new tab/browser window.
 */
export function useInvitation(): UseInvitationReturn {
  const [searchParams, setSearchParams] = useSearchParams();
  const [invitationToken, setInvitationToken] = React.useState<string | null>(
    () => {
      // Initialize from localStorage (persists across tabs and page refreshes)
      if (typeof window !== "undefined") {
        return localStorage.getItem(INVITATION_TOKEN_KEY);
      }
      return null;
    },
  );

  // Capture invitation token from URL and persist to localStorage
  // This only runs on the login page where the hook is used
  React.useEffect(() => {
    const tokenFromUrl = searchParams.get("invitation_token");

    if (tokenFromUrl) {
      // Store in localStorage for persistence across tabs and refreshes
      localStorage.setItem(INVITATION_TOKEN_KEY, tokenFromUrl);
      setInvitationToken(tokenFromUrl);

      // Remove token from URL to clean up (prevents token exposure in browser history)
      const newSearchParams = new URLSearchParams(searchParams);
      newSearchParams.delete("invitation_token");
      setSearchParams(newSearchParams, { replace: true });
    }
  }, [searchParams, setSearchParams]);

  // Clear invitation token when invitation flow completes (success or failure)
  // These query params are set by the backend after processing the invitation
  React.useEffect(() => {
    const invitationCompleted =
      searchParams.has("invitation_success") ||
      searchParams.has("invitation_expired") ||
      searchParams.has("invitation_invalid") ||
      searchParams.has("invitation_error") ||
      searchParams.has("already_member") ||
      searchParams.has("email_mismatch");

    if (invitationCompleted) {
      localStorage.removeItem(INVITATION_TOKEN_KEY);
      setInvitationToken(null);

      // Remove invitation params from URL to clean up
      const newSearchParams = new URLSearchParams(searchParams);
      newSearchParams.delete("invitation_success");
      newSearchParams.delete("invitation_expired");
      newSearchParams.delete("invitation_invalid");
      newSearchParams.delete("invitation_error");
      newSearchParams.delete("already_member");
      newSearchParams.delete("email_mismatch");
      setSearchParams(newSearchParams, { replace: true });
    }
  }, [searchParams, setSearchParams]);

  const clearInvitation = React.useCallback(() => {
    localStorage.removeItem(INVITATION_TOKEN_KEY);
    setInvitationToken(null);
  }, []);

  const buildOAuthStateData = React.useCallback(
    (baseStateData: Record<string, string>): Record<string, string> => {
      const stateData = { ...baseStateData };

      // Include invitation token in state if present
      if (invitationToken) {
        stateData.invitation_token = invitationToken;
      }

      return stateData;
    },
    [invitationToken],
  );

  return {
    invitationToken,
    hasInvitation: invitationToken !== null,
    clearInvitation,
    buildOAuthStateData,
  };
}
