import React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import DatabricksAuthService from "#/api/databricks-auth-service/databricks-auth-service.api";
import {
  useDatabricksAuthStatus,
  useDatabricksLogout,
} from "#/hooks/query/use-databricks-auth-status";
import { cn } from "#/utils/utils";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

/** Origin used for OAuth popups — must match the XHR origin so session cookies align. */
const getOAuthOrigin = () => window.location.origin;

interface DatabricksSignInButtonProps {
  /**
   * Whether the Databricks provider is the active selection. When ``false``
   * the button renders ``null`` so it doesn't clutter the settings tab for
   * users on other providers.
   */
  isActive: boolean;
  className?: string;
  /**
   * OAuth App Client ID from the user's settings.
   * When provided, the button uses it (via ``/prepare``) instead of the server
   * env var, so users don't need to set ``DATABRICKS_U2M_CLIENT_ID`` server-side.
   */
  u2mClientId?: string | null;
  /** Host URL from user settings, forwarded to the /prepare call. */
  u2mHost?: string | null;
  /** OAuth App Client Secret (only for confidential apps). */
  u2mClientSecret?: string | null;
  /**
   * Redirect URI registered with the Databricks OAuth app.
   * Must exactly match what is registered — forwarded to the /prepare call so
   * the backend uses it instead of its default ``localhost:{PORT}`` fallback.
   * Example: ``http://localhost:3000/auth/databricks/callback``
   */
  u2mRedirectUri?: string | null;
}

/**
 * "Sign in with Databricks" affordance shown in the LLM settings tab.
 *
 * Opens the OAuth flow in a **new tab** so the user doesn't lose their
 * settings page. Polls ``GET /auth/databricks/status`` every 2 s after the
 * popup opens and updates the UI automatically when authentication completes.
 *
 * Renders three states:
 * 1. Not configured — ``configured=false``: show a hint to enter a Client ID.
 * 2. Configured but not signed in — show a "Sign in with Databricks" CTA.
 * 3. Signed in — show "Signed in to {host}" and a compact Sign out button.
 */
export function DatabricksSignInButton({
  isActive,
  className,
  u2mClientId,
  u2mHost,
  u2mClientSecret,
  u2mRedirectUri,
}: DatabricksSignInButtonProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { data: status, isLoading } = useDatabricksAuthStatus({
    enabled: isActive,
  });
  const logout = useDatabricksLogout();
  const [isPreparing, setIsPreparing] = React.useState(false);

  // Poll status after the popup is opened so the UI flips to "Signed in"
  // automatically when the OAuth callback completes.
  const pollingRef = React.useRef<ReturnType<typeof setInterval> | null>(null);

  const startPolling = React.useCallback(() => {
    if (pollingRef.current) return; // already polling
    const INTERVAL_MS = 2000;
    const TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes
    const deadline = Date.now() + TIMEOUT_MS;

    pollingRef.current = setInterval(() => {
      if (Date.now() > deadline) {
        clearInterval(pollingRef.current!);
        pollingRef.current = null;
        return;
      }
      // Force-invalidate so the status hook refetches immediately.
      queryClient.invalidateQueries({ queryKey: ["databricks-auth-status"] });
    }, INTERVAL_MS);
  }, [queryClient]);

  const stopPolling = React.useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  // Stop polling once we detect authentication.
  React.useEffect(() => {
    if (status?.authenticated) stopPolling();
  }, [status?.authenticated, stopPolling]);

  // Clean up interval on unmount.
  React.useEffect(() => () => stopPolling(), [stopPolling]);

  if (!isActive) return null;
  if (isLoading) return null;

  // Configured if the user has entered a client_id in settings OR if the
  // server has DATABRICKS_U2M_CLIENT_ID env var set.
  const isConfigured = !!u2mClientId?.trim() || status?.configured;

  if (!isConfigured) {
    return (
      <p
        data-testid="databricks-u2m-not-configured"
        className="text-xs text-neutral-400"
      >
        {t("SETTINGS$DATABRICKS_U2M_NOT_CONFIGURED", {
          defaultValue:
            "Enter an OAuth App Client ID above, then click Sign in.",
        })}
      </p>
    );
  }

  // Normalise a host URL to bare hostname for comparison, so trailing slashes
  // and http/https differences don't cause false mismatches.
  const normaliseHost = (url: string) => {
    try {
      return new URL(url.trim()).hostname.toLowerCase();
    } catch {
      return url
        .trim()
        .toLowerCase()
        .replace(/^https?:\/\//, "")
        .replace(/\/$/, "");
    }
  };

  // Detect if the user is signed in to a DIFFERENT workspace than what they
  // have configured in the settings form.  In that case we still show the
  // sign-in button so they can initiate a fresh auth flow — the backend
  // /prepare endpoint will automatically clear the stale token.
  const configuredHost = u2mHost?.trim() ?? "";
  const signedInHost = status?.host ?? "";
  const hostMismatch =
    status?.authenticated &&
    configuredHost !== "" &&
    signedInHost !== "" &&
    normaliseHost(configuredHost) !== normaliseHost(signedInHost);

  const handleSignIn = async () => {
    const backendOrigin = getOAuthOrigin();

    if (u2mClientId?.trim()) {
      // Open a blank tab SYNCHRONOUSLY (inside the user-gesture tick) so
      // browsers don't treat it as a popup and block it.  We redirect the
      // tab to the real OAuth URL once the /prepare response arrives.
      const popup = window.open("about:blank", "_blank");
      setIsPreparing(true);
      try {
        const initiateRelativePath = await DatabricksAuthService.prepare(
          u2mClientId.trim(),
          u2mHost?.trim() || "",
          u2mClientSecret || null,
          u2mRedirectUri?.trim() || null,
        );
        const initiateUrl = `${backendOrigin}${initiateRelativePath}`;
        if (popup) {
          popup.location.href = initiateUrl;
        } else {
          // Fallback: popup was blocked after all — try a plain open.
          window.open(initiateUrl, "_blank");
        }
        startPolling();
      } catch {
        popup?.close();
        displayErrorToast(
          t("SETTINGS$DATABRICKS_SIGN_IN_FAILED", {
            defaultValue:
              "Could not start Databricks sign-in. Check your Client ID, workspace URL, and redirect URI, then try again.",
          }),
        );
      } finally {
        setIsPreparing(false);
      }
    } else {
      // Env-var path — open initiate URL directly (no async before open).
      window.open(
        `${backendOrigin}${DatabricksAuthService.INITIATE_URL}`,
        "_blank",
      );
      startPolling();
    }
  };

  const handleSignOut = () => {
    logout.mutate();
  };

  const baseClasses = cn(
    "inline-flex items-center gap-2 rounded-md px-3 py-2 text-sm",
    "border border-[#717888] bg-tertiary hover:bg-tertiary/80",
    "focus:outline-none focus:ring-2 focus:ring-offset-2",
    "disabled:opacity-60 disabled:cursor-not-allowed",
    className,
  );

  if (status?.authenticated && !hostMismatch) {
    return (
      <div
        data-testid="databricks-signed-in"
        className="flex flex-col gap-1 text-sm"
      >
        <span className="text-xs text-neutral-400">
          {t("SETTINGS$DATABRICKS_SIGNED_IN_PREFIX", {
            defaultValue: "Signed in to",
          })}{" "}
          <span className="font-mono">{status?.host}</span>
        </span>
        <button
          type="button"
          data-testid="databricks-sign-out-button"
          className={baseClasses}
          onClick={handleSignOut}
          disabled={logout.isPending}
        >
          {t("SETTINGS$DATABRICKS_SIGN_OUT", {
            defaultValue: "Sign out of Databricks",
          })}
        </button>
      </div>
    );
  }

  // Signed in to a different workspace than what is configured — show a
  // warning and allow the user to sign in to the configured workspace.  The
  // backend /prepare call will auto-clear the stale token when it detects the
  // host change, so no explicit sign-out is required.
  if (status?.authenticated && hostMismatch) {
    return (
      <div
        data-testid="databricks-host-mismatch"
        className="flex flex-col gap-2 text-sm"
      >
        <span className="text-xs text-amber-400">
          {t("SETTINGS$DATABRICKS_HOST_MISMATCH", {
            defaultValue: "Currently signed in to",
          })}{" "}
          <span className="font-mono">{status?.host}</span>
          {". "}
          {t("SETTINGS$DATABRICKS_HOST_MISMATCH_HINT", {
            defaultValue:
              "Sign in below to switch to the configured workspace.",
          })}
        </span>
        <button
          type="button"
          data-testid="databricks-sign-in-button"
          className={baseClasses}
          onClick={handleSignIn}
          disabled={isPreparing}
        >
          {isPreparing
            ? t("SETTINGS$DATABRICKS_SIGNING_IN", {
                defaultValue: "Opening sign-in…",
              })
            : t("SETTINGS$DATABRICKS_SIGN_IN", {
                defaultValue: "Sign in with Databricks",
              })}
        </button>
      </div>
    );
  }

  return (
    <button
      type="button"
      data-testid="databricks-sign-in-button"
      className={baseClasses}
      onClick={handleSignIn}
      disabled={isPreparing}
    >
      {isPreparing
        ? t("SETTINGS$DATABRICKS_SIGNING_IN", {
            defaultValue: "Opening sign-in…",
          })
        : t("SETTINGS$DATABRICKS_SIGN_IN", {
            defaultValue: "Sign in with Databricks",
          })}
    </button>
  );
}

export default DatabricksSignInButton;
