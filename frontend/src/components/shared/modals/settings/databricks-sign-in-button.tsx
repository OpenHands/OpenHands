import React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import DatabricksAuthService from "#/api/databricks-auth-service/databricks-auth-service.api";
import {
  useDatabricksAuthStatus,
  useDatabricksLogout,
} from "#/hooks/query/use-databricks-auth-status";
import { cn } from "#/utils/utils";

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

  const handleSignIn = async () => {
    if (u2mClientId?.trim()) {
      setIsPreparing(true);
      try {
        const initiateRelativePath = await DatabricksAuthService.prepare(
          u2mClientId.trim(),
          u2mHost?.trim() || "",
          u2mClientSecret || null,
          u2mRedirectUri?.trim() || null,
        );
        // Open the OAuth flow in a new tab so the user keeps the settings
        // page open. We target the backend (port 3000) directly rather than
        // going through the Vite proxy, so /auth/* cookies are set on the
        // correct origin (3000). The OAuth callback may come back to port 8080
        // (CLI shim) which then client-side redirects back to port 3000 —
        // keeping the session cookie valid throughout.
        const backendOrigin =
          window.location.port === "3001"
            ? `${window.location.protocol}//${window.location.hostname}:3000`
            : window.location.origin;
        window.open(`${backendOrigin}${initiateRelativePath}`, "_blank");
        startPolling();
      } finally {
        setIsPreparing(false);
      }
    } else {
      // Env-var path — open initiate URL in a new tab.
      const origin =
        window.location.port === "3001"
          ? `${window.location.protocol}//${window.location.hostname}:3000`
          : window.location.origin;
      window.open(`${origin}${DatabricksAuthService.INITIATE_URL}`, "_blank");
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

  if (status?.authenticated) {
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
