import { Check, KeyRound, Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { AcpAuthStatus } from "#/hooks/query/use-acp-auth-status";
import { resolveAcpAuthDisplay } from "#/utils/acp-auth-display";

interface AcpAuthStatusBannerProps {
  status: AcpAuthStatus;
  isChecking: boolean;
  providerName: string;
  /**
   * Whether a credential for the provider exists in the active backend's secret
   * store. On Docker/cloud backends the host-login probe can't run (it shells
   * the interactive CLI, which isn't installed there), so this is the only
   * accurate "the agent will authenticate" signal. Optional; defaults to
   * ``false`` so existing callers behave exactly as before.
   */
  credentialsConfigured?: boolean;
  /**
   * Prefix for the banner test ids, e.g. ``"onboarding-acp-auth"`` →
   * ``onboarding-acp-auth-detected`` / ``onboarding-acp-auth-checking`` /
   * ``onboarding-acp-auth-configured``.
   */
  testIdPrefix: string;
}

/**
 * Auth-status banner shared by the ACP credential forms (the onboarding step
 * and Settings → Agent):
 *
 * - a green "already signed in" banner when the host-login probe detects a
 *   session (native backend),
 * - a spinner while the probe is checking,
 * - a neutral "credentials configured" banner when the probe can't confirm a
 *   login but a credential for the provider is stored — the accurate signal on
 *   Docker/cloud backends, where the probe goes silent,
 * - nothing otherwise, so the caller falls back to the API-key fields.
 *
 * A stored credential never renders as "signed in": only the probe confirms an
 * actual host login. See issue #1244.
 */
export function AcpAuthStatusBanner({
  status,
  isChecking,
  providerName,
  credentialsConfigured = false,
  testIdPrefix,
}: AcpAuthStatusBannerProps) {
  const { t } = useTranslation("openhands");

  const display = resolveAcpAuthDisplay({
    status,
    isChecking,
    credentialsConfigured,
  });

  if (display === "signed-in") {
    return (
      <div
        data-testid={`${testIdPrefix}-detected`}
        // Matches the onboarding "backend connected" success banner
        // (check-backend-step.tsx) for a consistent look.
        className="flex items-start gap-2 rounded-xl border border-green-500/40 bg-green-500/10 px-4 py-3 text-sm text-green-200"
      >
        <Check className="mt-0.5 size-4 shrink-0 text-green-400" aria-hidden />
        <span>
          {t(I18nKey.ONBOARDING$ACP_AUTH_DETECTED, { provider: providerName })}
        </span>
      </div>
    );
  }

  if (display === "checking") {
    return (
      <div
        data-testid={`${testIdPrefix}-checking`}
        className="flex items-center gap-2 text-sm text-[var(--oh-muted)]"
      >
        <Loader2 className="size-4 shrink-0 animate-spin" aria-hidden />
        <span>
          {t(I18nKey.ONBOARDING$ACP_AUTH_CHECKING, { provider: providerName })}
        </span>
      </div>
    );
  }

  if (display === "configured") {
    return (
      <div
        data-testid={`${testIdPrefix}-configured`}
        // Neutral/info tone — deliberately NOT the green "signed in" look, since
        // a stored credential is not a verified host login.
        className="flex items-start gap-2 rounded-xl border border-blue-500/40 bg-blue-500/10 px-4 py-3 text-sm text-blue-200"
      >
        <KeyRound
          className="mt-0.5 size-4 shrink-0 text-blue-400"
          aria-hidden
        />
        <span>
          {t(I18nKey.ONBOARDING$ACP_CREDENTIALS_CONFIGURED, {
            provider: providerName,
          })}
        </span>
      </div>
    );
  }

  return null;
}
