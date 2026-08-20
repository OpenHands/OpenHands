import { Check, Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { AcpAuthStatus } from "#/hooks/query/use-acp-auth-status";

interface AcpAuthStatusBannerProps {
  status: AcpAuthStatus;
  isChecking: boolean;
  providerName: string;
  /**
   * Prefix for the banner test ids, e.g. ``"onboarding-acp-auth"`` →
   * ``onboarding-acp-auth-detected`` / ``onboarding-acp-auth-checking``.
   */
  testIdPrefix: string;
}

/**
 * Auth-status banner shared by the ACP credential forms (the onboarding step
 * and Settings → Agent): a green "already signed in" banner when the local
 * login probe detects a session, or a spinner while it's checking. Renders
 * nothing otherwise (unauthenticated / unknown / non-local backend), so the
 * caller falls back to the API-key fields.
 */
export function AcpAuthStatusBanner({
  status,
  isChecking,
  providerName,
  testIdPrefix,
}: AcpAuthStatusBannerProps) {
  const { t } = useTranslation("openhands");

  if (status === "authenticated") {
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

  if (isChecking) {
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

  return null;
}
