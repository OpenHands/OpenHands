import React from "react";
import { useTranslation } from "react-i18next";
import DigitalOceanLogo from "#/assets/branding/digitalocean-logo.svg?react";
import { DigitalOceanConnectPanel } from "#/components/features/backends/managed-agents/digitalocean-connect-panel";
import { BackNavButton } from "#/components/shared/buttons/back-nav-button";
import { I18nKey } from "#/i18n/declaration";

interface OnboardingDigitalOceanConnectProps {
  onBack: () => void;
  onConnected: () => void;
}

/**
 * The DigitalOcean provider view inside first-run onboarding. Reuses the same
 * `DigitalOceanConnectPanel` as Manage Backends → Add Backend → Providers so
 * the sign-in, agent/session selection, and LLM-setup flow stay identical
 * across both entry points. `onConnected` fires only after a session is
 * actually opened (the panel's own connect flow), so merely opening this view
 * never provisions a session.
 */
export function OnboardingDigitalOceanConnect({
  onBack,
  onConnected,
}: OnboardingDigitalOceanConnectProps) {
  const { t } = useTranslation("openhands");

  return (
    <div className="flex flex-col gap-4">
      <BackNavButton onClick={onBack} testId="onboarding-digitalocean-back">
        {t(I18nKey.BUTTON$BACK)}
      </BackNavButton>
      <div className="flex min-w-0 items-center gap-3">
        <span
          className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-surface-raised"
          aria-hidden
        >
          <DigitalOceanLogo width={32} height={32} />
        </span>
        <h2 className="text-xl font-medium text-contrast">
          {t(I18nKey.DO_AGENTS$BADGE)}
        </h2>
      </div>
      <DigitalOceanConnectPanel onConnected={onConnected} />
    </div>
  );
}
