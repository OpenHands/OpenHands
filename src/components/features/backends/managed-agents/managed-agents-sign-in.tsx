import React from "react";
import { useTranslation } from "react-i18next";
import { ExternalLink, Loader2 } from "lucide-react";

import DigitalOceanLogo from "#/assets/branding/digitalocean-logo.svg?react";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";

export const DO_CREATE_TOKEN_URL =
  "https://cloud.digitalocean.com/account/api/tokens";

interface ManagedAgentsSignInProps {
  canUseOAuth: boolean;
  isExpired: boolean;
  isSigningIn: boolean;
  isSavingToken: boolean;
  error: string | null;
  onSignInWithOAuth: () => void;
  onSaveToken: (token: string) => void;
  /** Offered when adding a second connection, to return to the list. */
  onCancel?: () => void;
  /** Off where the surrounding tab already shows the DigitalOcean logo. */
  showBranding?: boolean;
}

/**
 * OAuth is one click when this build has a registered client; otherwise the
 * token field is the primary path rather than hiding behind a disclosure.
 */
export function ManagedAgentsSignIn({
  canUseOAuth,
  isExpired,
  isSigningIn,
  isSavingToken,
  error,
  onSignInWithOAuth,
  onSaveToken,
  onCancel,
  showBranding = true,
}: ManagedAgentsSignInProps) {
  const { t } = useTranslation("openhands");
  const [token, setToken] = React.useState("");
  const isBusy = isSigningIn || isSavingToken;

  const submitToken = (event: React.FormEvent) => {
    event.preventDefault();
    if (token.trim() && !isBusy) onSaveToken(token);
  };

  return (
    <div
      data-testid="managed-agents-sign-in"
      className="mx-auto flex w-full max-w-sm flex-col items-center gap-4 py-6"
    >
      {showBranding ? (
        <div className="flex size-14 items-center justify-center rounded-2xl border border-[var(--oh-border)] bg-[var(--oh-surface-raised)]">
          <DigitalOceanLogo width={32} height={32} aria-hidden />
        </div>
      ) : null}
      <div className="flex flex-col items-center gap-1.5 text-center">
        <h3 className="text-balance text-base font-medium text-white">
          {t(I18nKey.DO_AGENTS$SIGN_IN_TITLE)}
        </h3>
        <p className="text-pretty text-sm leading-relaxed text-[var(--oh-muted)]">
          {t(I18nKey.DO_AGENTS$SIGN_IN_DESCRIPTION)}
        </p>
      </div>

      {isExpired ? (
        <p
          data-testid="managed-agents-expired"
          className="w-full rounded-md border border-[var(--oh-warning)]/40 bg-[var(--oh-warning)]/10 p-2.5 text-center text-xs text-[var(--oh-warning)]"
        >
          {t(I18nKey.BACKEND$DIGITALOCEAN_EXPIRED)}
        </p>
      ) : null}

      {canUseOAuth ? (
        <BrandButton
          type="button"
          variant="primary"
          isDisabled={isBusy}
          onClick={onSignInWithOAuth}
          testId="managed-agents-oauth"
          className="w-full"
          startContent={
            isSigningIn ? (
              <Loader2 className="size-4 animate-spin" aria-hidden />
            ) : null
          }
        >
          {isSigningIn
            ? t(I18nKey.BACKEND$DIGITALOCEAN_SIGNING_IN)
            : t(I18nKey.BACKEND$DIGITALOCEAN_SIGN_IN)}
        </BrandButton>
      ) : null}

      <form onSubmit={submitToken} className="flex w-full flex-col gap-2">
        <SettingsInput
          testId="managed-agents-token"
          name="managed-agents-token"
          type="password"
          label={t(I18nKey.BACKEND$DIGITALOCEAN_TOKEN_LABEL)}
          value={token}
          onChange={setToken}
          placeholder=""
          className="w-full"
        />
        <p className="text-xs leading-relaxed text-[var(--oh-muted)]">
          {t(I18nKey.BACKEND$DIGITALOCEAN_TOKEN_HINT)}{" "}
          <a
            href={DO_CREATE_TOKEN_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-primary hover:underline"
          >
            {t(I18nKey.DO_AGENTS$CREATE_TOKEN)}
            <ExternalLink className="size-3" aria-hidden />
          </a>
        </p>
        <BrandButton
          type="submit"
          variant={canUseOAuth ? "secondary" : "primary"}
          isDisabled={!token.trim() || isBusy}
          testId="managed-agents-token-submit"
          className="w-full"
          startContent={
            isSavingToken ? (
              <Loader2 className="size-4 animate-spin" aria-hidden />
            ) : null
          }
        >
          {t(I18nKey.BACKEND$DIGITALOCEAN_TOKEN_SUBMIT)}
        </BrandButton>
      </form>

      {error ? (
        <div
          role="alert"
          data-testid="managed-agents-auth-error"
          className="w-full rounded-md border border-[var(--oh-status-error)]/40 bg-[var(--oh-status-error)]/10 p-3 text-sm text-[var(--oh-status-error)] whitespace-pre-wrap break-words"
        >
          {error}
        </div>
      ) : null}

      {onCancel ? (
        <button
          type="button"
          onClick={onCancel}
          data-testid="managed-agents-sign-in-cancel"
          className="cursor-pointer text-xs text-[var(--oh-muted)] hover:text-white"
        >
          {t(I18nKey.BUTTON$CANCEL)}
        </button>
      ) : null}
    </div>
  );
}
