import React from "react";
import { useTranslation } from "react-i18next";
import { ExternalLink, Loader2 } from "lucide-react";

import DigitalOceanLogo from "#/assets/branding/digitalocean-logo.svg?react";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { ErrorAlert } from "#/components/shared/error-alert";
import type { PendingMarsSignIn } from "#/hooks/use-mars-sign-in";
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
  /** A browser sign-in waiting on DigitalOcean (web build). */
  pendingSignIn?: PendingMarsSignIn | null;
  isCompletingSignIn?: boolean;
  /** Finish from the address DigitalOcean sent the browser to. */
  onCompleteSignIn?: (returned: string) => void;
  onCancelSignIn?: () => void;
  /** Offered when adding a second connection, to return to the list. */
  onCancel?: () => void;
  /** Off where the surrounding tab already shows the DigitalOcean logo. */
  showBranding?: boolean;
}

interface PendingSignInProps {
  pending: PendingMarsSignIn;
  isCompleting: boolean;
  onComplete?: (returned: string) => void;
  onCancel?: () => void;
}

/**
 * While DigitalOcean's page is open: how the sign-in comes back, and a field
 * for the address it lands on when that cannot reach Canvas by itself.
 */
function PendingSignIn({
  pending,
  isCompleting,
  onComplete,
  onCancel,
}: PendingSignInProps) {
  const { t } = useTranslation("openhands");
  const [returned, setReturned] = React.useState("");

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (returned.trim() && !isCompleting) onComplete?.(returned);
  };

  return (
    <form
      onSubmit={submit}
      data-testid="managed-agents-pending-sign-in"
      className="flex w-full flex-col gap-2 rounded-md border border-border p-3"
    >
      <p
        className={
          pending.returnsHere
            ? "text-pretty text-xs leading-relaxed text-muted"
            : "text-pretty text-xs leading-relaxed text-warning"
        }
      >
        {t(
          pending.returnsHere
            ? I18nKey.DO_AGENTS$FINISH_ON_DO
            : I18nKey.DO_AGENTS$RETURN_WONT_LOAD,
        )}{" "}
        <a
          href={pending.authorizeUrl}
          target="_blank"
          rel="noopener noreferrer"
          data-testid="managed-agents-open-sign-in"
          className="inline-flex items-center gap-1 text-primary hover:underline"
        >
          {t(I18nKey.DO_AGENTS$OPEN_SIGN_IN_AGAIN)}
          <ExternalLink className="size-3" aria-hidden />
        </a>
      </p>
      <SettingsInput
        testId="managed-agents-return-url"
        name="managed-agents-return-url"
        type="text"
        label={t(I18nKey.DO_AGENTS$PASTE_RETURN_LABEL)}
        value={returned}
        onChange={setReturned}
        placeholder=""
        className="w-full"
      />
      <div className="flex items-center gap-3">
        <BrandButton
          type="submit"
          variant="secondary"
          isDisabled={!returned.trim() || isCompleting}
          testId="managed-agents-return-submit"
          className="flex-1"
          startContent={
            isCompleting ? (
              <Loader2 className="size-4 animate-spin" aria-hidden />
            ) : null
          }
        >
          {t(I18nKey.DO_AGENTS$FINISH_SIGN_IN)}
        </BrandButton>
        <button
          type="button"
          onClick={onCancel}
          data-testid="managed-agents-sign-in-abort"
          className="cursor-pointer text-xs text-muted hover:text-white"
        >
          {t(I18nKey.BUTTON$CANCEL)}
        </button>
      </div>
    </form>
  );
}

/**
 * Signing in with DigitalOcean is the one obvious action when this Canvas
 * has an OAuth client, with the token behind a disclosure; otherwise the
 * token field is the primary path.
 */
export function ManagedAgentsSignIn({
  canUseOAuth,
  isExpired,
  isSigningIn,
  isSavingToken,
  error,
  onSignInWithOAuth,
  onSaveToken,
  pendingSignIn = null,
  isCompletingSignIn = false,
  onCompleteSignIn,
  onCancelSignIn,
  onCancel,
  showBranding = true,
}: ManagedAgentsSignInProps) {
  const { t } = useTranslation("openhands");
  const [token, setToken] = React.useState("");
  const [isTokenOpen, setIsTokenOpen] = React.useState(false);
  const isBusy = isSigningIn || isSavingToken;
  // A browser sign-in that waits on DigitalOcean can be dropped for a token;
  // the desktop app's cannot.
  const isTokenBusy = isSavingToken || (isSigningIn && !pendingSignIn);
  const showTokenForm = !canUseOAuth || isTokenOpen;

  const submitToken = (event: React.FormEvent) => {
    event.preventDefault();
    if (token.trim() && !isTokenBusy) onSaveToken(token);
  };

  return (
    <div
      data-testid="managed-agents-sign-in"
      className="mx-auto flex w-full max-w-sm flex-col items-center gap-4 py-6"
    >
      {showBranding ? (
        <div className="flex size-14 items-center justify-center rounded-2xl border border-border bg-surface-raised">
          <DigitalOceanLogo width={32} height={32} aria-hidden />
        </div>
      ) : null}
      <div className="flex flex-col items-center gap-1.5 text-center">
        <h3 className="text-balance text-base font-medium text-white">
          {t(I18nKey.DO_AGENTS$SIGN_IN_TITLE)}
        </h3>
        <p className="text-pretty text-sm leading-relaxed text-muted">
          {t(I18nKey.DO_AGENTS$SIGN_IN_DESCRIPTION)}
        </p>
      </div>

      {isExpired ? (
        <p
          data-testid="managed-agents-expired"
          className="w-full rounded-md border border-warning/40 bg-warning/10 p-2.5 text-center text-xs text-warning"
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

      {pendingSignIn ? (
        <PendingSignIn
          pending={pendingSignIn}
          isCompleting={isCompletingSignIn}
          onComplete={onCompleteSignIn}
          onCancel={onCancelSignIn}
        />
      ) : null}

      {canUseOAuth ? (
        <button
          type="button"
          onClick={() => setIsTokenOpen((open) => !open)}
          aria-expanded={isTokenOpen}
          aria-controls="managed-agents-token-form"
          data-testid="managed-agents-use-token"
          className="cursor-pointer text-xs text-muted hover:text-white"
        >
          {t(I18nKey.DO_AGENTS$USE_TOKEN_INSTEAD)}
        </button>
      ) : null}

      {showTokenForm ? (
        <form
          id="managed-agents-token-form"
          onSubmit={submitToken}
          className="flex w-full flex-col gap-2"
        >
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
          <p className="text-xs leading-relaxed text-muted">
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
            isDisabled={!token.trim() || isTokenBusy}
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
      ) : null}

      {error ? (
        <ErrorAlert testId="managed-agents-auth-error" className="w-full">
          {error}
        </ErrorAlert>
      ) : null}

      {onCancel ? (
        <button
          type="button"
          onClick={onCancel}
          data-testid="managed-agents-sign-in-cancel"
          className="cursor-pointer text-xs text-muted hover:text-white"
        >
          {t(I18nKey.BUTTON$CANCEL)}
        </button>
      ) : null}
    </div>
  );
}
