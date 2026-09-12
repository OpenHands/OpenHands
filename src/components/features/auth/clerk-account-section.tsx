import { SignOutButton, useUser } from "@clerk/clerk-react";
import { useTranslation } from "react-i18next";
import { isClerkEnabled } from "#/api/clerk-config";
import { I18nKey } from "#/i18n/declaration";

function AccountCard() {
  const { t } = useTranslation("openhands");
  const { isLoaded, user } = useUser();

  if (!isLoaded || !user) return null;

  const email = user.primaryEmailAddress?.emailAddress ?? null;
  const name = user.fullName ?? null;
  // Show the email as a second line only when it isn't already the label.
  const secondaryEmail = name ? email : null;

  return (
    <div
      data-testid="clerk-account-section"
      className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-white/10 bg-base/60 px-5 py-4"
    >
      <div className="min-w-0">
        {(name || email) && (
          <p className="truncate text-sm font-medium">{name || email}</p>
        )}
        {secondaryEmail && (
          <p className="truncate text-xs text-[var(--oh-muted)]">
            {secondaryEmail}
          </p>
        )}
      </div>
      <SignOutButton>
        <button
          type="button"
          data-testid="clerk-sign-out-button"
          className="rounded-full border border-white/20 px-4 py-1.5 text-sm font-medium hover:bg-white/10"
        >
          {t(I18nKey.ACCOUNT_SETTINGS$LOGOUT)}
        </button>
      </SignOutButton>
    </div>
  );
}

/**
 * "Signed in as … / Logout" row for the App Settings page.
 *
 * Renders nothing when Clerk is not configured, which keeps App Settings
 * identical for the default (ungated) build. The `isClerkEnabled()` check
 * must come before any Clerk hook: `useUser` throws outside a
 * `ClerkProvider`, and the provider only exists when a publishable key is
 * configured.
 */
export function ClerkAccountSection() {
  if (!isClerkEnabled()) return null;
  return <AccountCard />;
}

export default ClerkAccountSection;
