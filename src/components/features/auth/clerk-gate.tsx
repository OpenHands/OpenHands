import React from "react";
import { useTranslation } from "react-i18next";
import {
  ClerkLoaded,
  ClerkLoading,
  ClerkProvider,
  SignIn,
  SignOutButton,
  SignedIn,
  SignedOut,
  useUser,
} from "@clerk/clerk-react";
import {
  getClerkAllowedEmailDomains,
  getClerkPublishableKey,
  isEmailDomainAllowed,
} from "#/api/clerk-config";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { I18nKey } from "#/i18n/declaration";
import { buildAgentCanvasPath } from "#/utils/base-path";

/**
 * Full-screen shell shared by every pre-app Clerk state. Mirrors the
 * `AgentServerBootstrapLoading` treatment in `root.tsx` so the gate does
 * not flash a differently-styled page before the canvas mounts.
 */
function ClerkScreen({ children }: { children: React.ReactNode }) {
  return (
    <main
      data-testid="clerk-gate-screen"
      className="min-h-screen bg-base px-6 py-10 text-white"
    >
      <div className="mx-auto flex min-h-screen max-w-6xl items-center justify-center">
        {children}
      </div>
    </main>
  );
}

function ClerkBootstrapLoading() {
  return (
    <ClerkScreen>
      <div className="rounded-3xl border border-white/10 bg-base/80 px-8 py-10 shadow-2xl">
        <LoadingSpinner size="large" />
      </div>
    </ClerkScreen>
  );
}

function SignInScreen() {
  return (
    <ClerkScreen>
      <div data-testid="clerk-sign-in">
        {/*
         * `virtual` routing keeps the whole sign-in flow inside this
         * component. Agent Canvas is an SPA with no server (`ssr: false`)
         * and no `/sign-in` route, so `path` routing has nothing to route
         * to and `hash` routing would fight the conversation deep links
         * the app already puts in the URL.
         */}
        <SignIn routing="virtual" />
      </div>
    </ClerkScreen>
  );
}

/**
 * Shown to a signed-in user whose email is not on
 * `VITE_CLERK_ALLOWED_EMAIL_DOMAINS`. The only way out is signing out, so
 * a wrong-account sign-in is not a dead end.
 */
function NotOnTeamScreen({ email }: { email: string | null }) {
  const { t } = useTranslation("openhands");

  return (
    <ClerkScreen>
      <div
        data-testid="clerk-not-on-team"
        className="max-w-md rounded-3xl border border-white/10 bg-base/80 px-8 py-10 text-center shadow-2xl"
      >
        <h1 className="text-lg font-semibold">
          {t(I18nKey.AUTH$NOT_ON_TEAM_TITLE)}
        </h1>
        {email && (
          <p className="mt-2 truncate text-sm text-[var(--oh-muted)]">
            {email}
          </p>
        )}
        <p className="mt-3 text-sm text-[var(--oh-muted)]">
          {t(I18nKey.AUTH$NOT_ON_TEAM_DESCRIPTION)}
        </p>
        <div className="mt-6">
          <SignOutButton>
            <button
              type="button"
              data-testid="clerk-not-on-team-sign-out"
              className="rounded-full border border-white/20 px-5 py-2 text-sm font-medium hover:bg-white/10"
            >
              {t(I18nKey.ACCOUNT_SETTINGS$LOGOUT)}
            </button>
          </SignOutButton>
        </div>
      </div>
    </ClerkScreen>
  );
}

/**
 * Defense-in-depth domain check for an already-signed-in user.
 *
 * Clerk only hands us a session for a user that its own restrictions
 * allowed to exist, so with the Dashboard set to *Restricted* this
 * component passes everyone through. It earns its place when that
 * configuration drifts — and it costs nothing when
 * `VITE_CLERK_ALLOWED_EMAIL_DOMAINS` is unset.
 */
function TeamMemberGate({ children }: { children: React.ReactNode }) {
  const { isLoaded, user } = useUser();
  const allowedDomains = getClerkAllowedEmailDomains();

  if (allowedDomains.length === 0) return <>{children}</>;
  if (!isLoaded) return <ClerkBootstrapLoading />;

  // Only a *verified* address counts: an unverified one is just a string
  // the user typed, so honoring it would hand anyone the allowlist.
  const primaryEmail = user?.primaryEmailAddress;
  const email = primaryEmail?.emailAddress ?? null;
  const isVerified = primaryEmail?.verification?.status === "verified";

  if (!isVerified || !isEmailDomainAllowed(email, allowedDomains)) {
    return <NotOnTeamScreen email={email} />;
  }

  return <>{children}</>;
}

/**
 * Gates the entire canvas behind a Clerk session.
 *
 * With no publishable key configured this renders `children` unchanged —
 * the local `npx @openhands/agent-canvas` workflow, the Playwright and
 * Vitest suites, and every OSS consumer keep their current behavior
 * without opting in.
 *
 * See `#/api/clerk-config` for the boundary this does and does not draw.
 */
export function ClerkGate({ children }: { children: React.ReactNode }) {
  const publishableKey = getClerkPublishableKey();

  if (!publishableKey) return <>{children}</>;

  return (
    <ClerkProvider
      publishableKey={publishableKey}
      afterSignOutUrl={buildAgentCanvasPath("/")}
    >
      <ClerkLoading>
        <ClerkBootstrapLoading />
      </ClerkLoading>
      <ClerkLoaded>
        <SignedOut>
          <SignInScreen />
        </SignedOut>
        <SignedIn>
          <TeamMemberGate>{children}</TeamMemberGate>
        </SignedIn>
      </ClerkLoaded>
    </ClerkProvider>
  );
}

export default ClerkGate;
