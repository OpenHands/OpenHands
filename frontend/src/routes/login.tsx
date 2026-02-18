import React from "react";
import { useNavigate, useSearchParams } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { useIsAuthed } from "#/hooks/query/use-is-authed";
import { useConfig } from "#/hooks/query/use-config";
import { useGitHubAuthUrl } from "#/hooks/use-github-auth-url";
import { useEmailVerification } from "#/hooks/use-email-verification";
import { LoginContent } from "#/components/features/auth/login-content";
import { EmailVerificationModal } from "#/components/features/waitlist/email-verification-modal";
import AuthService from "#/api/auth-service/auth-service.api";
import { I18nKey } from "#/i18n/declaration";

function BetterAuthLoginForm({ returnTo }: { returnTo: string }) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { t } = useTranslation();
  const [email, setEmail] = React.useState("");
  const [password, setPassword] = React.useState("");
  const [name, setName] = React.useState("");
  const [error, setError] = React.useState("");
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [isSignUp, setIsSignUp] = React.useState(false);
  const [providers, setProviders] = React.useState<string[]>([]);
  const emailRef = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    emailRef.current?.focus();
  }, []);

  React.useEffect(() => {
    AuthService.getProviders()
      .then(setProviders)
      .catch(() => setProviders([]));
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsSubmitting(true);

    try {
      if (isSignUp) {
        await AuthService.signUp(email, password, name || email.split("@")[0]);
        // Auto sign-in after registration
        await AuthService.signIn(email, password);
      } else {
        await AuthService.signIn(email, password);
      }
      await queryClient.invalidateQueries({
        queryKey: ["user", "authenticated"],
      });
      navigate(returnTo, { replace: true });
    } catch (err: unknown) {
      // Show actual error from server if available
      const axiosErr = err as { response?: { data?: { message?: string } } };
      const serverMessage = axiosErr?.response?.data?.message;
      setError(
        serverMessage ||
          (isSignUp
            ? t(I18nKey.AUTH$SIGN_UP_FAILED)
            : t(I18nKey.AUTH$INVALID_CREDENTIALS)),
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleOAuth = async (provider: string) => {
    try {
      const url = await AuthService.getOAuthUrl(
        provider,
        window.location.origin + returnTo,
      );
      window.location.href = url;
    } catch {
      setError(t(I18nKey.AUTH$OAUTH_FAILED));
    }
  };

  const providerLabel = (provider: string) =>
    provider.charAt(0).toUpperCase() + provider.slice(1);

  return (
    <main
      className="min-h-screen flex items-center justify-center bg-base p-4"
      data-testid="login-page"
    >
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          {/* eslint-disable-next-line i18next/no-literal-string */}
          <h1 className="text-2xl font-bold text-white">Build.One</h1>
          <p className="text-neutral-400 mt-2">
            {isSignUp
              ? t(I18nKey.AUTH$CREATE_ACCOUNT)
              : t(I18nKey.AUTH$SIGN_IN_TO_CONTINUE)}
          </p>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4">
          {isSignUp && (
            <div>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={t(I18nKey.AUTH$NAME_PLACEHOLDER)}
                className="w-full px-4 py-3 bg-neutral-800 border border-neutral-700 rounded-lg text-white placeholder-neutral-500 focus:outline-none focus:border-blue-500"
                data-testid="name-input"
              />
            </div>
          )}
          <div>
            <input
              ref={emailRef}
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder={t(I18nKey.AUTH$EMAIL_PLACEHOLDER)}
              className="w-full px-4 py-3 bg-neutral-800 border border-neutral-700 rounded-lg text-white placeholder-neutral-500 focus:outline-none focus:border-blue-500"
              required
              data-testid="email-input"
            />
          </div>
          <div>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={t(I18nKey.FEEDBACK$PASSWORD)}
              className="w-full px-4 py-3 bg-neutral-800 border border-neutral-700 rounded-lg text-white placeholder-neutral-500 focus:outline-none focus:border-blue-500"
              required
              data-testid="password-input"
            />
          </div>
          {error && (
            <p className="text-red-400 text-sm" data-testid="login-error">
              {error}
            </p>
          )}
          <button
            type="submit"
            disabled={isSubmitting || !email || !password}
            className="w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-neutral-700 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
            data-testid="login-submit"
          >
            {isSubmitting && t(I18nKey.AUTH$SIGNING_IN)}
            {!isSubmitting && isSignUp && t(I18nKey.AUTH$SIGN_UP)}
            {!isSubmitting && !isSignUp && t(I18nKey.AUTH$SIGN_IN)}
          </button>
        </form>

        <div className="mt-4 text-center">
          <button
            type="button"
            onClick={() => {
              setIsSignUp(!isSignUp);
              setError("");
            }}
            className="text-blue-400 hover:text-blue-300 text-sm"
            data-testid="toggle-sign-up"
          >
            {isSignUp
              ? t(I18nKey.AUTH$ALREADY_HAVE_ACCOUNT)
              : t(I18nKey.AUTH$DONT_HAVE_ACCOUNT)}
          </button>
        </div>

        {providers.length > 0 && (
          <div className="mt-6">
            <div className="relative mb-4">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-neutral-700" />
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-base text-neutral-400">
                  {t(I18nKey.AUTH$OR_CONTINUE_WITH)}
                </span>
              </div>
            </div>
            <div className="space-y-2">
              {providers.map((provider) => (
                <button
                  key={provider}
                  type="button"
                  onClick={() => handleOAuth(provider)}
                  className="w-full py-3 bg-neutral-800 hover:bg-neutral-700 text-white rounded-lg font-medium transition-colors border border-neutral-700"
                  data-testid={`oauth-${provider}`}
                >
                  {providerLabel(provider)}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

export default function LoginPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const returnTo = searchParams.get("returnTo") || "/";

  const config = useConfig();
  const { data: isAuthed, isLoading: isAuthLoading } = useIsAuthed();
  const {
    emailVerified,
    hasDuplicatedEmail,
    recaptchaBlocked,
    emailVerificationModalOpen,
    setEmailVerificationModalOpen,
    userId,
  } = useEmailVerification();

  const gitHubAuthUrl = useGitHubAuthUrl({
    appMode: config.data?.APP_MODE || null,
    gitHubClientId: config.data?.GITHUB_CLIENT_ID || null,
    authUrl: config.data?.AUTH_URL,
  });

  const isB1 = config.data?.APP_MODE === "b1";

  // Redirect plain OSS mode users to home (no login needed)
  React.useEffect(() => {
    if (!config.isLoading && config.data?.APP_MODE === "oss") {
      navigate("/", { replace: true });
    }
  }, [config.isLoading, config.data?.APP_MODE, navigate]);

  // Redirect authenticated users away from login page
  React.useEffect(() => {
    if (!isAuthLoading && isAuthed) {
      navigate(returnTo, { replace: true });
    }
  }, [isAuthed, isAuthLoading, navigate, returnTo]);

  if (isAuthLoading || config.isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-base">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white" />
      </div>
    );
  }

  // Don't render login content if user is authenticated or in plain OSS mode
  if (isAuthed || config.data?.APP_MODE === "oss") {
    return null;
  }

  // B1 mode: Better Auth email/password + OAuth login
  if (isB1) {
    return <BetterAuthLoginForm returnTo={returnTo} />;
  }

  return (
    <>
      <main
        className="min-h-screen flex items-center justify-center bg-base p-4"
        data-testid="login-page"
      >
        <LoginContent
          githubAuthUrl={gitHubAuthUrl}
          appMode={config.data?.APP_MODE}
          authUrl={config.data?.AUTH_URL}
          providersConfigured={config.data?.PROVIDERS_CONFIGURED}
          emailVerified={emailVerified}
          hasDuplicatedEmail={hasDuplicatedEmail}
          recaptchaBlocked={recaptchaBlocked}
        />
      </main>

      {emailVerificationModalOpen && (
        <EmailVerificationModal
          onClose={() => {
            setEmailVerificationModalOpen(false);
          }}
          userId={userId}
        />
      )}
    </>
  );
}
