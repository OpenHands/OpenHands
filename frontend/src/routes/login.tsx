import React from "react";
import { useNavigate, useSearchParams } from "react-router";
import { useIsAuthed } from "#/hooks/query/use-is-authed";
import { useConfig } from "#/hooks/query/use-config";
import { useGitHubAuthUrl } from "#/hooks/use-github-auth-url";
import { useEmailVerification } from "#/hooks/use-email-verification";
import { LoginContent } from "#/components/features/auth/login-content";
import { useModalStore } from "#/stores/modal-store";

export default function LoginPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const returnTo = searchParams.get("returnTo") || "/";
  const openModal = useModalStore((state) => state.openModal);

  const config = useConfig();
  const { data: isAuthed, isLoading: isAuthLoading } = useIsAuthed();
  const {
    emailVerified,
    hasDuplicatedEmail,
    recaptchaBlocked,
    emailVerificationModalOpen,
    userId,
  } = useEmailVerification();

  const gitHubAuthUrl = useGitHubAuthUrl({
    appMode: config.data?.APP_MODE || null,
    gitHubClientId: config.data?.GITHUB_CLIENT_ID || null,
    authUrl: config.data?.AUTH_URL,
  });

  // Redirect OSS mode users to home
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

  // Show email verification modal when needed
  React.useEffect(() => {
    if (emailVerificationModalOpen) {
      openModal("email-verification", { userId });
    }
  }, [emailVerificationModalOpen, userId, openModal]);

  if (isAuthLoading || config.isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-base">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white" />
      </div>
    );
  }

  // Don't render login content if user is authenticated or in OSS mode
  if (isAuthed || config.data?.APP_MODE === "oss") {
    return null;
  }

  return (
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
  );
}
