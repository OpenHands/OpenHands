import React from "react";
import { useNavigate, useSearchParams, useLocation } from "react-router";
import { useIsAuthed } from "#/hooks/query/use-is-authed";
import { useConfig } from "#/hooks/query/use-config";
import { useGitHubAuthUrl } from "#/hooks/use-github-auth-url";
import { useEmailVerification } from "#/hooks/use-email-verification";
import { useInvitation } from "#/hooks/use-invitation";
import { LoginContent } from "#/components/features/auth/login-content";
import { EmailVerificationModal } from "#/components/features/waitlist/email-verification-modal";
import { RequestSubmittedModal } from "#/components/features/onboarding/request-submitted-modal";

interface LocationState {
  showRequestSubmittedModal?: boolean;
}

export default function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const returnTo = searchParams.get("returnTo") || "/";
  const locationState = location.state as LocationState | null;

  const config = useConfig();
  const { data: isAuthed, isLoading: isAuthLoading } = useIsAuthed();
  const {
    emailVerified,
    hasDuplicatedEmail,
    recaptchaBlocked,
    wasRateLimited,
    emailVerificationModalOpen,
    setEmailVerificationModalOpen,
    userId,
  } = useEmailVerification();

  const { hasInvitation, buildOAuthStateData } = useInvitation();

  const gitHubAuthUrl = useGitHubAuthUrl({
    appMode: config.data?.app_mode || null,
    authUrl: config.data?.auth_url,
  });

  const [showRequestModal, setShowRequestModal] = React.useState(
    () => locationState?.showRequestSubmittedModal ?? false,
  );

  const handleRequestModalClose = () => {
    setShowRequestModal(false);
    navigate(location.pathname, { replace: true, state: {} });
  };

  // Redirect OSS mode users to home
  React.useEffect(() => {
    if (!config.isLoading && config.data?.app_mode === "oss") {
      navigate("/", { replace: true });
    }
  }, [config.isLoading, config.data?.app_mode, navigate]);

  // Redirect authenticated users away from login page
  // Preserve login_method param so useAuthCallback can store it for auto-login
  React.useEffect(() => {
    if (!isAuthLoading && isAuthed) {
      const loginMethod = searchParams.get("login_method");
      let destination = returnTo;
      if (loginMethod) {
        const separator = returnTo.includes("?") ? "&" : "?";
        destination = `${returnTo}${separator}login_method=${encodeURIComponent(loginMethod)}`;
      }
      navigate(destination, { replace: true });
    }
  }, [isAuthed, isAuthLoading, navigate, returnTo, searchParams]);

  if (isAuthLoading || config.isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-base">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white" />
      </div>
    );
  }

  // Don't render login content if user is authenticated or in OSS mode
  if (isAuthed || config.data?.app_mode === "oss") {
    return null;
  }

  return (
    <>
      <main
        className="min-h-screen flex items-center justify-center bg-base p-4"
        data-testid="login-page"
      >
        <LoginContent
          githubAuthUrl={gitHubAuthUrl}
          appMode={config.data?.app_mode}
          authUrl={config.data?.auth_url}
          providersConfigured={config.data?.providers_configured}
          emailVerified={emailVerified}
          hasDuplicatedEmail={hasDuplicatedEmail}
          recaptchaBlocked={recaptchaBlocked}
          hasInvitation={hasInvitation}
          buildOAuthStateData={buildOAuthStateData}
        />
      </main>

      {emailVerificationModalOpen && (
        <EmailVerificationModal
          onClose={() => {
            setEmailVerificationModalOpen(false);
          }}
          userId={userId}
          wasRateLimited={wasRateLimited}
        />
      )}

      {showRequestModal && (
        <RequestSubmittedModal onClose={handleRequestModalClose} />
      )}
    </>
  );
}
