import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import GitHubLogo from "#/assets/branding/github-logo.svg?react";
import GitLabLogo from "#/assets/branding/gitlab-logo.svg?react";
import BitbucketLogo from "#/assets/branding/bitbucket-logo.svg?react";
import { useAuthUrl } from "#/hooks/use-auth-url";
import { WebClientConfig } from "#/api/option-service/option.types";
import { Provider } from "#/types/settings";
import { useTracking } from "#/hooks/use-tracking";
import { TermsAndPrivacyNotice } from "#/components/shared/terms-and-privacy-notice";
import { useRecaptcha } from "#/hooks/use-recaptcha";
import { useConfig } from "#/hooks/query/use-config";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

export interface LoginContentProps {
  githubAuthUrl: string | null;
  appMode?: WebClientConfig["app_mode"] | null;
  authUrl?: WebClientConfig["auth_url"];
  providersConfigured?: Provider[];
  emailVerified?: boolean;
  hasDuplicatedEmail?: boolean;
  recaptchaBlocked?: boolean;
}

export function LoginContent({
  githubAuthUrl,
  appMode,
  authUrl,
  providersConfigured,
  emailVerified = false,
  hasDuplicatedEmail = false,
  recaptchaBlocked = false,
}: LoginContentProps) {
  const { t } = useTranslation();
  const { trackLoginButtonClick } = useTracking();
  const { data: config } = useConfig();

  const { isReady: recaptchaReady, executeRecaptcha } = useRecaptcha({
    siteKey: config?.recaptcha_site_key ?? undefined,
  });

  const gitlabAuthUrl = useAuthUrl({
    appMode: appMode || null,
    identityProvider: "gitlab",
    authUrl,
  });

  const bitbucketAuthUrl = useAuthUrl({
    appMode: appMode || null,
    identityProvider: "bitbucket",
    authUrl,
  });

  const handleAuthRedirect = async (
    redirectUrl: string,
    provider: Provider,
  ) => {
    trackLoginButtonClick({ provider });

    if (!config?.recaptcha_site_key || !recaptchaReady) {
      window.location.href = redirectUrl;
      return;
    }

    try {
      const token = await executeRecaptcha("LOGIN");
      if (token) {
        const url = new URL(redirectUrl);
        const currentState =
          url.searchParams.get("state") || window.location.origin;

        const stateData = {
          redirect_url: currentState,
          recaptcha_token: token,
        };
        url.searchParams.set("state", btoa(JSON.stringify(stateData)));
        window.location.href = url.toString();
      }
    } catch (err) {
      displayErrorToast(t(I18nKey.AUTH$RECAPTCHA_BLOCKED));
    }
  };

  const handleGitHubAuth = () => {
    if (githubAuthUrl) {
      handleAuthRedirect(githubAuthUrl, "github");
    }
  };

  const handleGitLabAuth = () => {
    if (gitlabAuthUrl) {
      handleAuthRedirect(gitlabAuthUrl, "gitlab");
    }
  };

  const handleBitbucketAuth = () => {
    if (bitbucketAuthUrl) {
      handleAuthRedirect(bitbucketAuthUrl, "bitbucket");
    }
  };

  const showGithub =
    providersConfigured &&
    providersConfigured.length > 0 &&
    providersConfigured.includes("github");
  const showGitlab =
    providersConfigured &&
    providersConfigured.length > 0 &&
    providersConfigured.includes("gitlab");
  const showBitbucket =
    providersConfigured &&
    providersConfigured.length > 0 &&
    providersConfigured.includes("bitbucket");

  const noProvidersConfigured =
    !providersConfigured || providersConfigured.length === 0;

  const buttonBaseClasses =
    "w-[320px] h-12 rounded-xl p-3 flex items-center justify-center cursor-pointer transition-all duration-200 hover:scale-[1.01] active:scale-[0.99] disabled:opacity-50 disabled:cursor-not-allowed shadow-sm";
  const buttonLabelClasses = "text-sm font-medium leading-5 px-2 tracking-[-0.01em]";

  return (
    <div
      className="flex flex-col items-center w-full gap-10"
      data-testid="login-content"
    >
      {/* neww.ai Logo & Brand */}
      <div className="flex flex-col items-center gap-4">
        <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-[#6366F1] to-[#4F46E5] shadow-[0_8px_32px_rgba(99,102,241,0.35)]">
          <span
            className="text-white font-bold text-2xl tracking-tight"
            style={{ fontFamily: "Inter, -apple-system, sans-serif" }}
          >
            n.
          </span>
        </div>
        <div className="flex flex-col items-center gap-1 mt-1">
          <h2
            className="text-[32px] font-bold text-white tracking-tight leading-tight"
            style={{ fontFamily: "Inter, -apple-system, sans-serif" }}
          >
            neww<span className="neww-gradient-text">.ai</span>
          </h2>
          <p className="text-[13px] text-[#71717A] font-medium tracking-[0.08em] uppercase">
            AI Coding Agent
          </p>
        </div>
      </div>

      {/* Welcome text */}
      <div className="flex flex-col items-center gap-2">
        <h1 className="text-[22px] font-semibold text-white text-center tracking-[-0.02em]">
          {t(I18nKey.AUTH$LETS_GET_STARTED)}
        </h1>
        <p className="text-[15px] text-[#A1A1AA] text-center max-w-[280px] leading-relaxed">
          Ship production code 10x faster with AI
        </p>
      </div>

      {emailVerified && (
        <p className="text-sm text-[#A1A1AA] text-center">
          {t(I18nKey.AUTH$EMAIL_VERIFIED_PLEASE_LOGIN)}
        </p>
      )}
      {hasDuplicatedEmail && (
        <p className="text-sm text-[#F43F5E] text-center">
          {t(I18nKey.AUTH$DUPLICATE_EMAIL_ERROR)}
        </p>
      )}
      {recaptchaBlocked && (
        <p className="text-sm text-[#F43F5E] text-center max-w-125">
          {t(I18nKey.AUTH$RECAPTCHA_BLOCKED)}
        </p>
      )}

      {/* Auth buttons */}
      <div className="flex flex-col items-center gap-3">
        {noProvidersConfigured ? (
          <div className="text-center p-4 text-[#71717A]">
            {t(I18nKey.AUTH$NO_PROVIDERS_CONFIGURED)}
          </div>
        ) : (
          <>
            {showGithub && (
              <button
                type="button"
                onClick={handleGitHubAuth}
                className={`${buttonBaseClasses} bg-white text-[#09090B] hover:bg-[#F4F4F5]`}
              >
                <GitHubLogo width={18} height={18} className="shrink-0" />
                <span className={buttonLabelClasses}>
                  {t(I18nKey.GITHUB$CONNECT_TO_GITHUB)}
                </span>
              </button>
            )}

            {showGitlab && (
              <button
                type="button"
                onClick={handleGitLabAuth}
                className={`${buttonBaseClasses} bg-[#FC6D26] text-white hover:bg-[#E5611F]`}
              >
                <GitLabLogo width={18} height={18} className="shrink-0" />
                <span className={buttonLabelClasses}>
                  {t(I18nKey.GITLAB$CONNECT_TO_GITLAB)}
                </span>
              </button>
            )}

            {showBitbucket && (
              <button
                type="button"
                onClick={handleBitbucketAuth}
                className={`${buttonBaseClasses} bg-[#0052CC] text-white hover:bg-[#0047B3]`}
              >
                <BitbucketLogo width={18} height={18} className="shrink-0" />
                <span className={buttonLabelClasses}>
                  {t(I18nKey.BITBUCKET$CONNECT_TO_BITBUCKET)}
                </span>
              </button>
            )}
          </>
        )}
      </div>

      <TermsAndPrivacyNotice className="max-w-[320px] text-[#52525B] text-[13px]" />
    </div>
  );
}
