import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import OpenHandsLogoWhite from "#/assets/branding/openhands-logo-white.svg?react";
import GitHubLogo from "#/assets/branding/github-logo.svg?react";
import GitLabLogo from "#/assets/branding/gitlab-logo.svg?react";
import BitbucketLogo from "#/assets/branding/bitbucket-logo.svg?react";
import { useAuthUrl } from "#/hooks/use-auth-url";
import { GetConfigResponse } from "#/api/option-service/option.types";
import { Provider } from "#/types/settings";
import { useTracking } from "#/hooks/use-tracking";
import { TermsAndPrivacyNotice } from "#/components/shared/terms-and-privacy-notice";

export interface LoginContentProps {
  githubAuthUrl: string | null;
  appMode?: GetConfigResponse["APP_MODE"] | null;
  authUrl?: GetConfigResponse["AUTH_URL"];
  providersConfigured?: Provider[];
  emailVerified?: boolean;
  hasDuplicatedEmail?: boolean;
}

export function LoginContent({
  githubAuthUrl,
  appMode,
  authUrl,
  providersConfigured,
  emailVerified = false,
  hasDuplicatedEmail = false,
}: LoginContentProps) {
  const { t } = useTranslation();
  const { trackLoginButtonClick } = useTracking();

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

  const handleGitHubAuth = () => {
    if (githubAuthUrl) {
      trackLoginButtonClick({ provider: "github" });
      window.location.href = githubAuthUrl;
    }
  };

  const handleGitLabAuth = () => {
    if (gitlabAuthUrl) {
      trackLoginButtonClick({ provider: "gitlab" });
      window.location.href = gitlabAuthUrl;
    }
  };

  const handleBitbucketAuth = () => {
    if (bitbucketAuthUrl) {
      trackLoginButtonClick({ provider: "bitbucket" });
      window.location.href = bitbucketAuthUrl;
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
    "w-[301.5px] h-10 rounded p-2 flex items-center justify-center cursor-pointer transition-opacity hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed";
  const buttonLabelClasses = "text-sm font-medium leading-5 px-1";
  return (
    <div
      className="flex flex-col items-center w-full gap-12.5"
      data-testid="login-content"
    >
      <div>
        <OpenHandsLogoWhite width={106} height={72} />
      </div>

      <h1 className="text-[39px] leading-5 font-medium text-white text-center">
        {t(I18nKey.AUTH$LETS_GET_STARTED)}
      </h1>

      {emailVerified && (
        <p className="text-sm text-muted-foreground text-center">
          {t(I18nKey.AUTH$EMAIL_VERIFIED_PLEASE_LOGIN)}
        </p>
      )}
      {hasDuplicatedEmail && (
        <p className="text-sm text-danger text-center">
          {t(I18nKey.AUTH$DUPLICATE_EMAIL_ERROR)}
        </p>
      )}

      <div className="flex flex-col items-center gap-3">
        {noProvidersConfigured ? (
          <div className="text-center p-4 text-muted-foreground">
            {t(I18nKey.AUTH$NO_PROVIDERS_CONFIGURED)}
          </div>
        ) : (
          <>
            {showGithub && (
              <button
                type="button"
                onClick={handleGitHubAuth}
                className={`${buttonBaseClasses} bg-[#9E28B0] text-white`}
              >
                <GitHubLogo width={14} height={14} className="shrink-0" />
                <span className={buttonLabelClasses}>
                  {t(I18nKey.GITHUB$CONNECT_TO_GITHUB)}
                </span>
              </button>
            )}

            {showGitlab && (
              <button
                type="button"
                onClick={handleGitLabAuth}
                className={`${buttonBaseClasses} bg-[#FC6B0E] text-white`}
              >
                <GitLabLogo width={14} height={14} className="shrink-0" />
                <span className={buttonLabelClasses}>
                  {t(I18nKey.GITLAB$CONNECT_TO_GITLAB)}
                </span>
              </button>
            )}

            {showBitbucket && (
              <button
                type="button"
                onClick={handleBitbucketAuth}
                className={`${buttonBaseClasses} bg-[#2684FF] text-white`}
              >
                <BitbucketLogo width={14} height={14} className="shrink-0" />
                <span className={buttonLabelClasses}>
                  {t(I18nKey.BITBUCKET$CONNECT_TO_BITBUCKET)}
                </span>
              </button>
            )}
          </>
        )}
      </div>

      <TermsAndPrivacyNotice className="max-w-[320px] text-[#A3A3A3]" />
    </div>
  );
}
