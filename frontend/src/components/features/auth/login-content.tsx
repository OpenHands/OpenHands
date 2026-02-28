import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import OpenHandsLogoWhite from "#/assets/branding/openhands-logo-white.svg?react";
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
  hasInvitation?: boolean;
  buildOAuthStateData?: (
    baseStateData: Record<string, string>,
  ) => Record<string, string>;
}

export function LoginContent({
  githubAuthUrl,
  appMode,
  authUrl,
  providersConfigured,
  emailVerified = false,
  hasDuplicatedEmail = false,
  recaptchaBlocked = false,
  hasInvitation = false,
  buildOAuthStateData,
}: LoginContentProps) {
  const { t } = useTranslation();
  const { trackLoginButtonClick } = useTracking();
  const { data: config } = useConfig();

  // reCAPTCHA - only need token generation, verification happens at backend callback
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

    const url = new URL(redirectUrl);
    const currentState =
      url.searchParams.get("state") || window.location.origin;

    // Build base state data
    let stateData: Record<string, string> = {
      redirect_url: currentState,
    };

    // Add invitation token if present
    if (buildOAuthStateData) {
      stateData = buildOAuthStateData(stateData);
    }

    // If reCAPTCHA is configured, add token to state
    if (config?.recaptcha_site_key && recaptchaReady) {
      try {
        const token = await executeRecaptcha("LOGIN");
        if (token) {
          stateData.recaptcha_token = token;
        }
      } catch (err) {
        displayErrorToast(t(I18nKey.AUTH$RECAPTCHA_BLOCKED));
        return;
      }
    }

    // Encode state and redirect
    url.searchParams.set("state", btoa(JSON.stringify(stateData)));
    window.location.href = url.toString();
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
    "w-[301.5px] h-10 rounded p-2 flex items-center justify-center cursor-pointer transition-opacity hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed";
  const buttonLabelClasses = "text-sm font-medium leading-5 px-1";

  const shouldShownHelperText =
    emailVerified ||
    hasDuplicatedEmail ||
    recaptchaBlocked ||
    hasInvitation ||
    showBitbucket;

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

      {shouldShownHelperText && (
        <div className="flex flex-col items-center gap-3">
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
          {recaptchaBlocked && (
            <p className="text-sm text-danger text-center max-w-125">
              {t(I18nKey.AUTH$RECAPTCHA_BLOCKED)}
            </p>
          )}
          {hasInvitation && (
            <p className="text-sm text-muted-foreground text-center">
              {t(I18nKey.AUTH$INVITATION_PENDING)}
            </p>
          )}
          {showBitbucket && (
            <p className="text-sm text-white text-center max-w-125">
              {t(I18nKey.AUTH$BITBUCKET_SIGNUP_DISABLED)}
            </p>
          )}
        </div>
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
