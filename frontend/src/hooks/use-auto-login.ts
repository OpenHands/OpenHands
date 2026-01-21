import { useEffect } from "react";
import { useConfig } from "./query/use-config";
import { useIsAuthed } from "./query/use-is-authed";
import { getLoginMethod, LoginMethod } from "#/utils/local-storage";
import { useAuthUrl } from "./use-auth-url";

/**
 * Hook to automatically log in the user if they have a login method stored in local storage
 * Only works in SAAS mode and when the user is not already logged in
 */
export const useAutoLogin = () => {
  const { data: config, isLoading: isConfigLoading } = useConfig();
  const { data: isAuthed, isLoading: isAuthLoading } = useIsAuthed();

  // Get the stored login method
  const loginMethod = getLoginMethod();

  // Get the auth URLs for all providers
  const githubAuthUrl = useAuthUrl({
    appMode: config?.APP_MODE || null,
    identityProvider: "github",
    authUrl: config?.AUTH_URL,
  });

  const gitlabAuthUrl = useAuthUrl({
    appMode: config?.APP_MODE || null,
    identityProvider: "gitlab",
    authUrl: config?.AUTH_URL,
  });

  const bitbucketAuthUrl = useAuthUrl({
    appMode: config?.APP_MODE || null,
    identityProvider: "bitbucket",
    authUrl: config?.AUTH_URL,
  });

  const azureDevOpsAuthUrl = useAuthUrl({
    appMode: config?.APP_MODE || null,
    identityProvider: "azure_devops",
    authUrl: config?.AUTH_URL,
  });

  const enterpriseSsoUrl = useAuthUrl({
    appMode: config?.APP_MODE || null,
    identityProvider: "enterprise_sso",
    authUrl: config?.AUTH_URL,
  });

  useEffect(() => {
    // Only auto-login in SAAS mode
    if (config?.APP_MODE !== "saas") {
      return;
    }

    // Wait for auth and config to load
    if (isConfigLoading || isAuthLoading) {
      return;
    }

    // Don't auto-login if already authenticated
    if (isAuthed) {
      return;
    }

    // Don't auto-login if no login method is stored
    if (!loginMethod) {
      return;
    }

    // Get the appropriate auth URL based on the stored login method
    let authUrl: string | null = null;
    if (loginMethod === LoginMethod.GITHUB) {
      authUrl = githubAuthUrl;
    } else if (loginMethod === LoginMethod.GITLAB) {
      authUrl = gitlabAuthUrl;
    } else if (loginMethod === LoginMethod.BITBUCKET) {
      authUrl = bitbucketAuthUrl;
    } else if (loginMethod === LoginMethod.AZURE_DEVOPS) {
      authUrl = azureDevOpsAuthUrl;
    } else if (loginMethod === LoginMethod.ENTERPRISE_SSO) {
      authUrl = enterpriseSsoUrl;
    }

    // If we have an auth URL, redirect to it
    if (authUrl) {
      // Add the login method as a query parameter
      const url = new URL(authUrl);
      url.searchParams.append("login_method", loginMethod);

      // After successful login, the user will be redirected back and can navigate to the last page
      window.location.href = url.toString();
    }
  }, [
    config?.APP_MODE,
    isAuthed,
    isConfigLoading,
    isAuthLoading,
    loginMethod,
    githubAuthUrl,
    gitlabAuthUrl,
    bitbucketAuthUrl,
    azureDevOpsAuthUrl,
    enterpriseSsoUrl,
  ]);
};
