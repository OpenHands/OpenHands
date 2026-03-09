import { useAuthUrl } from "./use-auth-url";
import { WebClientConfig } from "#/api/option-service/option.types";

interface UseGitHubAuthUrlConfig {
  appMode: WebClientConfig["app_mode"] | null;
  authUrl?: WebClientConfig["auth_url"];
}

export const useGitHubAuthUrl = (config: UseGitHubAuthUrlConfig) =>
  useAuthUrl({
    appMode: config.appMode,
    identityProvider: "github",
    authUrl: config.authUrl,
  });
