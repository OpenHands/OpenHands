import { Provider } from "#/types/settings";

export interface GetConfigResponse {
  APP_MODE: "saas" | "oss";
  APP_SLUG?: string;
  GITHUB_CLIENT_ID: string;
  POSTHOG_CLIENT_KEY: string;
  PROVIDERS_CONFIGURED?: Provider[];
  AUTH_URL?: string;
  FEATURE_FLAGS: {
    ENABLE_BILLING: boolean;
    HIDE_LLM_SETTINGS: boolean;
    HIDE_BILLING: boolean;
    ENABLE_JIRA: boolean;
    ENABLE_JIRA_DC: boolean;
    ENABLE_LINEAR: boolean;
  };
  MAINTENANCE?: {
    startTime: string;
  };
}
