import { WebClientConfig } from "#/api/option-service/option.types";

/**
 * Creates a mock WebClientConfig with all required fields.
 * Use this helper to create test config objects with sensible defaults.
 */
export const createMockWebClientConfig = (
  overrides: Partial<WebClientConfig> = {},
): WebClientConfig => ({
  app_mode: "oss",
  posthog_client_key: "test-posthog-key",
  feature_flags: {
    enable_billing: false,
    hide_llm_settings: false,
    enable_jira: false,
    enable_jira_dc: false,
    enable_linear: false,
    ...overrides.feature_flags,
  },
  providers_configured: [],
  maintenance_start_time: null,
  auth_url: null,
  recaptcha_site_key: null,
  faulty_models: [],
  error_message: null,
  updated_at: new Date().toISOString(),
  github_app_slug: null,
  ...overrides,
});
