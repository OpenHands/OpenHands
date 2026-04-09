import { Provider } from "#/types/settings";

/**
 * Structured response from ``GET /api/options/models``.
 *
 * The backend is the single source of truth — the frontend no longer carries
 * its own hardcoded verified-model lists.
 */
export interface ModelsResponse {
  /** Flat list of ``provider/model`` strings (bare names already prefixed). */
  models: string[];
  /** Model names (without provider) that OpenHands has verified to work well. */
  verified_models: string[];
  /** Provider names shown in the "Verified" section of the model selector. */
  verified_providers: string[];
  /** Recommended default model id (e.g. ``openhands/claude-opus-4-5-20251101``). */
  default_model: string;
}

export interface WebClientFeatureFlags {
  enable_billing: boolean;
  hide_llm_settings: boolean;
  enable_jira: boolean;
  enable_jira_dc: boolean;
  enable_linear: boolean;
  hide_users_page: boolean;
  hide_billing_page: boolean;
  hide_integrations_page: boolean;
}

export interface WebClientConfig {
  app_mode: "saas" | "oss";
  posthog_client_key: string | null;
  feature_flags: WebClientFeatureFlags;
  providers_configured: Provider[];
  maintenance_start_time: string | null;
  auth_url: string | null;
  recaptcha_site_key: string | null;
  faulty_models: string[];
  error_message: string | null;
  updated_at: string;
  github_app_slug: string | null;
}
