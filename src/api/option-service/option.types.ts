import { Provider } from "#/types/settings";

export interface WebClientFeatureFlags {
  hide_llm_settings: boolean;
  hide_users_page: boolean;
}

export interface WebClientConfig {
  feature_flags: WebClientFeatureFlags;
  providers_configured: Provider[];
  maintenance_start_time: string | null;
  recaptcha_site_key: string | null;
  faulty_models: string[];
  error_message: string | null;
  updated_at: string;
}
