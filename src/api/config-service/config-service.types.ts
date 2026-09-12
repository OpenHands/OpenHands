/** V1 Config API types for models and providers */

export interface LLMModel {
  provider: string | null;
  name: string;
  verified: boolean;
  /**
   * Whether the model is free to use on the OpenHands provider. Mirrors
   * `verified`: it is populated by the backend model list (DB-driven on
   * cloud) and defaults to `false` where no free metadata exists (e.g. the
   * local agent-server reconstruction path, which has no model database).
   */
  free: boolean;
  /**
   * Whether this is the provider's default model. Mirrors `free`: DB-driven on
   * cloud, `false` where no default metadata exists. Used to preselect the
   * model on onboarding and when creating a new model for the provider.
   */
  default: boolean;
}

export interface LLMModelPage {
  items: LLMModel[];
  next_page_id: string | null;
}

export interface SearchModelsParams {
  page_id?: string;
  limit?: number;
  query?: string;
  verified__eq?: boolean;
  provider__eq?: string;
}

export interface LLMProvider {
  name: string;
  verified: boolean;
}

export interface ProviderPage {
  items: LLMProvider[];
  next_page_id: string | null;
}

export interface SearchProvidersParams {
  page_id?: string;
  limit?: number;
  query?: string;
  verified__eq?: boolean;
}
