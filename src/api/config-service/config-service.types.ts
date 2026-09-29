/** V1 Config API types for models and providers */

/**
 * Display-only capability metadata, currently sourced from OpenRouter's live
 * catalog. Every field is optional and absence means "unknown" — never
 * "unsupported". Never used to block or filter a user's model choice.
 */
export interface LLMModelCapabilities {
  /** Raw `supported_parameters` values, e.g. `["tools", "reasoning"]`. */
  supportedParameters?: string[];
  inputModalities?: string[];
  outputModalities?: string[];
  contextLength?: number;
  /** Provider-reported max completion/output tokens, when published. */
  maxOutputTokens?: number;
  /** Per-token price strings as returned by OpenRouter, display-only. */
  pricing?: {
    prompt?: string;
    completion?: string;
  };
}

export interface LLMModel {
  provider: string | null;
  name: string;
  verified: boolean;
  /**
   * Whether the model is free to use on the OpenHands provider. Mirrors
   * `verified`: it is populated by the backend model list (DB-driven on
   * cloud) and defaults to `false` where no free metadata exists (e.g. the
   * local agent-server reconstruction path, which has no model database).
   * Never derived from an upstream provider's (e.g. OpenRouter's) price —
   * "free"/"default" here are OpenHands-specific meanings.
   */
  free: boolean;
  /**
   * Whether this is the provider's default model. Mirrors `free`: DB-driven on
   * cloud, `false` where no default metadata exists. Used to preselect the
   * model on onboarding and when creating a new model for the provider.
   */
  default: boolean;
  /**
   * Optional display metadata (context/output caps, tool/reasoning support,
   * pricing). Omitted entirely — not `{}` — when nothing is known, so it
   * never shows up as a spurious value in equality checks. See
   * {@link LLMModelCapabilities}.
   */
  capabilities?: LLMModelCapabilities;
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
