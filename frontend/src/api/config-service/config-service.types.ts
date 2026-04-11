/** V1 Config API types for models and providers */

export interface LLMModel {
  /** The name of the provider for this model */
  provider: string | null;
  /** The name of this model */
  name: string;
  /** Whether the model is verified by OpenHands */
  verified: boolean;
}

export interface LLMModelPage {
  /** List of LLM models in the current page */
  items: LLMModel[];
  /** ID for the next page, or null if there are no more pages */
  next_page_id: string | null;
}

export interface SearchModelsParams {
  /** Optional next_page_id from the previously returned page */
  page_id?: string;
  /** The max number of results in the page (default: 50, max: 100) */
  limit?: number;
  /** Filter models by name (case-insensitive substring match) */
  query?: string;
  /** Filter by verified status (true/false, omit for all) */
  verified__eq?: boolean;
  /** Filter by provider name (exact match) */
  provider__eq?: string;
}