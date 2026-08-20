/** V1 Config API types for models and providers */

export interface LLMModel {
  provider: string | null;
  name: string;
  verified: boolean;
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
