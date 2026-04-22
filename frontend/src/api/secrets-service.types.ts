import { Provider, ProviderToken } from "#/types/settings";

export type CustomSecret = {
  name: string;
  value: string;
  description?: string;
};

export type CustomSecretWithoutValue = Omit<CustomSecret, "value">;

/** Paginated response from GET /api/v1/secrets/search */
export interface CustomSecretPage {
  items: CustomSecretWithoutValue[];
  next_page_id: string | null;
}

/** @deprecated Use CustomSecretPage instead */
export interface GetSecretsResponse {
  custom_secrets: CustomSecretWithoutValue[];
}

export interface POSTProviderTokens {
  provider_tokens: Record<Provider, ProviderToken>;
}

export interface SearchSecretsParams {
  name__contains?: string;
  page_id?: string;
  limit?: number;
}
