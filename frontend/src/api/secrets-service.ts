import { openHands } from "./open-hands-axios";
import {
  CustomSecret,
  CustomSecretPage,
  CustomSecretWithoutValue,
  POSTProviderTokens,
  SearchSecretsParams,
} from "./secrets-service.types";
import { Provider, ProviderToken } from "#/types/settings";

export class SecretsService {
  /**
   * Search/list custom secrets with pagination support.
   * Uses the new V1 API endpoint: GET /api/v1/secrets/search
   */
  static async searchSecrets(
    params: SearchSecretsParams = {},
  ): Promise<CustomSecretPage> {
    const queryParams = new URLSearchParams();

    if (params.name__contains) {
      queryParams.set("name__contains", params.name__contains);
    }
    if (params.page_id) {
      queryParams.set("page_id", params.page_id);
    }
    if (params.limit) {
      queryParams.set("limit", params.limit.toString());
    }

    const queryString = queryParams.toString();
    const url = `/api/v1/secrets/search${queryString ? `?${queryString}` : ""}`;

    const { data } = await openHands.get<CustomSecretPage>(url);
    return data;
  }

  /**
   * @deprecated Use searchSecrets instead. This method uses the deprecated V0 API.
   */
  static async getSecrets(): Promise<CustomSecretWithoutValue[]> {
    // Fetch all secrets by iterating through pages
    const allSecrets: CustomSecretWithoutValue[] = [];
    let pageId: string | null = null;

    // eslint-disable-next-line no-await-in-loop
    for (;;) {
      // eslint-disable-next-line no-await-in-loop
      const page = await SecretsService.searchSecrets({
        page_id: pageId ?? undefined,
        limit: 100,
      });
      allSecrets.push(...page.items);
      pageId = page.next_page_id;
      if (!pageId) break;
    }

    return allSecrets;
  }

  static async createSecret(name: string, value: string, description?: string) {
    const secret: CustomSecret = {
      name,
      value,
      description,
    };

    const { status } = await openHands.post("/api/v1/secrets", secret);
    return status === 201;
  }

  static async updateSecret(id: string, name: string, description?: string) {
    const secret: CustomSecretWithoutValue = {
      name,
      description,
    };

    const { status } = await openHands.put(`/api/v1/secrets/${id}`, secret);
    return status === 200;
  }

  static async deleteSecret(id: string) {
    const { status } = await openHands.delete<boolean>(`/api/v1/secrets/${id}`);
    return status === 200;
  }

  static async addGitProvider(providers: Record<Provider, ProviderToken>) {
    const tokens: POSTProviderTokens = {
      provider_tokens: providers,
    };
    const { data } = await openHands.post<boolean>(
      "/api/v1/secrets/git-providers",
      tokens,
    );
    return data;
  }
}
