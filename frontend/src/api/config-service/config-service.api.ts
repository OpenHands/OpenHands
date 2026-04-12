import { openHands } from "../open-hands-axios";
import type { LLMModelPage, SearchModelsParams } from "./config-service.types";

/**
 * Service for handling V1 Config API endpoints
 */
class ConfigService {
  /**
   * Search for LLM models with pagination and filtering.
   *
   * @param params - Search parameters including pagination, query, verified status, and provider filter
   * @returns Paginated list of LLM models
   */
  static async searchModels(
    params: SearchModelsParams = {},
  ): Promise<LLMModelPage> {
    const searchParams = new URLSearchParams();

    if (params.page_id) {
      searchParams.append("page_id", params.page_id);
    }
    if (params.limit) {
      searchParams.append("limit", params.limit.toString());
    }
    if (params.query) {
      searchParams.append("query", params.query);
    }
    if (params.verified__eq !== undefined) {
      searchParams.append("verified__eq", params.verified__eq.toString());
    }
    if (params.provider__eq) {
      searchParams.append("provider__eq", params.provider__eq);
    }

    const { data } = await openHands.get<LLMModelPage>(
      `/api/v1/config/models/search?${searchParams.toString()}`,
    );
    return data;
  }

  /**
   * Get the list of verified providers.
   *
   * @returns List of verified provider names
   */
  static async getProviders(): Promise<string[]> {
    const { data } = await openHands.get<string[]>("/api/v1/config/providers");
    return data;
  }
}

export default ConfigService;
