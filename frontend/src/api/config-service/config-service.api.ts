import { openHands } from "../open-hands-axios";
import type {
  LLMModelPage,
  ProviderPage,
  SearchModelsParams,
  SearchProvidersParams,
} from "./config-service.types";

function toSearchParams(
  params: SearchModelsParams | SearchProvidersParams,
): string {
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) {
      searchParams.append(key, String(value));
    }
  }
  return searchParams.toString();
}

class ConfigService {
  static async searchModels(
    params: SearchModelsParams = {},
  ): Promise<LLMModelPage> {
    const qs = toSearchParams(params);
    const { data } = await openHands.get<LLMModelPage>(
      `/api/v1/config/models/search?${qs}`,
    );
    return data;
  }

  static async searchProviders(
    params: SearchProvidersParams = {},
  ): Promise<ProviderPage> {
    const qs = toSearchParams(params);
    const { data } = await openHands.get<ProviderPage>(
      `/api/v1/config/providers/search?${qs}`,
    );
    return data;
  }
}

export default ConfigService;
