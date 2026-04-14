import { openHands } from "../open-hands-axios";
import { normalizeStringArray } from "#/utils/normalize-string-array";
import { ModelsResponse, WebClientConfig } from "./option.types";

/**
 * Service for handling API options endpoints
 */
class OptionService {
  /**
   * Retrieve the structured models response from the backend.
   *
   * The backend is the single source of truth for verified models,
   * verified providers, and provider assignment for bare model names.
   */
  static async getModels(): Promise<ModelsResponse> {
    const { data } = await openHands.get<ModelsResponse>("/api/options/models");
    if (!data || typeof data !== "object") {
      return {
        models: [],
        verified_models: [],
        verified_providers: [],
        default_model: "",
      };
    }
    return {
      ...data,
      models: normalizeStringArray(data.models),
      verified_models: normalizeStringArray(data.verified_models),
      verified_providers: normalizeStringArray(data.verified_providers),
      default_model:
        typeof data.default_model === "string" ? data.default_model : "",
    };
  }

  /**
   * Retrieve the list of security analyzers available
   * @returns List of security analyzers available
   */
  static async getSecurityAnalyzers(): Promise<string[]> {
    const { data } = await openHands.get<unknown>(
      "/api/options/security-analyzers",
    );
    return normalizeStringArray(data);
  }

  /**
   * Get the web client configuration from the server
   * @returns Web client configuration response
   */
  static async getConfig(): Promise<WebClientConfig> {
    const { data } = await openHands.get<WebClientConfig>(
      "/api/v1/web-client/config",
    );
    return data;
  }
}

export default OptionService;
