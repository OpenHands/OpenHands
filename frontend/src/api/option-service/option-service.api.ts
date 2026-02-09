import { openHands } from "../open-hands-axios";
import { WebClientConfig } from "./option.types";

/**
 * Service for handling API options endpoints
 */
class OptionService {
  /**
   * Retrieve the list of models available
   * @returns List of models available
   */
  static async getModels(): Promise<string[]> {
    const { data } = await openHands.get<string[]>("/api/options/models");
    return data;
  }

  /**
   * Retrieve the list of agents available
   * @returns List of agents available
   */
  static async getAgents(): Promise<string[]> {
    const { data } = await openHands.get<string[]>("/api/options/agents");
    return data;
  }

  /**
   * Retrieve the list of security analyzers available
   * @returns List of security analyzers available
   */
  static async getSecurityAnalyzers(): Promise<string[]> {
    const { data } = await openHands.get<string[]>(
      "/api/options/security-analyzers",
    );
    return data;
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
