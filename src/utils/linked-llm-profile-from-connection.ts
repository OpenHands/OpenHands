import type { SaveProfileRequest } from "#/api/profiles-service/profiles-service.api";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { profileNameFromDisplayName } from "#/utils/derive-profile-name";
import {
  DEFAULT_OLLAMA_MODEL,
  OLLAMA_PROVIDER_ID,
} from "#/constants/provider-connection-options";

/**
 * First-run model for a newly added provider. The user can change it on the
 * profile afterward — Add provider should not also require Add LLM profile.
 */
export function defaultLlmModelForProvider(provider: string): string {
  switch (provider) {
    case OLLAMA_PROVIDER_ID:
      return DEFAULT_OLLAMA_MODEL;
    case "anthropic":
      return "anthropic/claude-sonnet-4-5";
    case "gemini":
      return "gemini/gemini-2.5-pro";
    case "openhands":
      return "openhands/glm-5.2";
    default:
      return DEFAULT_SETTINGS.llm_model;
  }
}

export function linkedLlmProfileFromConnection(connection: {
  id: string;
  display_name: string;
  provider: string;
}): { name: string; request: SaveProfileRequest } {
  return {
    name: profileNameFromDisplayName(connection.display_name),
    request: {
      llm: {
        model: defaultLlmModelForProvider(connection.provider),
        provider_connection_id: connection.id,
      } as SaveProfileRequest["llm"],
    },
  };
}
