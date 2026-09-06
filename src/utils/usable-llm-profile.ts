import type { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import { isSubscriptionLlmConfig } from "#/constants/llm-subscription";

type ProfileLike = Pick<ProfileInfo, "api_key_set" | "model" | "base_url"> & {
  provider_connection_id?: string | null;
  provider_connection_broken?: boolean;
};

function isOllamaEndpoint(
  model: string | null | undefined,
  baseUrl: string | null | undefined,
): boolean {
  if (!baseUrl?.trim()) return false;
  const id = (model ?? "").trim().toLowerCase();
  return (
    id === "ollama" || id.startsWith("ollama/") || id.startsWith("ollama_chat/")
  );
}

/**
 * Whether this LLM profile can power an OpenHands conversation: a stored API
 * key, a live provider connection (Ollama / hosted APIs), a local/remote
 * Ollama base URL, or ChatGPT-style subscription auth on the profile detail.
 */
export function isUsableLlmProfile(
  profile: ProfileLike | null | undefined,
  detail?: { config?: Record<string, unknown> } | null,
): boolean {
  if (!profile) return false;
  if (profile.api_key_set) return true;
  if (
    Boolean(profile.provider_connection_id) &&
    !profile.provider_connection_broken
  ) {
    return true;
  }
  if (isOllamaEndpoint(profile.model, profile.base_url)) return true;
  return isSubscriptionLlmConfig(detail?.config);
}
