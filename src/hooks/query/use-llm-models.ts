import { LLMMetadataClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { getActiveBackend } from "#/api/backend-registry/active-store";

export const LLM_MODELS_QUERY_KEY = ["config", "llm-models"] as const;
export const LLM_MODELS_STALE_TIME = 1000 * 60 * 5;
export const LLM_MODELS_GC_TIME = 1000 * 60 * 15;

/**
 * Fetch the full local model-id list once so both the provider search and the
 * per-provider model search can share a single network request via react-query.
 *
 * Mirrors {@link fetchVerifiedModelsByProvider}: cloud backends short-circuit
 * because `/api/v1/config/providers/search` and `/api/v1/config/models/search`
 * return provider/model data directly, so the local reconstruction map is a
 * no-op there.
 */
export async function fetchLLMModels(): Promise<string[]> {
  const active = getActiveBackend();
  if (active.backend.kind === "cloud") {
    return [];
  }
  const client = new LLMMetadataClient(getAgentServerClientOptions());
  return (await client.getModels()) ?? [];
}
