import { LLMMetadataClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { getActiveBackend } from "#/api/backend-registry/active-store";

export const LLM_MODELS_QUERY_KEY = ["config", "llm-models"] as const;
export const LLM_MODELS_STALE_TIME = 1000 * 60 * 5;
export const LLM_MODELS_GC_TIME = 1000 * 60 * 15;

export async function fetchLLMModels(): Promise<string[]> {
  const active = getActiveBackend();
  if (active.backend.kind === "cloud") {
    // Cloud search endpoints return models directly; local reconstruction only.
    return [];
  }
  const client = new LLMMetadataClient(getAgentServerClientOptions());
  return (await client.getModels()) ?? [];
}
