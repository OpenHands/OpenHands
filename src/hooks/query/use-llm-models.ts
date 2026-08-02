import { LLMMetadataClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { getActiveBackend } from "#/api/backend-registry/active-store";

export const LLM_MODELS_QUERY_KEY = ["config", "llm-models"] as const;

export async function fetchLlmModels(): Promise<string[]> {
  const active = getActiveBackend();
  if (active.backend.kind === "cloud") {
    // Cloud backends use /api/v1/config/providers/search and /api/v1/config/models/search
    // directly. The raw model-ID list is only used by the local ConfigService
    // reconstruction logic, so callers can safely treat this empty list as a
    // no-op for cloud.
    return [];
  }
  const client = new LLMMetadataClient(getAgentServerClientOptions());
  return (await client.getModels()) ?? [];
}
