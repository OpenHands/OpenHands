import { LLMMetadataClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { getActiveBackend } from "#/api/backend-registry/active-store";

export const VERIFIED_MODELS_QUERY_KEY = ["config", "verified-models"] as const;
export const VERIFIED_MODELS_STALE_TIME = 1000 * 60 * 5;
export const VERIFIED_MODELS_GC_TIME = 1000 * 60 * 15;

export async function fetchVerifiedModelsByProvider(): Promise<
  Record<string, string[]>
> {
  const active = getActiveBackend();
  if (active.backend.kind === "cloud") {
    // Cloud backends use /api/v1/config/providers/search and /api/v1/config/models/search,
    // which return verified status directly on each item. The intermediate
    // verifiedByProvider map is only used by the local ConfigService reconstruction
    // logic, so callers can safely treat this empty object as a no-op for cloud.
    return {};
  }
  const client = new LLMMetadataClient(getAgentServerClientOptions());
  return (await client.getVerifiedModels()) ?? {};
}
