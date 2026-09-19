import { getCachedAgentServerInfo } from "#/api/agent-server-compatibility";
import { getActiveBackend } from "#/api/backend-registry/active-store";

/** Only offer a scope when the serving backend advertises enforcement. */
export function agentProfileSupportsSecretRefs(): boolean {
  if (getActiveBackend().backend.kind === "cloud") return false;
  const capabilities = getCachedAgentServerInfo()?.capabilities;
  return (
    Array.isArray(capabilities) &&
    capabilities.includes("profile_secret_scope_v1")
  );
}

/**
 * Whether the active backend serves the tool catalog, and therefore whether the
 * editor can offer a per-profile tool selection.
 *
 * Capability-gated rather than version-gated: the catalog answers "what may a
 * user pick?", which no client can answer for itself, so a backend without it
 * gets no picker at all instead of a hardcoded list that drifts
 * (software-agent-sdk#4958).
 */
export function agentProfileSupportsToolCatalog(): boolean {
  if (getActiveBackend().backend.kind === "cloud") return false;
  const capabilities = getCachedAgentServerInfo()?.capabilities;
  return (
    Array.isArray(capabilities) && capabilities.includes("tool_catalog_v1")
  );
}
