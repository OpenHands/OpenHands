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

/** Only offer profile instructions where launches apply them. */
export function agentProfileSupportsInstructions(): boolean {
  return getActiveBackend().backend.kind !== "cloud";
}

/** Local only: cloud launches ignore a profile's tools for now. */
export function agentProfileSupportsTools(): boolean {
  return getActiveBackend().backend.kind !== "cloud";
}
