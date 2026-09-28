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
