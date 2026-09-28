import {
  type AgentServerInfo,
  getCachedAgentServerInfo,
} from "#/api/agent-server-compatibility";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import type { BackendKind } from "#/api/backend-registry/types";

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
 * Whether the active backend may serve the tool catalog the picker needs.
 * Cloud advertises no capabilities, so it is asked and a 404 means no.
 */
export function agentProfileMayServeToolCatalog(
  backendKind: BackendKind,
  serverInfo: AgentServerInfo | null | undefined,
): boolean {
  if (backendKind === "cloud") return true;
  const capabilities = serverInfo?.capabilities;
  return (
    Array.isArray(capabilities) && capabilities.includes("tool_catalog_v1")
  );
}
