import {
  compareAgentServerVersions,
  getCachedAgentServerVersion,
  getCachedAgentServerInfo,
} from "#/api/agent-server-compatibility";
import { getActiveBackend } from "#/api/backend-registry/active-store";

/**
 * First agent-server release whose *profile* model accepts
 * `enable_switch_llm_tool`; the settings schema advertised it earlier, and the
 * profile model is `extra="forbid"`.
 */
export const MIN_AGENT_SERVER_VERSION_FOR_PROFILE_SWITCH_LLM_TOOL = "1.31.0";

/** Whether the profile model accepts `enable_switch_llm_tool`; unknown counts as yes. */
export function agentProfileSupportsSwitchLlmTool(): boolean {
  const version = getCachedAgentServerVersion();
  if (!version) return true;
  const comparison = compareAgentServerVersions(
    version,
    MIN_AGENT_SERVER_VERSION_FOR_PROFILE_SWITCH_LLM_TOOL,
  );
  if (comparison === null) return true;
  return comparison >= 0;
}

/** Only offer a scope when the serving backend advertises enforcement. */
export function agentProfileSupportsSecretRefs(): boolean {
  if (getActiveBackend().backend.kind === "cloud") return false;
  const capabilities = getCachedAgentServerInfo()?.capabilities;
  return (
    Array.isArray(capabilities) &&
    capabilities.includes("profile_secret_scope_v1")
  );
}

/** Whether the active backend serves the tool catalog the picker needs. */
export function agentProfileSupportsToolCatalog(): boolean {
  if (getActiveBackend().backend.kind === "cloud") return false;
  const capabilities = getCachedAgentServerInfo()?.capabilities;
  return (
    Array.isArray(capabilities) && capabilities.includes("tool_catalog_v1")
  );
}
