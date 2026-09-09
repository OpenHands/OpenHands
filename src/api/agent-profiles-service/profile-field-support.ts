import {
  compareAgentServerVersions,
  getCachedAgentServerVersion,
} from "#/api/agent-server-compatibility";

/**
 * `enable_switch_llm_tool` reached `OpenHandsAgentProfile` in
 * software-agent-sdk#3870, first released in agent-server 1.31.0. The same
 * field reached `AgentSettingsConfig` far earlier (1.22.0), so the agent
 * settings schema advertises it on servers whose *profile* model still
 * rejects it.
 *
 * That gap sits inside the supported range rather than below it: agent
 * profiles shipped in 1.29.0 and Canvas accepts back to
 * `config/defaults.json`'s `compatibility.minimumAgentServer`. On 1.29.0
 * through 1.30.x, `AgentProfileBase` is `extra="forbid"`, so a profile save
 * carrying the key 422s and the whole save is lost. Hence the profile editor
 * gates on the server version instead of reusing the settings schema's
 * presence check, which answers a different question.
 */
export const MIN_AGENT_SERVER_VERSION_FOR_PROFILE_SWITCH_LLM_TOOL = "1.31.0";

/**
 * Whether the active backend's agent-profile model accepts
 * `enable_switch_llm_tool`.
 *
 * An unreadable version counts as support, matching the other version-gated
 * call sites: cloud backends expose no `/server_info` to probe and track the
 * current contract, and a local backend whose version does not parse never
 * gets past `assertAgentServerVersionIsSupported` at boot.
 */
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

/**
 * `tools` reached `OpenHandsAgentProfile` in software-agent-sdk#3968, first
 * released in agent-server 1.31.2 — the same release that taught
 * `_resolve_agent_from_profile` to skip its browser injection for a profile
 * that sets the field. Older servers in the supported range are `extra="forbid"`
 * on `AgentProfileBase`, so posting the key 422s and loses the whole save.
 */
export const MIN_AGENT_SERVER_VERSION_FOR_PROFILE_TOOLS = "1.31.2";

/** Whether the active backend's agent-profile model accepts `tools`. */
export function agentProfileSupportsTools(): boolean {
  const version = getCachedAgentServerVersion();
  if (!version) return true;
  const comparison = compareAgentServerVersions(
    version,
    MIN_AGENT_SERVER_VERSION_FOR_PROFILE_TOOLS,
  );
  if (comparison === null) return true;
  return comparison >= 0;
}

/**
 * `secret_refs` reaches `AgentProfileBase` in software-agent-sdk#4931. Pinned to
 * the next minor after v1.46.0; confirm against the release that actually ships
 * it. Older servers are `extra="forbid"`, so posting the key 422s the save.
 */
export const MIN_AGENT_SERVER_VERSION_FOR_PROFILE_SECRET_REFS = "1.47.0";

/** Whether the active backend's agent-profile model accepts `secret_refs`. */
export function agentProfileSupportsSecretRefs(): boolean {
  const version = getCachedAgentServerVersion();
  if (!version) return true;
  const comparison = compareAgentServerVersions(
    version,
    MIN_AGENT_SERVER_VERSION_FOR_PROFILE_SECRET_REFS,
  );
  if (comparison === null) return true;
  return comparison >= 0;
}
