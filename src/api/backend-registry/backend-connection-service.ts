import { ServerClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import {
  assertAgentServerVersionIsSupported,
  getDisplayAgentServerVersion,
} from "#/api/agent-server-compatibility";
import type { Backend } from "./types";

/** Timeout applied to the short preflight/status probes, in milliseconds. */
const CONNECTION_PROBE_TIMEOUT_MS = 5000;

export interface BackendConnectionTestMetadata {
  agentServerVersion: string | null;
}

/** The subset of a backend needed to reach its agent server. */
export type AgentServerTarget = Pick<Backend, "host" | "apiKey">;

/** {@link AgentServerTarget} plus the kind, for flows that branch on it. */
export type BackendConnectionTarget = AgentServerTarget & Pick<Backend, "kind">;

/**
 * The one place the backend-registry UI constructs a `ServerClient`.
 *
 * The add/edit form's submit preflight and both version badges go through
 * here, so their client options and timeout cannot drift apart. Two other
 * probes in the app deliberately stay separate, because their semantics
 * differ rather than merely their spelling: `useBackendsHealth` polls with a
 * shorter timeout and checks settings before server info, and
 * `loadAgentServerInfo` is the cached bootstrap probe that also validates the
 * session key.
 */
function getServerInfo(backend: AgentServerTarget) {
  return new ServerClient(
    getAgentServerClientOptions({
      host: backend.host,
      sessionApiKey: backend.apiKey || null,
      timeout: CONNECTION_PROBE_TIMEOUT_MS,
    }),
  ).getServerInfo();
}

/**
 * Preflight a backend before it is persisted. Throws when the server is
 * unreachable or reports an unsupported version.
 */
export async function testBackendConnection(
  backend: BackendConnectionTarget,
): Promise<BackendConnectionTestMetadata> {
  // Cloud backends authenticate via OAuth; preflight GET is not applicable.
  if (backend.kind !== "local") return { agentServerVersion: null };

  const serverInfo = await getServerInfo(backend);
  assertAgentServerVersionIsSupported(serverInfo);
  return { agentServerVersion: getDisplayAgentServerVersion(serverInfo) };
}

/**
 * Read the agent server's reported version for display.
 *
 * Unlike {@link testBackendConnection} this does not assert the version floor:
 * the status badges report whatever the server claims, and callers gate on
 * `kind === "local"` themselves.
 */
export async function fetchAgentServerVersion(
  backend: AgentServerTarget,
): Promise<string | null> {
  const serverInfo = await getServerInfo(backend);
  return getDisplayAgentServerVersion(serverInfo);
}
