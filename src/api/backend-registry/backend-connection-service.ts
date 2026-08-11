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
export type BackendConnectionTarget = Pick<Backend, "host" | "apiKey" | "kind">;

/**
 * The single place a `ServerClient` is constructed for backend probes.
 *
 * UI components must not build SDK clients directly — routing both the submit
 * preflight and the edit-form status row through here keeps the client
 * options (and the timeout) identical for the two calls.
 */
function getServerInfo(backend: BackendConnectionTarget) {
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
 * the status row reports whatever the server claims, and callers gate on
 * `kind === "local"` themselves.
 */
export async function fetchAgentServerVersion(
  backend: BackendConnectionTarget,
): Promise<string | null> {
  const serverInfo = await getServerInfo(backend);
  return getDisplayAgentServerVersion(serverInfo);
}
