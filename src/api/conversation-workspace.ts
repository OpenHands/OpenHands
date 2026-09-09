import { ServerClient } from "@openhands/typescript-client/clients";
import type { ServerInfo } from "@openhands/typescript-client";
import {
  getAgentServerClientOptions,
  assertConversationRuntimeClientSupport,
  type AgentServerClientOverrides,
} from "./agent-server-client-options";
import { getCachedAgentServerInfo } from "./agent-server-compatibility";
import {
  buildConversationWorkingDirForBackend,
  getWorkspaceRootForBackend,
} from "./agent-server-config";
import { resolveAbsoluteAgentServerPath } from "./agent-server-home";

export const ISOLATED_WORKSPACE_MESSAGE =
  "This backend runs conversations in isolated Docker workspaces at /workspace. Host folders and projects cannot be attached. Clear the host project selection to start in a new isolated workspace.";

export function usesIsolatedWorkspace(
  info: ServerInfo | null | undefined,
): boolean {
  return (
    info?.conversation_runtime === "docker" ||
    info?.workspace_mode === "isolated"
  );
}

export async function getConversationServerInfo(
  overrides: AgentServerClientOverrides = {},
): Promise<ServerInfo> {
  const options = getAgentServerClientOptions(overrides);
  return (
    getCachedAgentServerInfo({ host: options.host }) ??
    new ServerClient(options).getServerInfo()
  );
}

export async function resolveNewConversationWorkspace(options: {
  conversationId: string;
  workingDir?: string;
  selectedRepository?: string | null;
  parentConversationId?: string;
}) {
  const clientOptions = getAgentServerClientOptions();
  const info = await getConversationServerInfo(clientOptions);
  if (usesIsolatedWorkspace(info)) {
    if (
      options.selectedRepository ||
      (options.workingDir !== undefined &&
        !(options.parentConversationId && options.workingDir === "/workspace"))
    ) {
      throw new Error(ISOLATED_WORKSPACE_MESSAGE);
    }
    assertConversationRuntimeClientSupport();
    return { workingDir: "/workspace", hooksProjectDir: null, isolated: true };
  }
  // @spec WUP-001 — Resolve relative local defaults against the backend home.
  const base =
    options.workingDir ??
    buildConversationWorkingDirForBackend(
      options.conversationId,
      clientOptions.host,
    );
  const workingDir = await resolveAbsoluteAgentServerPath(base);
  const hooksProjectDir = options.workingDir
    ? workingDir
    : await resolveAbsoluteAgentServerPath(
        getWorkspaceRootForBackend(clientOptions.host),
      );
  return { workingDir, hooksProjectDir, isolated: false };
}
