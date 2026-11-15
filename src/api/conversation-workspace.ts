import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import { ServerClient } from "@openhands/typescript-client/clients";
import type { ServerInfo } from "@openhands/typescript-client";
import {
  getAgentServerClientOptions,
  type AgentServerClientOverrides,
} from "./agent-server-client-options";
import { getCachedAgentServerInfo } from "./agent-server-compatibility";
import {
  buildConversationWorkingDirForBackend,
  getWorkspaceRootForBackend,
} from "./agent-server-config";
import { resolveAbsoluteAgentServerPath } from "./agent-server-home";

const ISOLATED_WORKSPACE_DIR = "/workspace";

export function usesIsolatedWorkspace(
  info: ServerInfo | null | undefined,
): boolean {
  return info?.conversation_runtime === "docker";
}

export async function getConversationServerInfo(
  overrides: AgentServerClientOverrides = {},
): Promise<ServerInfo | null> {
  const options = getAgentServerClientOptions(overrides);
  const cached = getCachedAgentServerInfo({ host: options.host });
  if (cached) return cached;
  try {
    return await new ServerClient(options).getServerInfo();
  } catch {
    // An unreachable or non-responding agent-server is not fatal: callers
    // fall back to host-workspace behavior, matching how the rest of the
    // app treats an unavailable `server_info` (see
    // `fetchBackendServerInfo`). Only a *successful* probe that reports
    // `conversation_runtime: "docker"` may isolate the workspace.
    return null;
  }
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
    if (options.selectedRepository) {
      throw new Error(i18n.t(I18nKey.HOME$ISOLATED_WORKSPACE_NOTICE));
    }
    // The docker runtime bind-mounts a host `working_dir` at /workspace in the
    // conversation container, so an attached host folder is still usable and
    // its `.openhands/hooks.json` is read from the host side.
    if (
      options.workingDir !== undefined &&
      options.workingDir !== ISOLATED_WORKSPACE_DIR
    ) {
      const hostDir = await resolveAbsoluteAgentServerPath(options.workingDir);
      return { workingDir: hostDir, hooksProjectDir: hostDir, isolated: true };
    }
    return {
      workingDir: ISOLATED_WORKSPACE_DIR,
      hooksProjectDir: null,
      isolated: true,
    };
  }
  // @spec WUP-001 — Resolve relative local defaults against the backend home.
  const base =
    options.workingDir ??
    buildConversationWorkingDirForBackend(
      options.conversationId,
      clientOptions.host,
    );
  const workingDir = await resolveAbsoluteAgentServerPath(base);
  const hooksProjectDir = await resolveHooksProjectDir(
    options.workingDir,
    clientOptions.host,
  );
  return { workingDir, hooksProjectDir, isolated: false };
}

/**
 * Workspace root whose `.openhands/hooks.json` a host-workspace conversation
 * loads: the folder the user attached, else the backend's workspace root,
 * never the per-conversation working dir or worktree (#16907).
 */
function resolveHooksProjectDir(
  selectedWorkspace: string | null | undefined,
  host: string,
): Promise<string> {
  return resolveAbsoluteAgentServerPath(
    selectedWorkspace || getWorkspaceRootForBackend(host),
  );
}

/**
 * The hooks project dir of an existing conversation, resolved as
 * `resolveNewConversationWorkspace()` resolved it at start from the workspace
 * the user attached. On an isolated (Docker) backend, `null` unless a host
 * folder was attached (it is bind-mounted into the conversation container).
 */
export async function resolveConversationHooksProjectDir(
  selectedWorkspace: string | null | undefined,
): Promise<string | null> {
  const clientOptions = getAgentServerClientOptions();
  const info = await getConversationServerInfo(clientOptions);
  if (usesIsolatedWorkspace(info)) {
    if (!selectedWorkspace || selectedWorkspace === ISOLATED_WORKSPACE_DIR) {
      return null;
    }
    return resolveAbsoluteAgentServerPath(selectedWorkspace);
  }
  return resolveHooksProjectDir(selectedWorkspace, clientOptions.host);
}
