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
        !(
          options.parentConversationId &&
          options.workingDir === ISOLATED_WORKSPACE_DIR
        ))
    ) {
      throw new Error(i18n.t(I18nKey.HOME$ISOLATED_WORKSPACE_NOTICE));
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
  const hooksProjectDir = options.workingDir
    ? workingDir
    : await resolveAbsoluteAgentServerPath(
        getWorkspaceRootForBackend(clientOptions.host),
      );
  return { workingDir, hooksProjectDir, isolated: false };
}
