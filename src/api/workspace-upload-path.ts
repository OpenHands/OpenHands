// @spec WUP-001 — Resolve relative working dirs against /api/file/home
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { getAgentServerWorkingDir } from "#/api/agent-server-config";
import {
  type AgentServerClientOverrides,
  getAgentServerClientOptions,
} from "#/api/agent-server-client-options";
import { resolveAbsoluteAgentServerPath } from "#/api/agent-server-home";
import { getStoredConversationMetadata } from "#/api/conversation-metadata-store";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function getSafeUploadFileName(fileName: string): string {
  const parts = fileName.split(/[\\/]+/).filter(Boolean);
  const safeName = parts[parts.length - 1];

  if (!safeName || safeName === "." || safeName === "..") {
    throw new Error("Invalid file name");
  }

  return safeName;
}

/**
 * Resolve `workingDir` to an absolute path the agent-server's file APIs
 * accept. Relative paths are joined against `/api/file/home` (cached per
 * backend); absolute paths pass through.
 */
export async function resolveAbsoluteWorkspacePath(
  workingDir: string,
  overrides: AgentServerClientOverrides = {},
): Promise<string> {
  return resolveAbsoluteAgentServerPath(workingDir, overrides);
}

/**
 * Build the absolute destination path for a file upload, resolving the
 * working-dir leg via {@link resolveAbsoluteWorkspacePath}.
 */
export async function buildWorkspaceUploadPath(
  fileName: string,
  workingDir: string,
  overrides: AgentServerClientOverrides = {},
): Promise<string> {
  const safeName = getSafeUploadFileName(fileName);
  const absoluteDir = await resolveAbsoluteWorkspacePath(workingDir, overrides);
  return `${absoluteDir.replace(/[/\\]+$/, "")}/${safeName}`;
}

/**
 * Resolve the working directory for a file upload into a conversation's
 * workspace.
 *
 * **Returns the raw working dir string** — which may be relative (e.g.
 * `workspace/project/<hex>`) when the default `DEFAULT_WORKING_DIR` is in
 * use. Callers that need an actual filesystem-absolute path (e.g. to pass to
 * the agent-server's `/api/file/upload` endpoint) **must** funnel this result
 * through {@link buildWorkspaceUploadPath}, which calls
 * {@link resolveAbsoluteWorkspacePath} to anchor any relative segment against
 * the agent-server's home directory.
 *
 * Why not resolve here? Because this function is also called by cloud-runtime
 * upload paths where the overrides (conversationUrl, sessionApiKey) aren't
 * available until {@link uploadFilesToConversation} assembles them. Keeping
 * the resolution step in {@link buildWorkspaceUploadPath} means both the
 * local and cloud legs share a single resolution point with the correct
 * override context.
 */
export async function resolveConversationUploadWorkingDir(
  conversationId: string,
  currentConversation?: AppConversation | null,
): Promise<string> {
  if (
    currentConversation?.id === conversationId &&
    currentConversation.workspace?.working_dir?.trim()
  ) {
    return currentConversation.workspace.working_dir.trim();
  }

  const stored = getStoredConversationMetadata(conversationId);
  if (stored?.selected_workspace?.trim()) {
    return stored.selected_workspace.trim();
  }

  if (UUID_PATTERN.test(conversationId)) {
    return AgentServerConversationService.resolveConversationWorkingDir(
      conversationId,
    );
  }

  return getAgentServerWorkingDir();
}

// Re-export so callers can construct overrides matching what
// {@link buildWorkspaceUploadPath} expects without importing two modules.
export type { AgentServerClientOverrides };
export { getAgentServerClientOptions };
