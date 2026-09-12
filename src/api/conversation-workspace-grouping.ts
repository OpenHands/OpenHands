import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { isProvider, type Provider } from "#/types/settings";

export const REPOSITORY_TAG_KEY = "repository";
export const SELECTED_BRANCH_TAG_KEY = "selected_branch";
export const GIT_PROVIDER_TAG_KEY = "git_provider";
export const WORKSPACE_TAG_KEY = "workspace";

export function normalizeWorkspacePath(
  path: string | null | undefined,
): string {
  const trimmed = path?.trim();
  if (!trimmed) return "";
  return trimmed.replace(/\/+$/, "");
}

export function gitProviderFromTag(value: string | undefined): Provider | null {
  if (value && isProvider(value)) {
    return value;
  }
  return null;
}

/**
 * Resolve the workspace path used for sidebar grouping. Prefer explicit
 * client metadata (already on the conversation), then server tags, then
 * a direct match of runtime `working_dir` against registered workspaces
 * for conversations created before tags existed.
 */
export function resolveSelectedWorkspaceForGrouping(
  conversation: Pick<
    AppConversation,
    "selected_workspace" | "workspace" | "tags"
  >,
  registeredWorkspacePaths?: readonly string[],
): string | null {
  const fromMetadata = normalizeWorkspacePath(conversation.selected_workspace);
  if (fromMetadata) {
    return fromMetadata;
  }

  const fromTag = normalizeWorkspacePath(
    conversation.tags?.[WORKSPACE_TAG_KEY],
  );
  if (fromTag) {
    return fromTag;
  }

  const runtimeDir = normalizeWorkspacePath(
    conversation.workspace?.working_dir,
  );
  if (!runtimeDir || !registeredWorkspacePaths?.length) {
    return null;
  }

  const registered = new Set(
    registeredWorkspacePaths
      .map((path) => normalizeWorkspacePath(path))
      .filter(Boolean),
  );
  return registered.has(runtimeDir) ? runtimeDir : null;
}

export function buildConversationMetadataTags(options: {
  selectedRepository?: string | null;
  selectedBranch?: string | null;
  gitProvider?: Provider | null;
  selectedWorkspace?: string | null;
}): Record<string, string> {
  const tags: Record<string, string> = {};
  if (options.selectedRepository) {
    tags[REPOSITORY_TAG_KEY] = options.selectedRepository;
  }
  if (options.selectedBranch) {
    tags[SELECTED_BRANCH_TAG_KEY] = options.selectedBranch;
  }
  if (options.gitProvider) {
    tags[GIT_PROVIDER_TAG_KEY] = options.gitProvider;
  }
  const workspace = normalizeWorkspacePath(options.selectedWorkspace);
  if (workspace) {
    tags[WORKSPACE_TAG_KEY] = workspace;
  }
  return tags;
}
