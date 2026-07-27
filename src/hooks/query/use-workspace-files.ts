import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";

// Cap the number of files we render so a giant repo doesn't freeze the UI.
const MAX_FILES = 2000;

export interface WorkspaceFilesResult {
  data: string[] | undefined;
  isLoading: boolean;
}

// Directory names that we never want to descend into when listing files.
const EXCLUDED_DIRS = [
  ".git",
  "node_modules",
  ".venv",
  "venv",
  "__pycache__",
  "dist",
  "build",
  ".next",
  ".cache",
  ".pytest_cache",
  ".mypy_cache",
  ".turbo",
  ".parcel-cache",
  "target",
];

// Build a `find` invocation that lists files relative to the workspace root.
function buildListCommand(): string {
  const pruneExpr = EXCLUDED_DIRS.map((dir) => `-name '${dir}' -prune`).join(
    " -o ",
  );
  return `find . \\( ${pruneExpr} \\) -o -type f -print 2>/dev/null | sort | head -n ${MAX_FILES}`;
}

function normalizePath(path: string): string {
  // Strip a leading "./" so paths render cleanly in the UI.
  return path.startsWith("./") ? path.slice(2) : path;
}

/**
 * Local-backend listing: enumerate every regular file beneath the active
 * conversation's working directory via `find` over the agent-server's
 * `/api/bash/execute_bash_command`, excluding common heavy/build directories.
 * Returns paths relative to the working dir (e.g. `src/index.html`).
 *
 * Local only: the cloud API exposes no bash-exec / file-listing endpoint,
 * and the old cross-origin `/api/cloud-proxy` hop these calls relied on was
 * removed from the agent-server. See `useWorkspaceFiles` for the cloud path.
 */
function useLocalWorkspaceFiles(enabled: boolean): WorkspaceFilesResult {
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady();

  const conversationId = conversation?.id;
  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;
  const workingDir = conversation?.workspace?.working_dir?.trim();

  const query = useQuery<string[]>({
    queryKey: [
      "workspace-files",
      conversationId,
      conversationUrl,
      sessionApiKey,
      workingDir,
    ],
    queryFn: async () => {
      const result = await AgentServerRuntimeService.executeCommand(
        conversationUrl,
        sessionApiKey,
        buildListCommand(),
        workingDir,
        30,
      );

      if (result.exit_code !== 0) {
        throw new Error(
          result.stderr?.trim() || "Failed to list workspace files",
        );
      }

      const lines = result.stdout
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .map(normalizePath);

      // Defensive: keep results unique and bounded.
      return Array.from(new Set(lines)).slice(0, MAX_FILES);
    },
    enabled: enabled && runtimeIsReady && !!conversationId && !!workingDir,
    retry: false,
    staleTime: 1000 * 30,
    gcTime: 1000 * 60 * 5,
    meta: { disableToast: true },
  });

  return { data: query.data, isLoading: query.isLoading };
}

/**
 * Cloud-backend listing: derive the file list from the conversation's git
 * changes — the same data source the diff view uses (and the only
 * runtime-workspace transport the cloud API proxies, alongside git diff and
 * single-file read). `git status` reports created/modified/untracked files,
 * which covers the common Agent Canvas case (a fresh or agent-authored
 * workspace). It intentionally does NOT enumerate unchanged tracked files —
 * the cloud API has no full-workspace listing endpoint — so a conversation
 * attached to a large existing repo shows changed files rather than the whole
 * tree. Deleted files are dropped since they can't be opened.
 */
function useCloudWorkspaceFiles(enabled: boolean): WorkspaceFilesResult {
  const gitChanges = useUnifiedGetGitChanges();

  const data = useMemo(() => {
    if (!enabled) return undefined;
    const paths = gitChanges.data
      .filter((change) => change.status !== "D")
      .map((change) => change.path);
    return Array.from(new Set(paths)).slice(0, MAX_FILES);
  }, [enabled, gitChanges.data]);

  return {
    data: enabled ? data : undefined,
    isLoading: enabled ? gitChanges.isLoading : false,
  };
}

/**
 * Lists the files shown in the Files tab for the active conversation.
 *
 * Local backends enumerate the full workspace tree via bash `find`. Cloud
 * backends derive the list from git changes (see `useCloudWorkspaceFiles`
 * for the rationale and its limitation), because the cloud API exposes no
 * bash-exec or file-listing endpoint.
 */
export function useWorkspaceFiles(): WorkspaceFilesResult {
  const { backend } = useActiveBackend();
  const isCloud = backend.kind === "cloud";

  const local = useLocalWorkspaceFiles(!isCloud);
  const cloud = useCloudWorkspaceFiles(isCloud);

  return isCloud ? cloud : local;
}
