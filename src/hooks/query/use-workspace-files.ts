import { useQuery } from "@tanstack/react-query";

import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";

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
 * Enumerates every regular file beneath the active conversation's working
 * directory via `find` over the agent-server's `/api/bash/execute_bash_command`,
 * excluding common heavy/build directories. Works for both local and cloud
 * backends — the cloud path routes through `callCloudProxy` so CORS is not an
 * issue. Returns paths relative to the working dir (e.g. `src/index.html`).
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
 * Lists the files shown in the Files tab for the active conversation.
 *
 * Both local and cloud backends enumerate the full workspace tree via bash
 * `find` over the agent-server's `/api/bash/execute_bash_command` endpoint.
 * The cloud path routes through `callCloudProxy` so CORS is not an issue.
 */
export function useWorkspaceFiles(): WorkspaceFilesResult {
  return useLocalWorkspaceFiles(true);
}
