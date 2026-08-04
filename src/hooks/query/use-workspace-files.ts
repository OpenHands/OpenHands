import { useQuery } from "@tanstack/react-query";

import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { getGitPath } from "#/utils/get-git-path";

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
 * Cloud sandboxes require absolute cwd paths. Local agent-server accepts the
 * configured working dir as-is (often the relative `workspace/project`
 * convention). `getGitPath` returns that relative form when the conversation
 * omits `workspace.working_dir`.
 */
function resolveWorkspaceCwd(
  selectedRepository: string | null | undefined,
  workingDir: string | null | undefined,
  isCloud: boolean,
): string {
  const path = getGitPath(selectedRepository, workingDir);
  if (isCloud && !path.startsWith("/")) {
    return `/${path}`;
  }
  return path;
}

/**
 * Enumerate every regular file beneath the active conversation's working
 * directory via `find` over `/api/bash/execute_bash_command`, excluding
 * common heavy/build directories. Returns paths relative to the working dir
 * (e.g. `src/index.html`).
 *
 * Local: SDK `RemoteWorkspace.executeCommand`.
 * Cloud: same bash endpoint through Canvas `/api/cloud-proxy` → runtime
 * sandbox (`AgentServerRuntimeService.executeCommand` hostOverride hop).
 */
function useBashWorkspaceFiles(enabled: boolean): WorkspaceFilesResult {
  const { backend } = useActiveBackend();
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady();

  const isCloud = backend.kind === "cloud";
  const conversationId = conversation?.id;
  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;
  const workingDir = resolveWorkspaceCwd(
    conversation?.selected_repository,
    conversation?.workspace?.working_dir,
    isCloud,
  );

  const query = useQuery<string[]>({
    queryKey: [
      "workspace-files",
      conversationId,
      conversationUrl,
      sessionApiKey,
      workingDir,
      backend.kind,
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
    enabled:
      enabled &&
      runtimeIsReady &&
      !!conversationId &&
      !!workingDir &&
      (!isCloud || !!conversationUrl),
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
 * Both Local and Cloud backends enumerate the full workspace tree via bash
 * `find`. Cloud calls hop through Canvas `/api/cloud-proxy` to the sandbox.
 */
export function useWorkspaceFiles(): WorkspaceFilesResult {
  return useBashWorkspaceFiles(true);
}
