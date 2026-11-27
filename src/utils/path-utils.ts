/**
 * Path manipulation utilities
 */
import { DEFAULT_WORKING_DIR } from "#/api/agent-server-config";

/**
 * Strip workspace prefix from file paths
 * Removes /workspace/ and the next directory level from paths
 *
 * @param path - The file path to process
 * @returns The path with workspace prefix removed
 *
 * @example
 * stripWorkspacePrefix("/workspace/repo/src/file.py") // returns "src/file.py"
 * stripWorkspacePrefix("/workspace/my-project/components/Button.tsx") // returns "components/Button.tsx"
 */
export const stripWorkspacePrefix = (path: string): string => {
  // Strip /workspace/ and the next directory level
  const workspaceMatch = path.match(/^\/workspace\/[^/]+\/(.*)$/);
  return workspaceMatch ? workspaceMatch[1] : path;
};

function normalizeWorkspaceRoot(root: string): string {
  const trimmed = root.trim().replace(/\\/g, "/").replace(/\/+$/, "");
  if (!trimmed) return "";
  // Keep Windows drive roots (`C:` / `C:/repo`) and POSIX absolutes as-is.
  if (trimmed.startsWith("/") || /^[A-Za-z]:(\/|$)/.test(trimmed)) {
    return trimmed;
  }
  return `/${trimmed}`;
}

/**
 * Convert an agent/chat file path into a workspace-relative path for the Files
 * drawer / workspace file APIs.
 *
 * Strips editor `:line` / `:start-end` suffixes, then removes the conversation
 * working directory (or `DEFAULT_WORKING_DIR`) when present as a prefix.
 * Nested roots must match `workingDir` first — the generic `/workspace/<name>/`
 * heuristic is only a fallback when no root matches.
 */
export const toFilesTabPath = (
  path: string,
  workingDir?: string | null,
): string => {
  let result = path.trim().replace(/\\/g, "/");
  if (!result) return "";

  // Strip editor `:12` / `:12-40` suffixes. Safe on Windows paths too —
  // the drive colon is at the start (`C:`), never at the end.
  result = result.replace(/:(\d+)(-\d+)?$/, "");

  const roots = [workingDir, DEFAULT_WORKING_DIR].filter(
    (value): value is string => Boolean(value?.trim()),
  );

  for (const root of roots) {
    const normalizedRoot = normalizeWorkspaceRoot(root);
    if (!normalizedRoot) continue;
    if (result === normalizedRoot) {
      return "";
    }
    if (result.startsWith(`${normalizedRoot}/`)) {
      return result.slice(normalizedRoot.length + 1);
    }
  }

  if (!result.startsWith("/")) {
    return result.replace(/^\.\//, "");
  }

  return stripWorkspacePrefix(result).replace(/^\.\//, "");
};

const WORKSPACE_FILE_EXTENSION =
  /\.(md|txt|ts|tsx|js|jsx|mjs|cjs|py|json|html?|css|scss|ya?ml|toml|rs|go|java|kt|swift|c|cc|cpp|h|hpp|sh|bash|zsh|sql|xml|svg|pdf|env|rb|php|vue|svelte|lock|ini|cfg|docx?|xlsx?|pptx?|odt|rtf)$/i;

/**
 * Conservative check for inline chat tokens that should open in Files.
 * Rejects URLs, MIME types, versions, and dotted identifiers like `console.log`.
 */
export const looksLikeWorkspaceFilePath = (text: string): boolean => {
  const trimmed = text.trim();
  if (!trimmed || /\s/.test(trimmed) || /^https?:\/\//i.test(trimmed)) {
    return false;
  }

  let path = trimmed.replace(/\\/g, "/").replace(/^\.\//, "");
  path = path.replace(/:(\d+)(-\d+)?$/, "");

  if (
    /^(application|audio|image|text|video|font|multipart|message|model)\/[\w.+-]+$/i.test(
      path,
    )
  ) {
    return false;
  }
  if (/^v?\d+(\.\d+){1,3}([-+][\w.]+)?$/i.test(path)) {
    return false;
  }

  const lastSegment = path.includes("/") ? (path.split("/").pop() ?? "") : path;
  return WORKSPACE_FILE_EXTENSION.test(lastSegment);
};

/**
 * Returns the basename (top-level folder/file name) from a path string,
 * tolerating POSIX and Windows separators and trailing slashes.
 */
export const getPathBasename = (path: string): string => {
  const trimmed = path.trim();
  if (!trimmed) return "";

  const normalized = trimmed.replace(/[\\/]+$/, "");
  if (!normalized || /^[A-Za-z]:$/.test(normalized)) return "";

  const idx = Math.max(
    normalized.lastIndexOf("/"),
    normalized.lastIndexOf("\\"),
  );
  return idx >= 0 ? normalized.slice(idx + 1) : normalized;
};
