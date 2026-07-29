/**
 * Path manipulation utilities
 */

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

/**
 * Normalize a tool/chat path for the Files tab: strip `/workspace/…`,
 * the conversation working dir, and optional `:line` suffixes.
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

  result = stripWorkspacePrefix(result);

  const wd = workingDir?.trim().replace(/\\/g, "/").replace(/\/+$/, "");
  if (wd && result.startsWith(`${wd}/`)) {
    result = result.slice(wd.length + 1);
  } else if (wd && result === wd) {
    return "";
  }

  return result.replace(/^\.\//, "");
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
