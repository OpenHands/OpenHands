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

  // Keep Windows drive letters; only strip editor `:12` / `:12-40` ranges.
  if (!/^[A-Za-z]:(\/|$)/.test(result)) {
    result = result.replace(/:(\d+)(-\d+)?$/, "");
  }

  result = stripWorkspacePrefix(result);

  const wd = workingDir?.trim().replace(/\\/g, "/").replace(/\/+$/, "");
  if (wd && result.startsWith(`${wd}/`)) {
    result = result.slice(wd.length + 1);
  } else if (wd && result === wd) {
    return "";
  }

  return result.replace(/^\.\//, "");
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
