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
