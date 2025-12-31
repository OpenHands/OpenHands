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
