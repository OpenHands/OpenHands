export interface LocalWorkspace {
  id: string;
  name: string;
  path: string;
  /**
   * If this workspace was dynamically derived from a workspace parent,
   * the path of the parent that produced it. Static workspaces leave this
   * unset.
   */
  parentPath?: string;
}

/**
 * A directory whose immediate subdirectories should be listed dynamically
 * as workspaces. Saved alongside `LocalWorkspace`s but not surfaced as a
 * selectable workspace itself.
 */
export interface LocalWorkspaceParent {
  id: string;
  name: string;
  path: string;
}
