export interface FileTreeNode {
  name: string;
  path: string; // full path relative to workspace root
  isDirectory: boolean;
  children: FileTreeNode[];
}

function sortTreeInPlace(node: FileTreeNode): void {
  node.children.sort((a, b) => {
    if (a.isDirectory !== b.isDirectory) return a.isDirectory ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
  for (const child of node.children) {
    if (child.isDirectory) sortTreeInPlace(child);
  }
}

/**
 * Build a tree representation of a flat list of relative file paths.
 * Directories are sorted before files; siblings are sorted alphabetically.
 *
 * Implementation notes:
 *   - We keep a side-table mapping each parent node to its own
 *     `Map<name, child>` so adding a segment is O(1) instead of the
 *     O(n) `children.find(...)` linear scan we used to do. With a
 *     directory containing 1000 siblings, the old code did ~500k string
 *     comparisons just to bucket them all in; the Map cuts that to
 *     ~1000. The side-table lives only for the duration of the build
 *     call so the public FileTreeNode shape stays clean.
 *   - If we encounter a path like `"src"` followed by `"src/index.ts"`,
 *     the first entry creates `src` as a file (`isDirectory: false`),
 *     then the second needs to add a child to it. We promote the leaf
 *     to a directory in that case rather than dropping the child on
 *     the floor. (The flat input shouldn't normally contain both, but
 *     defensive coding here is cheap and keeps the tree well-formed.)
 */

function getOrCreateChild(
  parent: FileTreeNode,
  childMap: Map<FileTreeNode, Map<string, FileTreeNode>>,
  segment: string,
  prefix: string,
  isLast: boolean,
): FileTreeNode {
  let map = childMap.get(parent);
  if (!map) {
    map = new Map();
    childMap.set(parent, map);
  }
  const existing = map.get(segment);
  if (existing) {
    // Promote a leaf to a directory if we're about to descend into it.
    if (!isLast && !existing.isDirectory) {
      existing.isDirectory = true;
    }
    return existing;
  }
  const node: FileTreeNode = {
    name: segment,
    path: prefix,
    isDirectory: !isLast,
    children: [],
  };
  parent.children.push(node);
  map.set(segment, node);
  return node;
}

export function buildFileTree(paths: string[]): FileTreeNode {
  const root: FileTreeNode = {
    name: "",
    path: "",
    isDirectory: true,
    children: [],
  };

  // Side-table mapping a parent node to its child lookup map. Keeping the
  // accelerator out of the FileTreeNode shape itself means consumers never
  // see it — no post-build cleanup pass, and the public type stays clean.
  const childMap = new Map<FileTreeNode, Map<string, FileTreeNode>>();

  for (const path of paths) {
    const segments = path.split("/").filter(Boolean);
    if (segments.length > 0) {
      let cursor: FileTreeNode = root;
      let prefix = "";
      for (let i = 0; i < segments.length; i += 1) {
        const segment = segments[i];
        prefix = prefix ? `${prefix}/${segment}` : segment;
        const isLast = i === segments.length - 1;
        cursor = getOrCreateChild(cursor, childMap, segment, prefix, isLast);
      }
    }
  }

  sortTreeInPlace(root);
  return root;
}
