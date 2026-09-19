import { useState } from "react";
import type { GitChangeStatus } from "#/api/open-hands.types";
import { FileDiffViewer } from "./file-diff-viewer";

export interface DiffChangeListItem {
  path: string;
  status: GitChangeStatus;
}

export interface DiffChangeListProps {
  changes: DiffChangeListItem[];
  /**
   * When set, each row shows that commit's version of the file instead of
   * the working-tree diff.
   */
  commit?: string;
}

/**
 * Single-open accordion of file diffs. Expanding one path collapses the
 * previously open one (same behavior as the Commits list).
 */
export function DiffChangeList({ changes, commit }: DiffChangeListProps) {
  const [expandedPath, setExpandedPath] = useState<string | null>(null);

  return (
    <div data-testid="diff-change-list" className="w-full flex flex-col">
      {changes.map((change) => (
        <FileDiffViewer
          key={change.path}
          path={change.path}
          type={change.status}
          commit={commit}
          isExpanded={expandedPath === change.path}
          onToggle={() =>
            setExpandedPath((prev) =>
              prev === change.path ? null : change.path,
            )
          }
        />
      ))}
    </div>
  );
}
