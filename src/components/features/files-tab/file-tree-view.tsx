import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import { I18nKey } from "#/i18n/declaration";
import { buildFileTree } from "#/utils/file-tree";
import { TreeNode } from "./tree-node";

interface FileTreeViewProps {
  paths: string[];
  selectedPath: string | null;
  onSelectFile: (path: string) => void;
}

export function FileTreeView({
  paths,
  selectedPath,
  onSelectFile,
}: FileTreeViewProps) {
  const { t } = useTranslation("openhands");
  const root = useMemo(() => buildFileTree(paths), [paths]);

  if (root.children.length === 0) {
    return (
      <div className="px-3 py-4 text-sm text-[var(--oh-muted)]">
        {t(I18nKey.FILES$NO_FILES)}
      </div>
    );
  }

  return (
    <ul className="py-1 custom-scrollbar-always" data-testid="file-tree-view">
      {root.children.map((child) => (
        <TreeNode
          key={child.path}
          node={child}
          depth={0}
          selectedPath={selectedPath}
          onSelectFile={onSelectFile}
        />
      ))}
    </ul>
  );
}
