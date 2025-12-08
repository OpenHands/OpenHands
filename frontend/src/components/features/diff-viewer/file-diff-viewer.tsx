import { DiffEditor, Monaco } from "@monaco-editor/react";
import React from "react";
import { editor as editor_t } from "monaco-editor";
import { LuFileDiff, LuFileMinus, LuFilePlus } from "react-icons/lu";
import { IconType } from "react-icons/lib";
import { GitChangeStatus } from "#/api/open-hands.types";
import { getLanguageFromPath } from "#/utils/get-language-from-path";
import { cn } from "#/utils/utils";
import ChevronUp from "#/icons/chveron-up.svg?react";
import { useUnifiedGitDiff } from "#/hooks/query/use-unified-git-diff";

interface LoadingSpinnerProps {
  className?: string;
}

// TODO: Move out of this file and replace the current spinner with this one
function LoadingSpinner({ className }: LoadingSpinnerProps) {
  return (
    <div className="flex items-center justify-center">
      <div
        className={cn(
          "animate-spin rounded-full border-4 border-gray-200 border-t-blue-500",
          className,
        )}
        role="status"
        aria-label="Loading"
      />
    </div>
  );
}

const STATUS_MAP: Record<GitChangeStatus, string | IconType> = {
  A: LuFilePlus,
  D: LuFileMinus,
  M: LuFileDiff,
  R: "Renamed",
  U: "Untracked",
};

export interface FileDiffViewerProps {
  path: string;
  type: GitChangeStatus;
}

export function FileDiffViewer({ path, type }: FileDiffViewerProps) {
  const [isCollapsed, setIsCollapsed] = React.useState(true);
  const [editorHeight, setEditorHeight] = React.useState(400);
  const diffEditorRef = React.useRef<editor_t.IStandaloneDiffEditor>(null);

  const isAdded = type === "A" || type === "U";
  const isDeleted = type === "D";

  const filePath = React.useMemo(() => {
    if (type === "R") {
      const parts = path.split(/\s+/).slice(1);
      return parts[parts.length - 1];
    }

    return path;
  }, [path, type]);

  const {
    data: diff,
    isLoading,
    isSuccess,
    isRefetching,
  } = useUnifiedGitDiff({
    filePath,
    type,
    enabled: !isCollapsed,
  });

  // Function to update editor height based on content
  const updateEditorHeight = React.useCallback(() => {
    if (diffEditorRef.current) {
      const originalEditor = diffEditorRef.current.getOriginalEditor();
      const modifiedEditor = diffEditorRef.current.getModifiedEditor();

      if (originalEditor && modifiedEditor) {
        // Get the content height from both editors and use the larger one
        const originalHeight = originalEditor.getContentHeight();
        const modifiedHeight = modifiedEditor.getContentHeight();
        const contentHeight = Math.max(originalHeight, modifiedHeight);

        // Add a small buffer to avoid scrollbar
        setEditorHeight(contentHeight + 20);
      }
    }
  }, []);

  const beforeMount = (monaco: Monaco) => {
    monaco.editor.defineTheme("custom-diff-theme", {
      base: "vs-dark",
      inherit: true,
      rules: [
        { token: "comment", foreground: "6a9955" },
        { token: "keyword", foreground: "569cd6" },
        { token: "string", foreground: "ce9178" },
        { token: "number", foreground: "b5cea8" },
      ],
      colors: {
        "diffEditor.insertedTextBackground": "#014b01AA", // Stronger green background
        "diffEditor.removedTextBackground": "#750000AA", // Stronger red background
        "diffEditor.insertedLineBackground": "#003f00AA", // Dark green for added lines
        "diffEditor.removedLineBackground": "#5a0000AA", // Dark red for removed lines
        "diffEditor.border": "#444444", // Border between diff editors

        "editorUnnecessaryCode.border": "#00000000", // No border for unnecessary code
        "editorUnnecessaryCode.opacity": "#00000077", // Slightly faded
      },
    });
  };

  const handleEditorDidMount = (editor: editor_t.IStandaloneDiffEditor) => {
    diffEditorRef.current = editor;
    updateEditorHeight();

    const originalEditor = editor.getOriginalEditor();
    const modifiedEditor = editor.getModifiedEditor();

    originalEditor.onDidContentSizeChange(updateEditorHeight);
    modifiedEditor.onDidContentSizeChange(updateEditorHeight);
  };

  const status = (type === "U" ? STATUS_MAP.A : STATUS_MAP[type]) || "?";

  let statusIcon: React.ReactNode;
  if (typeof status === "string") {
    statusIcon = <span>{status}</span>;
  } else {
    const StatusIcon = status; // now it's recognized as a component
    statusIcon = <StatusIcon className="w-5 h-5" />;
  }

  const isFetchingData = isLoading || isRefetching;

  return (
    <div data-testid="file-diff-viewer-outer" className="w-full flex flex-col">
      <div
        className={cn(
          "flex justify-between items-center px-2.5 py-3.5 border border-neutral-600 rounded-xl hover:cursor-pointer",
          !isCollapsed && !isLoading && "border-b-0 rounded-b-none",
        )}
        onClick={() => setIsCollapsed((prev) => !prev)}
      >
        <span className="text-sm w-full text-content flex items-center gap-2">
          {isFetchingData && <LoadingSpinner className="w-5 h-5" />}
          {!isFetchingData && statusIcon}
          <strong className="w-full truncate">{filePath}</strong>
          <button data-testid="collapse" type="button">
            <ChevronUp
              className={cn(
                "w-4 h-4 transition-transform",
                isCollapsed && "transform rotate-180",
              )}
            />
          </button>
        </span>
      </div>
      {isSuccess && !isCollapsed && (
        <div
          className="w-full border border-neutral-600 overflow-hidden"
          style={{ height: `${editorHeight}px` }}
        >
          <DiffEditor
            data-testid="file-diff-viewer"
            className="w-full h-full"
            language={getLanguageFromPath(filePath)}
            original={isAdded ? "" : diff.original}
            modified={isDeleted ? "" : diff.modified}
            theme="custom-diff-theme"
            onMount={handleEditorDidMount}
            beforeMount={beforeMount}
            options={{
              renderValidationDecorations: "off",
              readOnly: true,
              renderSideBySide: !isAdded && !isDeleted,
              scrollBeyondLastLine: false,
              minimap: {
                enabled: false,
              },
              hideUnchangedRegions: {
                enabled: true,
              },
              automaticLayout: true,
              scrollbar: {
                // Make scrollbar less intrusive
                alwaysConsumeMouseWheel: false,
              },
            }}
          />
        </div>
      )}
    </div>
  );
}
