import { DiffEditor, Editor, Monaco } from "@monaco-editor/react";
import React from "react";
import { editor as editor_t } from "monaco-editor";
import {
  LuFileDiff,
  LuFileMinus,
  LuFilePlus,
  LuHistory,
  LuGitCompareArrows,
  LuFileCheck,
} from "react-icons/lu";
import { IconType } from "react-icons/lib";
import { GitChangeStatus } from "#/api/open-hands.types";
import { getLanguageFromPath } from "#/utils/get-language-from-path";
import { cn } from "#/utils/utils";
import ChevronUp from "#/icons/chveron-up.svg?react";
import { useUnifiedGitDiff } from "#/hooks/query/use-unified-git-diff";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";
import { Typography } from "#/ui/typography";
import { LoadingSpinner } from "./loading-spinner";
import { EditorContainer } from "./editor-container";

type ViewMode = "diff" | "old" | "new";

const VIEW_MODES: { mode: ViewMode; icon: IconType }[] = [
  { mode: "old", icon: LuHistory },
  { mode: "diff", icon: LuGitCompareArrows },
  { mode: "new", icon: LuFileCheck },
];

const SHARED_EDITOR_OPTIONS: editor_t.IEditorOptions = {
  renderValidationDecorations: "off",
  readOnly: true,
  scrollBeyondLastLine: false,
  minimap: { enabled: false },
  automaticLayout: true,
  scrollbar: { alwaysConsumeMouseWheel: false },
};

const STATUS_MAP: Record<GitChangeStatus, string | IconType> = {
  A: LuFilePlus,
  D: LuFileMinus,
  M: LuFileDiff,
  R: "Renamed",
  U: "Untracked",
};

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
      "diffEditor.insertedTextBackground": "#014b01AA",
      "diffEditor.removedTextBackground": "#750000AA",
      "diffEditor.insertedLineBackground": "#003f00AA",
      "diffEditor.removedLineBackground": "#5a0000AA",
      "diffEditor.border": "#444444",
      "editorUnnecessaryCode.border": "#00000000",
      "editorUnnecessaryCode.opacity": "#00000077",
    },
  });
};

export interface FileDiffViewerProps {
  path: string;
  type: GitChangeStatus;
}

export function FileDiffViewer({ path, type }: FileDiffViewerProps) {
  const [isCollapsed, setIsCollapsed] = React.useState(true);
  const [editorHeight, setEditorHeight] = React.useState(400);
  const [viewMode, setViewMode] = React.useState<ViewMode>("diff");
  const diffEditorRef = React.useRef<editor_t.IStandaloneDiffEditor>(null);
  const singleEditorRef = React.useRef<editor_t.IStandaloneCodeEditor>(null);

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

  const updateEditorHeight = React.useCallback(() => {
    if (!diffEditorRef.current) return;
    const originalEditor = diffEditorRef.current.getOriginalEditor();
    const modifiedEditor = diffEditorRef.current.getModifiedEditor();
    if (originalEditor && modifiedEditor) {
      setEditorHeight(
        Math.max(
          originalEditor.getContentHeight(),
          modifiedEditor.getContentHeight(),
        ) + 20,
      );
    }
  }, []);

  const updateSingleEditorHeight = React.useCallback(() => {
    if (singleEditorRef.current) {
      setEditorHeight(singleEditorRef.current.getContentHeight() + 20);
    }
  }, []);

  const handleDiffEditorMount = (editor: editor_t.IStandaloneDiffEditor) => {
    diffEditorRef.current = editor;
    updateEditorHeight();
    editor.getOriginalEditor().onDidContentSizeChange(updateEditorHeight);
    editor.getModifiedEditor().onDidContentSizeChange(updateEditorHeight);
  };

  const handleSingleEditorMount = (editor: editor_t.IStandaloneCodeEditor) => {
    singleEditorRef.current = editor;
    updateSingleEditorHeight();
    editor.onDidContentSizeChange(updateSingleEditorHeight);
  };

  const status = (type === "U" ? STATUS_MAP.A : STATUS_MAP[type]) || "?";
  const statusIcon =
    typeof status === "string" ? (
      <Typography.Text>{status}</Typography.Text>
    ) : (
      React.createElement(status, { className: "w-5 h-5" })
    );

  const isFetchingData = isLoading || isRefetching;
  const language = getLanguageFromPath(filePath);
  const isMarkdownFile = language === "markdown";
  const singleViewContent =
    viewMode === "old" ? (diff?.original ?? "") : (diff?.modified ?? "");

  const renderContent = () => {
    if (viewMode === "diff") {
      return (
        <EditorContainer height={editorHeight}>
          <DiffEditor
            data-testid="file-diff-viewer"
            className="w-full h-full"
            language={language}
            original={isAdded ? "" : (diff?.original ?? "")}
            modified={isDeleted ? "" : (diff?.modified ?? "")}
            theme="custom-diff-theme"
            onMount={handleDiffEditorMount}
            beforeMount={beforeMount}
            options={{
              ...SHARED_EDITOR_OPTIONS,
              renderSideBySide: !isAdded && !isDeleted,
              hideUnchangedRegions: { enabled: true },
            }}
          />
        </EditorContainer>
      );
    }

    if (isMarkdownFile) {
      return (
        <div
          className="w-full border border-neutral-600 overflow-auto p-4 bg-neutral-900 prose prose-invert max-w-none"
          data-testid="markdown-preview"
        >
          <MarkdownRenderer
            content={singleViewContent}
            includeStandard
            includeHeadings
          />
        </div>
      );
    }

    return (
      <EditorContainer height={editorHeight}>
        <Editor
          data-testid="file-single-viewer"
          className="w-full h-full"
          language={language}
          value={singleViewContent}
          theme="custom-diff-theme"
          beforeMount={beforeMount}
          onMount={handleSingleEditorMount}
          options={SHARED_EDITOR_OPTIONS}
        />
      </EditorContainer>
    );
  };

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
          {isFetchingData ? <LoadingSpinner className="w-5 h-5" /> : statusIcon}
          <strong className="w-full truncate">{filePath}</strong>
          {!isCollapsed && (
            <span
              className="flex items-center gap-0.5 shrink-0"
              onClick={(e) => e.stopPropagation()}
            >
              {VIEW_MODES.map(({ mode, icon: Icon }) => (
                <button
                  key={mode}
                  data-testid={`view-mode-${mode}`}
                  type="button"
                  onClick={() => setViewMode(mode)}
                  className={cn(
                    "p-1 rounded transition-colors cursor-pointer",
                    viewMode === mode
                      ? "bg-neutral-600 text-white"
                      : "text-neutral-400 hover:bg-neutral-700 hover:text-neutral-200",
                  )}
                >
                  <Icon className="w-4 h-4" />
                </button>
              ))}
            </span>
          )}
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

      {isSuccess && !isCollapsed && renderContent()}
    </div>
  );
}
