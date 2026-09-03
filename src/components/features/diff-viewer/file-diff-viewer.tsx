import { DiffEditor, Editor, Monaco } from "@monaco-editor/react";
import React from "react";
import { editor as editor_t } from "monaco-editor";
import { useTranslation } from "react-i18next";
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
import { I18nKey } from "#/i18n/declaration";
import { getLanguageFromPath } from "#/utils/get-language-from-path";
import { cn } from "#/utils/utils";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useUnifiedGitDiff } from "#/hooks/query/use-unified-git-diff";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";
import { Typography } from "#/ui/typography";
import { LoadingSpinner } from "./loading-spinner";
import { EditorContainer } from "./editor-container";
import { AccordionPanel } from "./accordion-panel";

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

/** Cap opened diff/editor panes so a large file doesn't dominate the drawer. */
export const MAX_DIFF_EDITOR_HEIGHT_PX = 600;

function clampDiffEditorHeight(height: number): number {
  return Math.min(height, MAX_DIFF_EDITOR_HEIGHT_PX);
}

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
      "diffEditor.border": "var(--oh-border-subtle)",
      "editorUnnecessaryCode.border": "#00000000",
      "editorUnnecessaryCode.opacity": "rgba(0, 0, 0, 0.467)",
    },
  });
};

export interface FileDiffViewerProps {
  path: string;
  type: GitChangeStatus;
  /**
   * When set, show the file's diff as changed by this commit instead of
   * the working-tree-vs-base diff. Deleted files render their content in
   * commit mode (both sides come from git objects).
   */
  commit?: string;
  /**
   * Controlled accordion open state. When omitted, the row manages its own
   * expand/collapse (used by unit tests and standalone embeds).
   */
  isExpanded?: boolean;
  /** Required with `isExpanded` for controlled accordion lists. */
  onToggle?: () => void;
}

export function FileDiffViewer({
  path,
  type,
  commit,
  isExpanded: controlledExpanded,
  onToggle,
}: FileDiffViewerProps) {
  const { t } = useTranslation("openhands");
  const [uncontrolledExpanded, setUncontrolledExpanded] = React.useState(false);
  const isControlled = controlledExpanded !== undefined;
  const isExpanded = isControlled ? controlledExpanded : uncontrolledExpanded;
  const isCollapsed = !isExpanded;
  const handleToggle = () => {
    if (isControlled) {
      onToggle?.();
      return;
    }
    setUncontrolledExpanded((prev) => !prev);
  };
  const [editorHeight, setEditorHeight] = React.useState(0);
  const [hasMeasuredEditorHeight, setHasMeasuredEditorHeight] =
    React.useState(false);
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
    commit,
  });

  const updateEditorHeight = React.useCallback(() => {
    if (!diffEditorRef.current) return;
    const originalEditor = diffEditorRef.current.getOriginalEditor();
    const modifiedEditor = diffEditorRef.current.getModifiedEditor();
    if (originalEditor && modifiedEditor) {
      setEditorHeight(
        clampDiffEditorHeight(
          Math.max(
            originalEditor.getContentHeight(),
            modifiedEditor.getContentHeight(),
          ) + 20,
        ),
      );
      setHasMeasuredEditorHeight(true);
    }
  }, []);

  const updateSingleEditorHeight = React.useCallback(() => {
    if (singleEditorRef.current) {
      setEditorHeight(
        clampDiffEditorHeight(singleEditorRef.current.getContentHeight() + 20),
      );
      setHasMeasuredEditorHeight(true);
    }
  }, []);

  // Reset measured height when collapsing or switching view modes so the
  // next paint doesn't flash a stale editor size before Monaco remeasures.
  React.useEffect(() => {
    setHasMeasuredEditorHeight(false);
    setEditorHeight(0);
  }, [isCollapsed, viewMode]);

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
      React.createElement(status, { className: "w-4 h-4 shrink-0" })
    );

  const isFetchingData = isLoading || isRefetching;
  const language = getLanguageFromPath(filePath);
  const isMarkdownFile = language === "markdown";
  const singleViewContent =
    viewMode === "old" ? (diff?.original ?? "") : (diff?.modified ?? "");

  const showViewModeToggle = !isDeleted;
  const viewModeControlsVisible = !isCollapsed && showViewModeToggle;

  const renderEditorShell = (editor: React.ReactNode) => (
    <div
      className={cn(
        "relative w-full",
        // Collapse layout space until Monaco reports a real content height so
        // expand doesn't flash a placeholder editor size.
        !hasMeasuredEditorHeight && "h-0 overflow-hidden",
      )}
    >
      <EditorContainer
        height={
          hasMeasuredEditorHeight ? editorHeight : clampDiffEditorHeight(400)
        }
        className={cn(
          !hasMeasuredEditorHeight && "absolute inset-x-0 top-0 invisible",
        )}
      >
        {editor}
      </EditorContainer>
    </div>
  );

  const renderContent = () => {
    if (viewMode === "diff") {
      return renderEditorShell(
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
        />,
      );
    }

    if (isMarkdownFile) {
      return (
        <div
          className="w-full border-b border-[var(--oh-border)] overflow-auto p-4 bg-base prose prose-invert max-w-none"
          data-testid="markdown-preview"
          style={{ maxHeight: MAX_DIFF_EDITOR_HEIGHT_PX }}
        >
          <MarkdownRenderer
            content={singleViewContent}
            includeStandard
            includeHeadings
          />
        </div>
      );
    }

    return renderEditorShell(
      <Editor
        data-testid="file-single-viewer"
        className="w-full h-full"
        language={language}
        value={singleViewContent}
        theme="custom-diff-theme"
        beforeMount={beforeMount}
        onMount={handleSingleEditorMount}
        options={SHARED_EDITOR_OPTIONS}
      />,
    );
  };

  return (
    <div data-testid="file-diff-viewer-outer" className="w-full flex flex-col">
      <div
        className="flex h-10 items-center px-3 border-b border-[var(--oh-border)] hover:cursor-pointer"
        onClick={handleToggle}
      >
        <span className="text-sm w-full text-content flex items-center gap-2 min-w-0">
          <span className="inline-flex w-4 h-4 shrink-0 items-center justify-center">
            {isFetchingData ? (
              <LoadingSpinner className="w-4 h-4" />
            ) : (
              statusIcon
            )}
          </span>
          <strong className="min-w-0 flex-1 truncate font-medium">
            {filePath}
          </strong>
          {showViewModeToggle && (
            <span
              className={cn(
                "flex items-center gap-0.5 shrink-0",
                !viewModeControlsVisible && "invisible pointer-events-none",
              )}
              onClick={(e) => e.stopPropagation()}
              aria-hidden={!viewModeControlsVisible}
            >
              {VIEW_MODES.map(({ mode, icon: Icon }) => (
                <button
                  key={mode}
                  data-testid={`view-mode-${mode}`}
                  type="button"
                  tabIndex={viewModeControlsVisible ? 0 : -1}
                  aria-pressed={viewMode === mode}
                  onClick={() => setViewMode(mode)}
                  className={cn(
                    "p-1 rounded transition-colors cursor-pointer",
                    viewMode === mode
                      ? "bg-[var(--oh-interactive-hover)] text-white"
                      : "text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-hover)] hover:text-white",
                  )}
                >
                  <Icon className="w-4 h-4" />
                </button>
              ))}
            </span>
          )}
          <button
            data-testid="collapse"
            type="button"
            className="shrink-0 text-[var(--oh-muted)]"
          >
            {isCollapsed ? (
              <ChevronRight className="w-4 h-4" aria-hidden />
            ) : (
              <ChevronDown className="w-4 h-4" aria-hidden />
            )}
          </button>
        </span>
      </div>

      <AccordionPanel open={isExpanded}>
        {isDeleted && !commit ? (
          <div
            data-testid="file-deleted-message"
            className="w-full border-b border-[var(--oh-border)] p-4 bg-base text-[var(--oh-text-dim)] text-sm"
          >
            {t(I18nKey.DIFF_VIEWER$FILE_DELETED)}
          </div>
        ) : (
          isSuccess && renderContent()
        )}
      </AccordionPanel>
    </div>
  );
}
