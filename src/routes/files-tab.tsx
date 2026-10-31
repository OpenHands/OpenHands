import {
  type CSSProperties,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import toast from "react-hot-toast";

import { NoFileSelectedMessage } from "#/components/features/files-tab/no-file-selected-message";
import { I18nKey } from "#/i18n/declaration";
import { useFilesTabStore } from "#/stores/files-tab-store";
import { useWorkspaceFiles } from "#/hooks/query/use-workspace-files";
import { useWorkspaceFileContent } from "#/hooks/query/use-workspace-file-content";
import { useAutoRefreshFilesOnEdit } from "#/hooks/use-auto-refresh-files-on-edit";
import { useDownloadWorkspaceFile } from "#/hooks/use-download-workspace-file";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useResizableDrawerWidth } from "#/hooks/use-resizable-drawer-width";
import {
  getConversationState,
  useConversationLocalStorageState,
} from "#/utils/conversation-local-storage";
import { useWorkspaceMutationCounter } from "#/stores/use-workspace-mutation-counter";
import { FileQuickRow } from "#/components/features/files-tab/file-quick-row";
import { FileTreeView } from "#/components/features/files-tab/file-tree-view";
import { FileContentViewer } from "#/components/features/files-tab/file-content-viewer";
import { SegmentedToggle } from "#/components/features/files-tab/segmented-toggle";
import { WorkspacePath } from "#/components/features/files-tab/workspace-path";
import { WorkspaceFileDiscoverySettings } from "#/components/features/files-tab/workspace-file-discovery-settings";
import type { ViewMode } from "#/components/features/files-tab/view-mode";
import {
  FILES_TAB_TREE_DEFAULT_WIDTH_PX,
  FILES_TAB_TREE_MAX_WIDTH_PX,
  FILES_TAB_TREE_MIN_WIDTH_PX,
  FILES_TAB_TREE_RESIZE_HANDLE_TEST_ID,
  FILES_TAB_TREE_WIDTH_STORAGE_KEY,
} from "#/components/features/files-tab/files-tab-tree.constants";
import { ResizeHandle } from "#/components/ui/resize-handle";
import RefreshIcon from "#/icons/u-refresh.svg?react";
import LinkExternalIcon from "#/icons/link-external.svg?react";
import DownloadIcon from "#/icons/u-download.svg?react";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";
import { getGitPath } from "#/utils/get-git-path";

/**
 * Workspace file browser. Diff/Commits live in the sibling Commits
 * conversation tab — this surface is only the file tree / content viewer.
 */
function FilesTab() {
  const { t } = useTranslation("openhands");

  // Keep the list / content caches fresh as the agent writes files.
  useAutoRefreshFilesOnEdit();

  const { data: activeConversation } = useActiveConversation();
  const workspacePath = activeConversation?.workspace?.working_dir;

  const { conversationId } = useOptionalConversationId();
  const {
    state: persistedState,
    setFilesTabContentViewMode,
    setFilesTabTreeVisible,
  } = useConversationLocalStorageState(conversationId ?? "");
  const contentViewMode = persistedState.filesTabContentViewMode;
  const isTreeVisible = persistedState.filesTabTreeVisible ?? true;
  const toggleTreeVisible = useCallback(() => {
    setFilesTabTreeVisible?.(!isTreeVisible);
  }, [isTreeVisible, setFilesTabTreeVisible]);

  const treeLayoutRef = useRef<HTMLDivElement>(null);
  const {
    drawerWidth: treeWidth,
    isDragging: isTreeResizing,
    handleMouseDown: handleTreeResizeMouseDown,
  } = useResizableDrawerWidth({
    containerRef: treeLayoutRef,
    defaultWidth: FILES_TAB_TREE_DEFAULT_WIDTH_PX,
    minWidth: FILES_TAB_TREE_MIN_WIDTH_PX,
    maxWidth: FILES_TAB_TREE_MAX_WIDTH_PX,
    storageKey: FILES_TAB_TREE_WIDTH_STORAGE_KEY,
    enabled: isTreeVisible,
    edge: "left",
  });

  const filesQuery = useWorkspaceFiles();
  const paths = useMemo(() => filesQuery.data ?? [], [filesQuery.data]);

  const storedSelectedPath = useFilesTabStore((s) => s.selectedPath);
  const selectedConversationId = useFilesTabStore(
    (s) => s.selectedConversationId,
  );
  const openPaths = useFilesTabStore((s) => s.openPaths);
  const setSelectedPath = useFilesTabStore((s) => s.setSelectedPath);
  const closeOpenPath = useFilesTabStore((s) => s.closeOpenPath);
  const hydrateForConversation = useFilesTabStore(
    (s) => s.hydrateForConversation,
  );

  // A selection is scoped to the conversation it was made in. Ignore a path
  // that belongs to a different conversation so we never try to open a file
  // that only exists in the previous conversation's workspace (issue #1350).
  const selectedPath =
    selectedConversationId === conversationId ? storedSelectedPath : null;
  const conversationOpenPaths =
    selectedConversationId === conversationId ? openPaths : [];

  // Tag every selection with the active conversation so it can't leak into
  // the next one. Opening a path also appends it to the tab strip.
  const handleSelectFile = useCallback(
    (path: string) => setSelectedPath(path, conversationId),
    [conversationId, setSelectedPath],
  );

  // Pre-fetch the selected file's content here too so the toolbar's
  // "open in new window" action can build an authenticated object URL.
  // react-query dedupes against `FileContentViewer`'s identical call, so this
  // costs nothing extra.
  const selectedFileContent = useWorkspaceFileContent(selectedPath);
  const mutationCounter = useWorkspaceMutationCounter((state) => state.count);
  const bumpWorkspaceMutationCounter = useWorkspaceMutationCounter(
    (state) => state.bump,
  );
  const selectedFile = selectedFileContent.data;
  const isDataUrl = selectedFile?.staticUrl.startsWith("data:") ?? false;
  const gitPath = getGitPath(
    activeConversation?.selected_repository,
    workspacePath,
  );
  const workspaceRoot = gitPath.startsWith("/") ? gitPath : `/${gitPath}`;
  const absoluteFilePath = selectedPath
    ? `${workspaceRoot}/${selectedPath}`
    : null;
  const fileLoadError = t(I18nKey.FILES$LOAD_ERROR);
  const [selectedFileBlobUrl, setSelectedFileBlobUrl] = useState<string | null>(
    null,
  );

  // A local workspace URL relies on an HttpOnly session cookie. Opening that
  // URL directly in a new top-level tab loses a Partitioned cookie's top-level
  // site key when Canvas and the runtime are on different sites. Fetch through
  // the typed, authenticated runtime client first and open a same-origin blob
  // instead. Text (including HTML/SVG) stays text/plain so it cannot execute in
  // the Canvas origin. Cloud data URLs keep their existing decoded-text path.
  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;
    setSelectedFileBlobUrl(null);

    const prepareObjectUrl = async () => {
      if (!selectedFile || !absoluteFilePath) return;

      try {
        let blob: Blob;
        if (selectedFile.text !== null) {
          blob = new Blob([selectedFile.text], {
            type: "text/plain;charset=utf-8",
          });
        } else {
          // Cloud binary files intentionally keep the existing no-link
          // behavior; their data URL has no decoded body to reuse.
          if (isDataUrl) return;
          const bytes = await AgentServerRuntimeService.downloadFile(
            activeConversation?.conversation_url,
            activeConversation?.session_api_key,
            absoluteFilePath,
          );
          blob = new Blob([bytes], { type: selectedFile.mimeType });
        }

        objectUrl = URL.createObjectURL(blob);
        if (cancelled) {
          URL.revokeObjectURL(objectUrl);
          objectUrl = null;
          return;
        }
        setSelectedFileBlobUrl(objectUrl);
      } catch {
        if (!cancelled) {
          toast.error(fileLoadError);
        }
      }
    };
    void prepareObjectUrl();
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [
    absoluteFilePath,
    activeConversation?.conversation_url,
    activeConversation?.session_api_key,
    fileLoadError,
    isDataUrl,
    mutationCounter,
    selectedFile?.mimeType,
    selectedFile?.path,
    selectedFile?.text,
  ]);
  const download = useDownloadWorkspaceFile();
  // Cloud backends hand the file back as UTF-8 text, so a Cloud file without
  // decoded text (image, PDF, binary) can't be saved faithfully until
  // OHE-3631 lands: offer no download for it, like the new-window link.
  const downloadableFile =
    selectedFile && (!isDataUrl || selectedFile.text !== null)
      ? selectedFile
      : null;

  // Restore open tabs / selection from conversation-scoped localStorage on
  // mount and when switching conversations (survives a full page refresh).
  // Read localStorage directly — the React hook mirror can briefly hold the
  // previous conversation's state until its sync effect runs.
  useEffect(() => {
    if (!conversationId) return;
    if (selectedConversationId === conversationId) return;
    const persisted = getConversationState(conversationId);
    hydrateForConversation(
      conversationId,
      persisted.filesTabOpenPaths ?? [],
      persisted.filesTabSelectedPath ?? null,
    );
  }, [conversationId, selectedConversationId, hydrateForConversation]);

  const queryClient = useQueryClient();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const refreshFiles = async () => {
    setIsRefreshing(true);
    // Bumping the counter re-keys every `workspace-file-content` query, so
    // the open file refetches, and gives the rich preview's <iframe>/<img> a
    // new `?v=` URL: the only way to make them skip a cached response.
    bumpWorkspaceMutationCounter();
    try {
      await queryClient.invalidateQueries({ queryKey: ["workspace-files"] });
    } finally {
      setIsRefreshing(false);
    }
  };

  const quickRowActions = (
    <>
      <WorkspaceFileDiscoverySettings workingDir={workspacePath?.trim()} />
      <button
        type="button"
        onClick={refreshFiles}
        disabled={isRefreshing}
        aria-label={t(I18nKey.FILES$REFRESH)}
        title={t(I18nKey.FILES$REFRESH)}
        data-testid="files-tab-refresh"
        className="flex items-center justify-center w-6.5 py-1 rounded-[7px] hover:enabled:bg-interactive-hover cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <RefreshIcon
          width={12.75}
          height={15}
          color="#ffffff"
          className={isRefreshing ? "animate-spin" : ""}
        />
      </button>
    </>
  );

  return (
    <main
      className="h-full w-full flex flex-col items-stretch"
      data-testid="files-tab"
    >
      <WorkspacePath path={workspacePath} />
      {filesQuery.isTruncated && (
        <p
          role="status"
          className="border-b border-border px-3 py-2 text-xs text-muted"
        >
          {t(I18nKey.FILES$DISCOVERY_TRUNCATED, { count: paths.length })}
        </p>
      )}
      {filesQuery.isLoading ? (
        <div className="flex flex-1 items-center justify-center text-sm text-muted">
          {t(I18nKey.FILES$LOADING_FILES)}
        </div>
      ) : (
        <>
          <FileQuickRow
            openPaths={conversationOpenPaths}
            selectedPath={selectedPath}
            onSelectFile={handleSelectFile}
            onCloseFile={closeOpenPath}
            isTreeVisible={isTreeVisible}
            onToggleTree={toggleTreeVisible}
            actions={quickRowActions}
          />
          <div ref={treeLayoutRef} className="flex h-full min-h-0 flex-1">
            {isTreeVisible && (
              <>
                <aside
                  className="w-(--files-tab-tree-width) shrink-0 border-r border-border overflow-y-auto custom-scrollbar-always"
                  data-testid="files-tab-tree"
                  style={
                    {
                      "--files-tab-tree-width": `${treeWidth}px`,
                    } as CSSProperties
                  }
                >
                  <FileTreeView
                    paths={paths}
                    selectedPath={selectedPath}
                    onSelectFile={handleSelectFile}
                  />
                </aside>
                <ResizeHandle
                  testId={FILES_TAB_TREE_RESIZE_HANDLE_TEST_ID}
                  onMouseDown={handleTreeResizeMouseDown}
                  isDragging={isTreeResizing}
                />
              </>
            )}
            <section
              className="flex h-full min-h-0 min-w-0 flex-1 flex-col"
              data-testid="files-tab-content"
            >
              {selectedPath ? (
                <>
                  <div className="flex items-center gap-3 px-3 py-1.5 border-b border-border">
                    <SegmentedToggle<ViewMode>
                      ariaLabel={t(I18nKey.FILES$RICH)}
                      testId="files-tab-content-mode-toggle"
                      value={contentViewMode}
                      options={[
                        { value: "rich", label: t(I18nKey.FILES$RICH) },
                        { value: "plain", label: t(I18nKey.FILES$PLAIN) },
                      ]}
                      onChange={setFilesTabContentViewMode}
                    />
                    <div className="ml-auto flex items-center gap-1">
                      {downloadableFile ? (
                        <button
                          type="button"
                          onClick={() => download.mutate(downloadableFile)}
                          disabled={download.isPending}
                          aria-label={t(I18nKey.FILES$DOWNLOAD)}
                          title={t(I18nKey.FILES$DOWNLOAD)}
                          data-testid="files-tab-download"
                          className="flex items-center justify-center w-6.5 py-1 rounded-[7px] hover:enabled:bg-interactive-hover cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed text-contrast"
                        >
                          <DownloadIcon width={14} height={14} />
                        </button>
                      ) : null}
                      {selectedFileBlobUrl ? (
                        <a
                          href={selectedFileBlobUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          aria-label={t(I18nKey.FILES$OPEN_IN_NEW_WINDOW)}
                          title={t(I18nKey.FILES$OPEN_IN_NEW_WINDOW)}
                          data-testid="files-tab-open-in-new-window"
                          className="flex items-center justify-center w-6.5 py-1 rounded-[7px] hover:bg-interactive-hover cursor-pointer text-contrast"
                        >
                          <LinkExternalIcon width={14} height={14} />
                        </a>
                      ) : null}
                    </div>
                  </div>
                  <FileContentViewer
                    path={selectedPath}
                    viewMode={contentViewMode}
                  />
                </>
              ) : (
                <NoFileSelectedMessage />
              )}
            </section>
          </div>
        </>
      )}
    </main>
  );
}

export default FilesTab;
