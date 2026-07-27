import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { FileTreeView } from "#/components/features/files-tab/file-tree-view";
import { HighlightedSourceView } from "#/components/features/files-tab/highlighted-source-view";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { usePluginFileContent } from "#/hooks/query/use-plugin-file-content";

interface PluginFilesSectionProps {
  /** Plugin directory on the agent-server (`path`/`install_path`). */
  basePath: string;
  /** File paths relative to `basePath`, as reported by the agent-server. */
  files: string[];
}

/**
 * "Files" section of the plugin detail modal: the plugin's file tree with an
 * inline viewer for the selected file. Clicking the selected file again
 * deselects it and closes the viewer.
 */
export function PluginFilesSection({
  basePath,
  files,
}: PluginFilesSectionProps) {
  const { t } = useTranslation("openhands");
  const [selectedPath, setSelectedPath] = React.useState<string | null>(null);

  // Derived rather than synced: a refetch that drops the selected file cannot
  // leave a dangling selection.
  const effectivePath =
    selectedPath && files.includes(selectedPath) ? selectedPath : null;

  const {
    data: content,
    isLoading,
    isError,
  } = usePluginFileContent(effectivePath ? basePath : null, effectivePath);

  return (
    <section
      data-testid="plugin-files-section"
      className="flex min-w-0 flex-col gap-2"
    >
      <h3 className="text-sm font-medium text-white">
        {t(I18nKey.COMMON$FILES)}
      </h3>
      <div className="min-w-0 rounded-lg border border-[var(--oh-border)] bg-[rgba(255,255,255,0.04)]">
        <div className="max-h-48 overflow-y-auto custom-scrollbar">
          <FileTreeView
            paths={files}
            selectedPath={effectivePath}
            onSelectFile={(path) =>
              setSelectedPath((prev) => (prev === path ? null : path))
            }
          />
        </div>
        {effectivePath ? (
          <div
            data-testid="plugin-file-content"
            className="flex h-60 min-w-0 flex-col border-t border-[var(--oh-border)]"
          >
            <p
              className="truncate px-3 py-2 text-xs text-tertiary-alt"
              title={effectivePath}
            >
              {effectivePath}
            </p>
            <div className="min-h-0 flex-1">
              {isLoading ? (
                <div className="flex h-full items-center justify-center">
                  <LoadingSpinner size="small" />
                </div>
              ) : isError ? (
                <p className="px-3 py-2 text-xs text-tertiary-light">
                  {t(I18nKey.FILES$LOAD_ERROR)}
                </p>
              ) : content?.kind === "binary" ? (
                <p className="px-3 py-2 text-xs text-tertiary-light">
                  {t(I18nKey.FILES$BINARY_FALLBACK)}
                </p>
              ) : (
                <HighlightedSourceView
                  path={effectivePath}
                  text={content?.text ?? ""}
                />
              )}
            </div>
          </div>
        ) : null}
      </div>
    </section>
  );
}
