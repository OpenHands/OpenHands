import { ListTree } from "lucide-react";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import CloseIcon from "#/icons/u-close.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface FileQuickRowProps {
  /** Open file tabs only (paths the user or agent has opened). */
  openPaths: string[];
  selectedPath: string | null;
  onSelectFile: (path: string) => void;
  onCloseFile: (path: string) => void;
  /** Whether the left-hand file tree is currently visible. */
  isTreeVisible: boolean;
  /** Toggle the visibility of the left-hand file tree. */
  onToggleTree: () => void;
  /** Trailing actions (e.g. refresh), pinned to the right. */
  actions?: ReactNode;
}

const HIDDEN_SCROLLBAR_CLASSNAME =
  "[scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden";

/**
 * Horizontal strip of open-file tabs. A path only appears after the user or
 * agent opens it. Overflow scrolls horizontally without a visible scrollbar.
 */
export function FileQuickRow({
  openPaths,
  selectedPath,
  onSelectFile,
  onCloseFile,
  isTreeVisible,
  onToggleTree,
  actions,
}: FileQuickRowProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      className="flex h-[34px] shrink-0 items-stretch gap-1.5 overflow-y-hidden border-b border-[var(--oh-border)] px-2"
      data-testid="file-quick-row"
    >
      <button
        type="button"
        onClick={onToggleTree}
        data-testid="file-quick-row-tree-toggle"
        aria-pressed={isTreeVisible}
        aria-label={t(
          isTreeVisible
            ? I18nKey.FILES$HIDE_FILE_TREE
            : I18nKey.FILES$SHOW_FILE_TREE,
        )}
        title={t(
          isTreeVisible
            ? I18nKey.FILES$HIDE_FILE_TREE
            : I18nKey.FILES$SHOW_FILE_TREE,
        )}
        className={cn(
          "shrink-0 self-center inline-flex items-center justify-center w-6 h-6 rounded-md cursor-pointer",
          "text-[var(--oh-text-tertiary)] hover:bg-tertiary",
          isTreeVisible && "bg-[var(--oh-surface-raised)]",
        )}
      >
        <ListTree className="w-3 h-3" aria-hidden strokeWidth={2} />
      </button>

      {openPaths.length > 0 ? (
        <div
          role="tablist"
          aria-label={t(I18nKey.COMMON$FILES)}
          className={cn(
            "flex min-h-0 min-w-0 flex-1 flex-nowrap items-stretch overflow-x-auto overflow-y-hidden overscroll-y-none",
            HIDDEN_SCROLLBAR_CLASSNAME,
          )}
        >
          {openPaths.map((path, index) => {
            const isSelected = selectedPath === path;
            const fileName = path.split("/").pop() || path;
            return (
              <div
                key={path}
                className={cn(
                  "group/file-tab relative flex shrink-0 items-stretch",
                  // Vertical edges on every tab (left on the first, right on all).
                  "border-r border-r-[var(--oh-border)]",
                  index === 0 && "border-l border-l-[var(--oh-border)]",
                  "border-b-2 -mb-px transition-colors",
                  isSelected
                    ? "border-b-white text-white"
                    : "border-b-transparent text-[var(--oh-muted)] hover:text-white hover:border-b-white/25",
                )}
              >
                <button
                  type="button"
                  role="tab"
                  aria-selected={isSelected}
                  onClick={() => onSelectFile(path)}
                  title={path}
                  data-testid={`file-quick-row-item-${path}`}
                  className="flex min-w-0 max-w-[160px] items-center pl-2.5 pr-1 text-xs cursor-pointer text-inherit"
                >
                  <span className="truncate">{fileName}</span>
                </button>
                <button
                  type="button"
                  data-testid={`file-quick-row-close-${path}`}
                  aria-label={t(I18nKey.FILES$CLOSE_TAB, { path })}
                  title={t(I18nKey.FILES$CLOSE_TAB, { path })}
                  className={cn(
                    "inline-flex items-center justify-center size-5 self-center mr-1 rounded-sm shrink-0 cursor-pointer",
                    "text-inherit hover:bg-white/10",
                    // Always visible on small / touch-first viewports; hover to
                    // reveal on fine pointers (desktop).
                    "opacity-100 transition-opacity",
                    "md:opacity-0 md:group-hover/file-tab:opacity-100 md:group-focus-within/file-tab:opacity-100",
                  )}
                  onClick={() => onCloseFile(path)}
                >
                  <CloseIcon width={10} height={10} aria-hidden />
                </button>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="flex-1 min-w-0" />
      )}

      {actions ? (
        <div className="ml-auto shrink-0 self-center flex items-center gap-1">
          {actions}
        </div>
      ) : null}
    </div>
  );
}
