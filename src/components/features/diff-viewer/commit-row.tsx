import React from "react";
import { useTranslation } from "react-i18next";
import { GitCommit } from "#/api/open-hands.types";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { formatTimeDelta } from "#/utils/format-time-delta";
import ChevronUp from "#/icons/chveron-up.svg?react";
import { useCommitChanges } from "#/hooks/query/use-commit-changes";
import { FileDiffViewer } from "./file-diff-viewer";
import { LoadingSpinner } from "./loading-spinner";

export interface CommitRowProps {
  commit: GitCommit;
  /** Only shown when the listed commits have more than one author. */
  showAuthor: boolean;
  isExpanded: boolean;
  onToggle: () => void;
}

/**
 * One collapsible commit: header row (short SHA, subject, relative time),
 * expanding into the files that commit changed, each rendered with the
 * regular FileDiffViewer in per-commit mode.
 */
export function CommitRow({
  commit,
  showAuthor,
  isExpanded,
  onToggle,
}: CommitRowProps) {
  const { t } = useTranslation("openhands");

  // Only fetch a commit's file list once the row is expanded.
  const {
    data: changes,
    isLoading,
    isSuccess,
  } = useCommitChanges(commit.sha, { enabled: isExpanded });

  return (
    <div data-testid="commit-row" className="w-full flex flex-col">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={isExpanded}
        data-testid="commit-row-toggle"
        className="w-full flex items-center gap-2 px-3 py-2.5 border-b border-[var(--oh-border)] text-sm text-content text-left hover:cursor-pointer"
      >
        <code className="font-mono text-xs text-[var(--oh-muted)] flex-shrink-0">
          {commit.shortSha}
        </code>
        <strong className="flex-1 truncate font-medium">
          {commit.subject}
        </strong>
        {showAuthor && (
          <span className="text-xs text-[var(--oh-muted)] truncate max-w-32 flex-shrink-0">
            {commit.author}
          </span>
        )}
        <span className="text-xs text-[var(--oh-muted)] flex-shrink-0">
          {`${formatTimeDelta(commit.timestamp)} ${t(I18nKey.CONVERSATION$AGO)}`}
        </span>
        <ChevronUp
          className={cn(
            "w-4 h-4 transition-transform flex-shrink-0",
            !isExpanded && "transform rotate-180",
          )}
        />
      </button>

      {isExpanded && (
        <div
          data-testid="commit-row-content"
          className="w-full flex flex-col pl-6"
        >
          {isLoading && (
            <div className="p-3">
              <LoadingSpinner className="w-4 h-4" />
            </div>
          )}
          {isSuccess &&
            changes?.map((change) => (
              <FileDiffViewer
                key={change.path}
                path={change.path}
                type={change.status}
                commit={commit.sha}
              />
            ))}
        </div>
      )}
    </div>
  );
}
