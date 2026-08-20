import React from "react";
import { useTranslation } from "react-i18next";
import { GitCommit } from "#/api/open-hands.types";
import { I18nKey } from "#/i18n/declaration";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useCommitChanges } from "#/hooks/query/use-commit-changes";
import { AccordionPanel } from "./accordion-panel";
import { DiffChangeList } from "./diff-change-list";
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
 * expanding into the files that commit changed as a single-open accordion.
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
        className="w-full flex h-10 items-center gap-2 px-3 border-b border-[var(--oh-border)] text-sm text-content text-left hover:cursor-pointer"
      >
        <code className="w-[7ch] flex-shrink-0 text-center font-mono text-xs text-[var(--oh-muted)]">
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
        {isExpanded ? (
          <ChevronDown
            className="w-4 h-4 shrink-0 text-[var(--oh-muted)]"
            aria-hidden
          />
        ) : (
          <ChevronRight
            className="w-4 h-4 shrink-0 text-[var(--oh-muted)]"
            aria-hidden
          />
        )}
      </button>

      <AccordionPanel
        open={isExpanded}
        testId="commit-row-content"
        className="w-full flex flex-col pl-6"
      >
        {isLoading && (
          <div className="p-3">
            <LoadingSpinner className="w-4 h-4" />
          </div>
        )}
        {isSuccess && changes && changes.length > 0 && (
          <DiffChangeList changes={changes} commit={commit.sha} />
        )}
      </AccordionPanel>
    </div>
  );
}
