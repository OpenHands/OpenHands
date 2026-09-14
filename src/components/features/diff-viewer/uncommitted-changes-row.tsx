import { useTranslation } from "react-i18next";
import { ChevronDown, ChevronRight } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { AccordionPanel } from "./accordion-panel";
import { DiffChangeList, type DiffChangeListItem } from "./diff-change-list";

const EMPTY_COMMIT_SHA_PLACEHOLDER = "-";
/** Base key for i18next pluralization (`_one` / `_other` suffixes). */
const UNCOMMITTED_FILE_COUNT_I18N_KEY = "DIFF_VIEWER$UNCOMMITTED_FILE_COUNT";

export interface UncommittedChangesRowProps {
  changes: DiffChangeListItem[];
  isExpanded: boolean;
  onToggle: () => void;
}

/**
 * Top accordion row in the Commits pane for working-tree changes that
 * are not yet associated with a commit. Same single-open accordion
 * contract as CommitRow.
 */
export function UncommittedChangesRow({
  changes,
  isExpanded,
  onToggle,
}: UncommittedChangesRowProps) {
  const { t } = useTranslation("openhands");

  return (
    <div data-testid="uncommitted-changes-row" className="w-full flex flex-col">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={isExpanded}
        data-testid="uncommitted-changes-row-toggle"
        className="w-full flex h-10 items-center gap-2 px-3 border-b border-[var(--oh-border)] text-sm text-content text-left hover:cursor-pointer"
      >
        <code className="w-[7ch] flex-shrink-0 text-center font-mono text-xs text-[var(--oh-muted)]">
          {EMPTY_COMMIT_SHA_PLACEHOLDER}
        </code>
        <strong className="flex-1 truncate font-medium">
          {t(I18nKey.DIFF_VIEWER$UNCOMMITTED)}
        </strong>
        {changes.length > 0 ? (
          <span
            data-testid="uncommitted-changes-count"
            className="text-xs text-[var(--oh-muted)] tabular-nums flex-shrink-0"
          >
            {t(UNCOMMITTED_FILE_COUNT_I18N_KEY, { count: changes.length })}
          </span>
        ) : null}
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
        testId="uncommitted-changes-row-content"
        className="w-full flex flex-col pl-6"
      >
        {changes.length > 0 ? <DiffChangeList changes={changes} /> : null}
      </AccordionPanel>
    </div>
  );
}
