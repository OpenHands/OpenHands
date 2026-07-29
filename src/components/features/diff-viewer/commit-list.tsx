import React from "react";
import { useTranslation } from "react-i18next";
import { GitCommit } from "#/api/open-hands.types";
import { I18nKey } from "#/i18n/declaration";
import { CommitRow } from "./commit-row";

export interface CommitListProps {
  commits: GitCommit[];
  /** True when the server capped the page — surfaces the cap notice. */
  hasMore: boolean;
}

/**
 * The workspace's recent commit history (newest first). Single-open
 * accordion: expanding a commit collapses the previously expanded one.
 * Empty/loading/waiting states are the host view's job
 * (see routes/commits-tab.tsx).
 */
export function CommitList({ commits, hasMore }: CommitListProps) {
  const { t } = useTranslation("openhands");
  const [expandedSha, setExpandedSha] = React.useState<string | null>(null);

  // Author is noise when every commit has the same one (the usual
  // single-agent conversation); show it only when authors differ.
  const showAuthor = new Set(commits.map((commit) => commit.author)).size > 1;

  return (
    <section data-testid="commit-list" className="w-full flex flex-col">
      {commits.map((commit) => (
        <CommitRow
          key={commit.sha}
          commit={commit}
          showAuthor={showAuthor}
          isExpanded={expandedSha === commit.sha}
          onToggle={() =>
            setExpandedSha((prev) => (prev === commit.sha ? null : commit.sha))
          }
        />
      ))}

      {hasMore && (
        <div
          data-testid="commit-list-cap-notice"
          className="px-3 py-2.5 text-xs text-[var(--oh-muted)]"
        >
          {t(I18nKey.DIFF_VIEWER$COMMITS_CAP, { count: commits.length })}
        </div>
      )}
    </section>
  );
}
