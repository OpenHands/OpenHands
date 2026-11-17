import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";
import { useCloudOrgMember } from "#/hooks/query/use-cloud-org-member";
import type { WorklogEntry } from "#/api/cloud/worklog-service.api";
import { extensionModuleCardSurfaceClassName } from "#/utils/extension-module-card-classes";
import { formatRelativeTime } from "#/utils/format-relative-time";

function UserAuthorName({ userId }: { userId: string | null }) {
  const { t } = useTranslation("openhands");
  const { data: member } = useCloudOrgMember(userId ?? undefined);
  return <>{member?.email ?? t(I18nKey.WORKLOGS$AUTHOR_USER)}</>;
}

function AuthorName({ author }: { author: WorklogEntry["author"] }) {
  const { t } = useTranslation("openhands");
  if (author.kind === "automation") {
    return <>{author.name ?? t(I18nKey.WORKLOGS$AUTHOR_AUTOMATION)}</>;
  }
  return <UserAuthorName userId={author.id} />;
}

export function WorklogEntryCard({ entry }: { entry: WorklogEntry }) {
  const { t, i18n } = useTranslation("openhands");

  return (
    <li
      data-testid="worklog-entry"
      className={`${extensionModuleCardSurfaceClassName} flex flex-col gap-3 p-4`}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 text-xs text-muted">
        <span data-testid="worklog-entry-author" className="text-contrast">
          <AuthorName author={entry.author} />
        </span>
        <time
          dateTime={entry.created_at}
          title={new Date(entry.created_at).toLocaleString(i18n.language)}
        >
          {formatRelativeTime(entry.created_at, i18n.language, t)}
        </time>
      </div>
      {entry.content_type === "json" ? (
        <>
          {entry.title && (
            <h3 className="text-sm font-medium text-contrast">{entry.title}</h3>
          )}
          <pre
            data-testid="worklog-entry-json"
            className="overflow-x-auto whitespace-pre-wrap break-words rounded-lg bg-canvas-base p-3 font-mono text-xs text-contrast"
          >
            {entry.body}
          </pre>
        </>
      ) : (
        <div className="min-w-0 text-sm text-contrast">
          <MarkdownRenderer includeStandard includeHeadings>
            {entry.body}
          </MarkdownRenderer>
        </div>
      )}
    </li>
  );
}
