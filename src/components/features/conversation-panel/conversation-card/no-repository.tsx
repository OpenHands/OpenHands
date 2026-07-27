import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import RepoForkedIcon from "#/icons/repo-forked.svg?react";
import { getPathBasename } from "#/utils/path-utils";

interface NoRepositoryProps {
  workspaceWorkingDir?: string | null;
}

export function NoRepository({ workspaceWorkingDir }: NoRepositoryProps) {
  const { t } = useTranslation("openhands");

  const folderName = workspaceWorkingDir
    ? getPathBasename(workspaceWorkingDir).trim()
    : "";

  if (folderName) {
    return (
      <span
        className="truncate min-w-0 text-xs text-[var(--oh-muted)] flex-1"
        title={workspaceWorkingDir ?? undefined}
      >
        {folderName}
      </span>
    );
  }

  return (
    <div className="flex items-center gap-1 text-xs text-[var(--oh-muted)] flex-1 min-w-0 overflow-hidden">
      <RepoForkedIcon
        width={14}
        height={14}
        className="text-[var(--oh-muted)]"
      />
      <span className="truncate min-w-0">
        {t(I18nKey.COMMON$NO_REPOSITORY)}
      </span>
    </div>
  );
}
