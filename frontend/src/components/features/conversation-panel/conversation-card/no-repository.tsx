import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import RepoForkedIcon from "#/icons/repo-forked.svg?react";

export function NoRepository() {
  const { t } = useTranslation();

  return (
    <div className="flex items-center gap-1 text-xs text-[#A3A3A3]">
      <RepoForkedIcon width={14} height={14} className="text-[#A3A3A3]" />
      <span>{t(I18nKey.COMMON$NO_REPOSITORY)}</span>
    </div>
  );
}
