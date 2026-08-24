import { useTranslation } from "react-i18next";
import { Folder } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import RepoForkedIcon from "#/icons/repo-forked.svg?react";
import { getPathBasename } from "#/utils/path-utils";
import {
  CONVERSATION_CARD_META_CHIP_CLASSNAME,
  CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME,
  CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME,
} from "./conversation-card-meta-chip";

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
        data-testid="conversation-card-workspace-folder"
        className={CONVERSATION_CARD_META_CHIP_CLASSNAME}
        title={workspaceWorkingDir ?? undefined}
      >
        <span
          className={CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME}
          aria-hidden
        >
          <Folder className={CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME} />
        </span>
        <span className="truncate leading-4">{folderName}</span>
      </span>
    );
  }

  return (
    <span
      data-testid="conversation-card-no-repository"
      className={CONVERSATION_CARD_META_CHIP_CLASSNAME}
    >
      <span
        className={CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME}
        aria-hidden
      >
        <RepoForkedIcon
          className={CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME}
        />
      </span>
      <span className="truncate leading-4">
        {t(I18nKey.COMMON$NO_REPOSITORY)}
      </span>
    </span>
  );
}
