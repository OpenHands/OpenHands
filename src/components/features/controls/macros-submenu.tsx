import { useTranslation } from "react-i18next";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "../context-menu/context-menu-list-item";
import { ToolsContextMenuIconText } from "./tools-context-menu-icon-text";

import TachometerFastIcon from "#/icons/tachometer-fast.svg?react";
import PrStatusIcon from "#/icons/pr-status.svg?react";
import DocumentIcon from "#/icons/document.svg?react";
import WaterIcon from "#/icons/u-water.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { useConversationStore } from "#/stores/conversation-store";
import { REPO_SUGGESTIONS } from "#/utils/suggestions/repo-suggestions";

const submenuListItemClassName = "!w-auto whitespace-nowrap";

interface MacrosSubmenuProps {
  onClose: () => void;
}

export function MacrosSubmenu({ onClose }: MacrosSubmenuProps) {
  const { t } = useTranslation("openhands");
  const { setMessageToSend } = useConversationStore();

  const onIncreaseTestCoverage = () => {
    setMessageToSend(REPO_SUGGESTIONS.INCREASE_TEST_COVERAGE);
    onClose();
  };
  const onFixReadme = () => {
    setMessageToSend(REPO_SUGGESTIONS.FIX_README);
    onClose();
  };
  const onAutoMergePRs = () => {
    setMessageToSend(REPO_SUGGESTIONS.AUTO_MERGE_PRS);
    onClose();
  };
  const onCleanDependencies = () => {
    setMessageToSend(REPO_SUGGESTIONS.CLEAN_DEPENDENCIES);
    onClose();
  };

  return (
    <ContextMenu testId="macros-submenu" className="overflow-visible">
      <ContextMenuListItem
        testId="increase-test-coverage-button"
        onClick={onIncreaseTestCoverage}
        className={submenuListItemClassName}
      >
        <ToolsContextMenuIconText
          icon={<TachometerFastIcon width={16} height={16} />}
          text={t(I18nKey.INCREASE_TEST_COVERAGE)}
        />
      </ContextMenuListItem>

      <ContextMenuListItem
        testId="fix-readme-button"
        onClick={onFixReadme}
        className={submenuListItemClassName}
      >
        <ToolsContextMenuIconText
          icon={<DocumentIcon width={16} height={16} />}
          text={t(I18nKey.FIX_README)}
        />
      </ContextMenuListItem>

      <ContextMenuListItem
        testId="auto-merge-prs-button"
        onClick={onAutoMergePRs}
        className={submenuListItemClassName}
      >
        <ToolsContextMenuIconText
          icon={<PrStatusIcon width={16} height={16} />}
          text={t(I18nKey.AUTO_MERGE_PRS)}
        />
      </ContextMenuListItem>

      <ContextMenuListItem
        testId="clean-dependencies-button"
        onClick={onCleanDependencies}
        className={submenuListItemClassName}
      >
        <ToolsContextMenuIconText
          icon={<WaterIcon width={16} height={16} />}
          text={t(I18nKey.CLEAN_DEPENDENCIES)}
        />
      </ContextMenuListItem>
    </ContextMenu>
  );
}
