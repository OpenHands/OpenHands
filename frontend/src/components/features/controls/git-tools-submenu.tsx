import { useTranslation } from "react-i18next";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "../context-menu/context-menu-list-item";
import { ToolsContextMenuIconText } from "./tools-context-menu-icon-text";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { Provider } from "#/types/settings";
import {
  getGitPullPrompt,
  getGitPushPrompt,
  getCreatePRPrompt,
  getCreateNewBranchPrompt,
} from "#/utils/utils";
import { useConversationStore } from "#/stores/conversation-store";

import ArrowUpIcon from "#/icons/u-arrow-up.svg?react";
import ArrowDownIcon from "#/icons/u-arrow-down.svg?react";
import PrIcon from "#/icons/u-pr.svg?react";
import CodeBranchIcon from "#/icons/u-code-branch.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { CONTEXT_MENU_ICON_TEXT_CLASSNAME } from "#/utils/constants";

const contextMenuListItemClassName =
  "cursor-pointer p-0 h-auto hover:bg-transparent !w-auto whitespace-nowrap";
interface GitToolsSubmenuProps {
  onClose: () => void;
}

export function GitToolsSubmenu({ onClose }: GitToolsSubmenuProps) {
  const { t } = useTranslation();
  const { setMessageToSend } = useConversationStore();
  const { data: conversation } = useActiveConversation();

  const currentGitProvider = conversation?.git_provider as Provider;

  const onGitPull = () => {
    setMessageToSend(getGitPullPrompt());
    onClose();
  };

  const onGitPush = () => {
    setMessageToSend(getGitPushPrompt(currentGitProvider));
    onClose();
  };

  const onCreatePR = () => {
    setMessageToSend(getCreatePRPrompt(currentGitProvider));
    onClose();
  };

  const onCreateNewBranch = () => {
    setMessageToSend(getCreateNewBranchPrompt());
    onClose();
  };

  return (
    <ContextMenu
      testId="git-tools-submenu"
      className="text-white bg-tertiary rounded-[6px] py-[6px] px-1 flex flex-col gap-2 w-max"
    >
      <ContextMenuListItem
        testId="git-pull-button"
        onClick={onGitPull}
        className={contextMenuListItemClassName}
      >
        <ToolsContextMenuIconText
          icon={<ArrowDownIcon width={16} height={16} />}
          text={t(I18nKey.COMMON$GIT_PULL)}
          className={CONTEXT_MENU_ICON_TEXT_CLASSNAME}
        />
      </ContextMenuListItem>

      <ContextMenuListItem
        testId="git-push-button"
        onClick={onGitPush}
        className={contextMenuListItemClassName}
      >
        <ToolsContextMenuIconText
          icon={<ArrowUpIcon width={16} height={16} />}
          text={t(I18nKey.COMMON$GIT_PUSH)}
          className={CONTEXT_MENU_ICON_TEXT_CLASSNAME}
        />
      </ContextMenuListItem>

      <ContextMenuListItem
        testId="create-pr-button"
        onClick={onCreatePR}
        className={contextMenuListItemClassName}
      >
        <ToolsContextMenuIconText
          icon={<PrIcon width={16} height={16} />}
          text={t(I18nKey.COMMON$CREATE_PR)}
          className={CONTEXT_MENU_ICON_TEXT_CLASSNAME}
        />
      </ContextMenuListItem>

      <ContextMenuListItem
        testId="create-new-branch-button"
        onClick={onCreateNewBranch}
        className={contextMenuListItemClassName}
      >
        <ToolsContextMenuIconText
          icon={<CodeBranchIcon width={16} height={16} />}
          text={t(I18nKey.COMMON$CREATE_NEW_BRANCH)}
          className={CONTEXT_MENU_ICON_TEXT_CLASSNAME}
        />
      </ContextMenuListItem>
    </ContextMenu>
  );
}
