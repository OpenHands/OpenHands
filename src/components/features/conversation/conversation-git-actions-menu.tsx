import React, { useLayoutEffect, useState } from "react";
import ReactDOM from "react-dom";
import { GitCommitHorizontal } from "lucide-react";
import { useTranslation } from "react-i18next";
import ArrowDownIcon from "#/icons/u-arrow-down.svg?react";
import ArrowUpIcon from "#/icons/u-arrow-up.svg?react";
import CodeBranchIcon from "#/icons/u-code-branch.svg?react";
import PrIcon from "#/icons/u-pr.svg?react";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { useConversationStore } from "#/stores/conversation-store";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "#/components/features/context-menu/context-menu-list-item";
import { ToolsContextMenuIconText } from "#/components/features/controls/tools-context-menu-icon-text";
import { I18nKey } from "#/i18n/declaration";
import {
  getCreateNewBranchPrompt,
  getCreatePRPrompt,
  getGitCommitPrompt,
  getGitPullPrompt,
  getGitPushPrompt,
} from "#/utils/utils";
import { Provider } from "#/types/settings";

const GIT_MENU_ITEM_CLASSNAME = "!w-auto whitespace-nowrap";

interface ConversationGitActionsMenuProps {
  anchorRef: React.RefObject<HTMLButtonElement | null>;
  onClose: () => void;
  gitProvider: Provider;
  /** Prefix for menu/item `data-testid` values (suffixes: -menu, -commit, …). */
  testIdPrefix?: string;
}

/**
 * Portaled dropdown of agent-prompt git actions (commit, pull, push, PR, branch).
 * Anchored under a trigger; shared by the overview diffs row and header toggle.
 */
export function ConversationGitActionsMenu({
  anchorRef,
  onClose,
  gitProvider,
  testIdPrefix = "conversation-git-actions",
}: ConversationGitActionsMenuProps) {
  const { t } = useTranslation("openhands");
  const menuRef = useClickOutsideElement<HTMLUListElement>(onClose);
  const setMessageToSend = useConversationStore(
    (state) => state.setMessageToSend,
  );
  const [portalStyle, setPortalStyle] = useState<React.CSSProperties>();

  useLayoutEffect(() => {
    const anchor = anchorRef.current;
    if (!anchor) {
      return undefined;
    }

    const updatePosition = () => {
      const rect = anchor.getBoundingClientRect();
      setPortalStyle({
        position: "fixed",
        top: rect.bottom + 4,
        right: window.innerWidth - rect.right,
        zIndex: 50,
      });
    };

    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [anchorRef]);

  const handleCommit = () => {
    setMessageToSend(getGitCommitPrompt());
    onClose();
  };

  const handlePull = () => {
    setMessageToSend(getGitPullPrompt());
    onClose();
  };

  const handlePush = () => {
    setMessageToSend(getGitPushPrompt(gitProvider));
    onClose();
  };

  const handleCreatePr = () => {
    setMessageToSend(getCreatePRPrompt(gitProvider));
    onClose();
  };

  const handleCreateNewBranch = () => {
    setMessageToSend(getCreateNewBranchPrompt());
    onClose();
  };

  if (!portalStyle) {
    return null;
  }

  return ReactDOM.createPortal(
    <ContextMenu
      ref={menuRef}
      testId={`${testIdPrefix}-menu`}
      theme="popover"
      style={portalStyle}
      className="w-max min-w-[8rem]"
    >
      <ContextMenuListItem
        testId={`${testIdPrefix}-commit`}
        onClick={handleCommit}
        className={GIT_MENU_ITEM_CLASSNAME}
      >
        <ToolsContextMenuIconText
          icon={<GitCommitHorizontal className="size-4" aria-hidden />}
          text={t(I18nKey.DIFF_VIEWER$COMMITS)}
        />
      </ContextMenuListItem>
      <ContextMenuListItem
        testId={`${testIdPrefix}-pull`}
        onClick={handlePull}
        className={GIT_MENU_ITEM_CLASSNAME}
      >
        <ToolsContextMenuIconText
          icon={<ArrowDownIcon width={16} height={16} aria-hidden />}
          text={t(I18nKey.COMMON$PULL)}
        />
      </ContextMenuListItem>
      <ContextMenuListItem
        testId={`${testIdPrefix}-push`}
        onClick={handlePush}
        className={GIT_MENU_ITEM_CLASSNAME}
      >
        <ToolsContextMenuIconText
          icon={<ArrowUpIcon width={16} height={16} aria-hidden />}
          text={t(I18nKey.COMMON$PUSH)}
        />
      </ContextMenuListItem>
      <ContextMenuListItem
        testId={`${testIdPrefix}-create-pr`}
        onClick={handleCreatePr}
        className={GIT_MENU_ITEM_CLASSNAME}
      >
        <ToolsContextMenuIconText
          icon={<PrIcon width={16} height={16} aria-hidden />}
          text={t(I18nKey.COMMON$CREATE_PR)}
        />
      </ContextMenuListItem>
      <ContextMenuListItem
        testId={`${testIdPrefix}-create-new-branch`}
        onClick={handleCreateNewBranch}
        className={GIT_MENU_ITEM_CLASSNAME}
      >
        <ToolsContextMenuIconText
          icon={<CodeBranchIcon width={16} height={16} aria-hidden />}
          text={t(I18nKey.COMMON$CREATE_NEW_BRANCH)}
        />
      </ContextMenuListItem>
    </ContextMenu>,
    document.body,
  );
}
