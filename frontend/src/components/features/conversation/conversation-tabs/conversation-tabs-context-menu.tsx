import { useTranslation } from "react-i18next";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "../../context-menu/context-menu-list-item";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useConversationLocalStorageState } from "#/utils/conversation-local-storage";
import { I18nKey } from "#/i18n/declaration";
import TerminalIcon from "#/icons/terminal.svg?react";
import GlobeIcon from "#/icons/globe.svg?react";
import ServerIcon from "#/icons/server.svg?react";
import GitChanges from "#/icons/git_changes.svg?react";
import VSCodeIcon from "#/icons/vscode.svg?react";
import PillIcon from "#/icons/pill.svg?react";
import PillFillIcon from "#/icons/pill-fill.svg?react";
import { USE_PLANNING_AGENT } from "#/utils/feature-flags";
import LessonPlanIcon from "#/icons/lesson-plan.svg?react";

interface ConversationTabsContextMenuProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ConversationTabsContextMenu({
  isOpen,
  onClose,
}: ConversationTabsContextMenuProps) {
  const ref = useClickOutsideElement<HTMLUListElement>(onClose);
  const { t } = useTranslation();
  const { conversationId } = useConversationId();
  const { state, setUnpinnedTabs } =
    useConversationLocalStorageState(conversationId);

  const shouldUsePlanningAgent = USE_PLANNING_AGENT();

  const tabConfig = [
    { tab: "editor", icon: GitChanges, i18nKey: I18nKey.COMMON$CHANGES },
    { tab: "vscode", icon: VSCodeIcon, i18nKey: I18nKey.COMMON$CODE },
    { tab: "terminal", icon: TerminalIcon, i18nKey: I18nKey.COMMON$TERMINAL },
    { tab: "served", icon: ServerIcon, i18nKey: I18nKey.COMMON$APP },
    { tab: "browser", icon: GlobeIcon, i18nKey: I18nKey.COMMON$BROWSER },
  ];

  if (shouldUsePlanningAgent) {
    tabConfig.unshift({
      tab: "planner",
      icon: LessonPlanIcon,
      i18nKey: I18nKey.COMMON$PLANNER,
    });
  }

  if (!isOpen) return null;

  const handleTabClick = (tab: string) => {
    if (state.unpinnedTabs.includes(tab)) {
      setUnpinnedTabs(state.unpinnedTabs.filter((item) => item !== tab));
    } else {
      setUnpinnedTabs([...state.unpinnedTabs, tab]);
    }
  };

  return (
    <ContextMenu
      ref={ref}
      alignment="right"
      position="bottom"
      className="mt-2 w-fit z-[9999]"
    >
      {tabConfig.map(({ tab, icon: Icon, i18nKey }) => {
        const pinned = !state.unpinnedTabs.includes(tab);
        return (
          <ContextMenuListItem
            key={tab}
            onClick={() => handleTabClick(tab)}
            className="flex items-center gap-2 p-2 hover:bg-[#5C5D62] rounded h-[30px]"
          >
            <Icon className="w-4 h-4" />
            <span className="text-white text-sm">{t(i18nKey)}</span>
            {pinned ? (
              <PillFillIcon className="w-7 h-7 ml-auto -mr-[5px]" />
            ) : (
              <PillIcon className="w-4.5 h-4.5 ml-auto" />
            )}
          </ContextMenuListItem>
        );
      })}
    </ContextMenu>
  );
}
