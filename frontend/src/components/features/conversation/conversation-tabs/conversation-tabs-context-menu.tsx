import { useTranslation } from "react-i18next";
import { useLocalStorage } from "@uidotdev/usehooks";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "../../context-menu/context-menu-list-item";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
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
  const [unpinnedTabs, setUnpinnedTabs] = useLocalStorage<string[]>(
    "conversation-unpinned-tabs",
    [],
  );

  const shouldUsePlanningAgent = USE_PLANNING_AGENT();

  const tabConfig = [
    {
      tab: "editor",
      icon: GitChanges,
      i18nKey: I18nKey.COMMON$CHANGES,
    },
    {
      tab: "vscode",
      icon: VSCodeIcon,
      i18nKey: I18nKey.COMMON$CODE,
    },
    {
      tab: "terminal",
      icon: TerminalIcon,
      i18nKey: I18nKey.COMMON$TERMINAL,
    },
    {
      tab: "served",
      icon: ServerIcon,
      i18nKey: I18nKey.COMMON$APP,
    },
    {
      tab: "browser",
      icon: GlobeIcon,
      i18nKey: I18nKey.COMMON$BROWSER,
    },
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
    const tabString = tab;
    if (unpinnedTabs.includes(tabString)) {
      // Tab is unpinned, pin it (remove from unpinned list)
      setUnpinnedTabs(
        unpinnedTabs.filter((unpinnedTab) => unpinnedTab !== tabString),
      );
    } else {
      // Tab is pinned, unpin it (add to unpinned list)
      setUnpinnedTabs([...unpinnedTabs, tabString]);
    }
  };

  const isTabPinned = (tab: string) => !unpinnedTabs.includes(tab as string);

  return (
    <ContextMenu
      testId="conversation-tabs-context-menu"
      ref={ref}
      alignment="right"
      position="bottom"
      className="mt-2 w-fit z-[9999]"
    >
      {tabConfig.map(({ tab, icon: Icon, i18nKey }) => {
        const pinned = isTabPinned(tab);
        return (
          <ContextMenuListItem
            key={tab}
            onClick={() => handleTabClick(tab)}
            className="flex items-center gap-2 p-2 hover:bg-[#5C5D62] rounded h-[30px]"
          >
            <Icon className="w-4 h-4" />
            <span className="text-white text-sm">{t(i18nKey)}</span>
            {pinned ? (
              <PillFillIcon className="w-7 h-7 ml-auto flex-shrink-0 text-white -mr-[5px]" />
            ) : (
              <PillIcon className="w-4.5 h-4.5 ml-auto flex-shrink-0 text-white" />
            )}
          </ContextMenuListItem>
        );
      })}
    </ContextMenu>
  );
}
