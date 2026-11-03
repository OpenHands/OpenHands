import { useTranslation } from "react-i18next";
import XMarkIcon from "#/icons/x-mark.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { useConversationOverviewDrawer } from "./conversation-overview-drawer-context";
import {
  CONVERSATION_OVERVIEW_DRAWER_SECTION,
  type ConversationOverviewDrawerSection,
} from "./conversation-overview-drawer.types";
import {
  ConversationOverviewGitItemsHeaderLink,
  ConversationOverviewGitItemsPanel,
} from "./conversation-overview-git-items-panel";
import { useConversationOverviewStats } from "#/hooks/use-conversation-overview-stats";
import { useCloseOnEscape } from "#/hooks/use-close-on-escape";
import {
  CONVERSATION_SECONDARY_DRAWER_CLOSE_BUTTON_CLASSNAME,
  CONVERSATION_SECONDARY_DRAWER_HEADER_CLASSNAME,
} from "./conversation-secondary-drawer.classes";

interface ConversationOverviewDrawerContentProps {
  className?: string;
}

function getSectionTitleKey(section: ConversationOverviewDrawerSection) {
  switch (section) {
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests:
      return I18nKey.CONVERSATION$OVERVIEW_PULL_REQUESTS;
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.issues:
      return I18nKey.CONVERSATION$OVERVIEW_ISSUES;
    default: {
      const _exhaustive: never = section;
      return _exhaustive;
    }
  }
}

export function ConversationOverviewDrawerContent({
  className,
}: ConversationOverviewDrawerContentProps) {
  const { t } = useTranslation("openhands");
  const { section, closeDrawer } = useConversationOverviewDrawer();
  const { workspaceName } = useConversationOverviewStats();

  // A menu opened after the drawer (such as the Overview ⋯ menu) takes
  // Escape first.
  useCloseOnEscape(Boolean(section), closeDrawer);

  if (!section) {
    return null;
  }

  return (
    <aside
      data-testid="conversation-overview-drawer-content"
      aria-label={t(getSectionTitleKey(section))}
      className={cn(
        "flex h-full w-full min-h-0 flex-col overflow-hidden bg-base-secondary",
        className,
      )}
    >
      <header className={CONVERSATION_SECONDARY_DRAWER_HEADER_CLASSNAME}>
        <button
          type="button"
          data-testid="conversation-overview-drawer-close"
          aria-label={t(I18nKey.BUTTON$CLOSE)}
          onClick={closeDrawer}
          className={CONVERSATION_SECONDARY_DRAWER_CLOSE_BUTTON_CLASSNAME}
        >
          <XMarkIcon className="size-4" aria-hidden />
        </button>
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <h2 className="truncate text-sm font-medium text-content">
            {t(getSectionTitleKey(section))}
          </h2>
          {workspaceName ? (
            <p
              data-testid="conversation-overview-drawer-workspace"
              className="truncate text-xs text-muted"
            >
              {workspaceName}
            </p>
          ) : null}
        </div>
        <div className="shrink-0">
          <ConversationOverviewGitItemsHeaderLink kind={section} />
        </div>
      </header>

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden custom-scrollbar-always">
        <ConversationOverviewGitItemsPanel kind={section} />
      </div>
    </aside>
  );
}
