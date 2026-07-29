import React from "react";
import { ChevronLeft } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  mobileTopBarIconButtonClassName,
  mobileTopBarIconClassName,
} from "#/utils/mobile-top-bar-icon-button-classes";
import { useConversationStore } from "#/stores/conversation-store";
import { ConversationTabContent } from "../conversation-tabs/conversation-tab-content/conversation-tab-content";
import { ConversationTabs } from "../conversation-tabs/conversation-tabs";

export function ConversationMobilePanelPage({
  onNavigateBack,
}: {
  onNavigateBack: () => void;
}) {
  const { t } = useTranslation("openhands");
  const { setIsRightPanelShown, setHasRightPanelToggled, setSelectedTab } =
    useConversationStore();

  React.useLayoutEffect(() => {
    setIsRightPanelShown(true);
    setHasRightPanelToggled(true);
    const st = useConversationStore.getState();
    if (!st.selectedTab) {
      setSelectedTab("files");
    }
    return () => {
      setIsRightPanelShown(false);
      setHasRightPanelToggled(false);
    };
  }, [setIsRightPanelShown, setHasRightPanelToggled, setSelectedTab]);

  const handleBack = () => {
    onNavigateBack();
  };

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--oh-surface)]">
      <div
        data-testid="conversation-mobile-panel-top"
        className="flex h-10 min-h-10 shrink-0 items-center gap-1.5 border-b border-[var(--oh-border)] pl-2.5"
      >
        <button
          type="button"
          data-testid="conversation-mobile-panel-back"
          onClick={handleBack}
          aria-label={t(I18nKey.COMMON$BACK)}
          className={mobileTopBarIconButtonClassName}
        >
          <ChevronLeft
            size={20}
            className={mobileTopBarIconClassName}
            aria-hidden
            strokeWidth={2}
          />
        </button>
        <div className="flex min-h-0 min-w-0 flex-1 items-center self-stretch">
          <div
            data-testid="tabs-pane-header"
            className="flex h-full min-h-0 w-full min-w-0 flex-col justify-center"
          >
            <ConversationTabs variant="compact" />
          </div>
        </div>
      </div>
      <div className="flex min-h-0 flex-1 flex-col bg-[var(--oh-surface)]">
        <ConversationTabContent />
      </div>
    </div>
  );
}
