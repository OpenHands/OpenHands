import { useTranslation } from "react-i18next";
import { TabButton } from "./tab-button";
import { I18nKey } from "#/i18n/declaration";

interface TabNavigationProps {
  activeTab: "system" | "tools";
  onTabChange: (tab: "system" | "tools") => void;
  hasTools: boolean;
}

export function TabNavigation({
  activeTab,
  onTabChange,
  hasTools,
}: TabNavigationProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      className="mb-2 flex border-b border-[var(--oh-border)]"
      role="tablist"
    >
      <TabButton
        isActive={activeTab === "system"}
        onClick={() => onTabChange("system")}
      >
        {t(I18nKey.SYSTEM_MESSAGE_MODAL$SYSTEM_MESSAGE_TAB)}
      </TabButton>
      {hasTools && (
        <TabButton
          isActive={activeTab === "tools"}
          onClick={() => onTabChange("tools")}
        >
          {t(I18nKey.SYSTEM_MESSAGE_MODAL$TOOLS_TAB)}
        </TabButton>
      )}
    </div>
  );
}
