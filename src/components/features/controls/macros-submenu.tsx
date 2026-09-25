import { useTranslation } from "react-i18next";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "../context-menu/context-menu-list-item";
import { ToolsContextMenuIconText } from "./tools-context-menu-icon-text";
import { Divider } from "#/ui/divider";
import { NavigationLink } from "#/components/shared/navigation-link";

import TachometerFastIcon from "#/icons/tachometer-fast.svg?react";
import PrStatusIcon from "#/icons/pr-status.svg?react";
import DocumentIcon from "#/icons/document.svg?react";
import WaterIcon from "#/icons/u-water.svg?react";
import SettingsGearIcon from "#/icons/settings-gear.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { useConversationStore } from "#/stores/conversation-store";
import { useMacrosStore } from "#/stores/macros-store";
import { cn } from "#/utils/utils";

const submenuListItemClassName = "!w-auto whitespace-nowrap";

interface MacrosSubmenuProps {
  onClose: () => void;
}

function macroIcon(id: string) {
  switch (id) {
    case "INCREASE_TEST_COVERAGE":
      return <TachometerFastIcon width={16} height={16} />;
    case "FIX_README":
    case "ADD_DOCS":
      return <DocumentIcon width={16} height={16} />;
    case "AUTO_MERGE_PRS":
      return <PrStatusIcon width={16} height={16} />;
    case "CLEAN_DEPENDENCIES":
      return <WaterIcon width={16} height={16} />;
    default:
      return <DocumentIcon width={16} height={16} />;
  }
}

export function MacrosSubmenu({ onClose }: MacrosSubmenuProps) {
  const { t } = useTranslation("openhands");
  const { setMessageToSend } = useConversationStore();
  const macros = useMacrosStore((state) => state.macros);

  return (
    <ContextMenu testId="macros-submenu" className="overflow-visible">
      {macros.map((macro) => (
        <ContextMenuListItem
          key={macro.id}
          testId={`macro-${macro.id}`}
          onClick={() => {
            setMessageToSend(macro.prompt);
            onClose();
          }}
          className={submenuListItemClassName}
        >
          <ToolsContextMenuIconText
            icon={macroIcon(macro.catalogId ?? macro.id)}
            text={t(macro.title as I18nKey)}
          />
        </ContextMenuListItem>
      ))}
      <Divider inset="menu" />
      <NavigationLink
        to="/settings/macros"
        onClick={onClose}
        data-testid="manage-macros-button"
        className={cn(
          "flex h-[30px] items-center gap-2 rounded p-2 leading-5 text-[var(--oh-foreground)] hover:bg-[var(--oh-interactive-hover)] transition-colors",
        )}
      >
        <SettingsGearIcon
          width={16}
          height={16}
          className="shrink-0"
          aria-hidden
        />
        <span>{t(I18nKey.SETTINGS$MACROS_MANAGE)}</span>
      </NavigationLink>
    </ContextMenu>
  );
}
