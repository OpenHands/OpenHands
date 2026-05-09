import React from "react";
import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "../context-menu/context-menu-list-item";
import { Divider } from "#/ui/divider";
import { SettingsNavHeader } from "../settings/settings-nav-header";
import { ToolsContextMenuIconText } from "../controls/tools-context-menu-icon-text";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import CircuitIcon from "#/icons/u-circuit.svg?react";
import SettingsIcon from "#/icons/settings.svg?react";
import CheckIcon from "#/icons/checkmark.svg?react";
import { cn } from "#/utils/utils";
import { CONTEXT_MENU_ICON_TEXT_CLASSNAME } from "#/utils/constants";
import type { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";

const itemClassName = cn(
  "cursor-pointer p-0 h-auto hover:bg-transparent",
  CONTEXT_MENU_ICON_TEXT_CLASSNAME,
);

interface SwitchProfileContextMenuProps {
  profiles: LlmProfileSummary[];
  activeProfileName: string | null;
  onSelect: (profileName: string) => void;
  onClose: () => void;
}

export function SwitchProfileContextMenu({
  profiles,
  activeProfileName,
  onSelect,
  onClose,
}: SwitchProfileContextMenuProps) {
  const { t } = useTranslation();
  const ref = useClickOutsideElement<HTMLUListElement>(onClose);

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const handleSelect = (
    event: React.MouseEvent<HTMLButtonElement>,
    name: string,
  ) => {
    event.preventDefault();
    event.stopPropagation();
    onSelect(name);
    onClose();
  };

  return (
    <ContextMenu
      ref={ref}
      testId="switch-profile-context-menu"
      position="top"
      alignment="left"
      className="left-0 mb-2 bottom-full min-w-[220px] max-h-[60vh] overflow-y-auto"
    >
      <SettingsNavHeader
        text={I18nKey.SETTINGS$AVAILABLE_PROFILES}
        className="px-2 pt-1 pb-1"
      />
      {profiles.map((profile) => {
        const isActive = profile.name === activeProfileName;
        return (
          <ContextMenuListItem
            key={profile.name}
            testId={`switch-profile-option-${profile.name}`}
            onClick={(event) => handleSelect(event, profile.name)}
            className={itemClassName}
            ariaCurrent={isActive ? "true" : undefined}
          >
            <ToolsContextMenuIconText
              icon={<CircuitIcon width={16} height={16} />}
              text={profile.name}
              rightIcon={
                isActive ? <CheckIcon width={14} height={14} /> : undefined
              }
              className={cn(
                CONTEXT_MENU_ICON_TEXT_CLASSNAME,
                isActive && "bg-[#5C5D62]",
              )}
            />
          </ContextMenuListItem>
        );
      })}
      <Divider />
      <Link
        to="/settings"
        onClick={onClose}
        data-testid="switch-profile-open-settings"
        className={cn("block", itemClassName)}
      >
        <ToolsContextMenuIconText
          icon={<SettingsIcon width={16} height={16} />}
          text={t(I18nKey.MODEL$OPEN_SETTINGS)}
          className={CONTEXT_MENU_ICON_TEXT_CLASSNAME}
        />
      </Link>
    </ContextMenu>
  );
}
