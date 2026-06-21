import React from "react";
import { useTranslation } from "react-i18next";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "#/components/features/context-menu/context-menu-list-item";
import { ConversationNameContextMenuIconText } from "#/components/features/conversation/conversation-name-context-menu-icon-text";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { I18nKey } from "#/i18n/declaration";
import SettingsGearIcon from "#/icons/settings-gear.svg?react";
import EditIcon from "#/icons/u-edit.svg?react";
import DeleteIcon from "#/icons/u-delete.svg?react";
import CheckmarkIcon from "#/icons/checkmark.svg?react";

interface ProfileActionsMenuProps {
  onEdit: () => void;
  onRename: () => void;
  onSetActive: () => void;
  onDelete: () => void;
  isActive: boolean;
  isActivating: boolean;
  onClose: () => void;
}

type MenuIcon = React.ComponentType<{ width: number; height: number }>;

interface MenuItemSpec {
  testId: string;
  icon: MenuIcon;
  label: string;
  onSelect: () => void;
  isDisabled?: boolean;
  isDestructive?: boolean;
}

export function ProfileActionsMenu({
  onEdit,
  onRename,
  onSetActive,
  onDelete,
  isActive,
  isActivating,
  onClose,
}: ProfileActionsMenuProps) {
  const { t } = useTranslation();
  const ref = useClickOutsideElement<HTMLUListElement>(onClose);

  const items: MenuItemSpec[] = [
    {
      testId: "profile-edit",
      icon: SettingsGearIcon,
      label: t(I18nKey.SETTINGS$PROFILE_EDIT),
      onSelect: onEdit,
    },
    {
      testId: "profile-rename",
      icon: EditIcon,
      label: t(I18nKey.BUTTON$RENAME),
      onSelect: onRename,
    },
    {
      testId: "profile-set-active",
      icon: CheckmarkIcon,
      label: t(I18nKey.SETTINGS$PROFILE_SET_ACTIVE),
      onSelect: onSetActive,
      isDisabled: isActive || isActivating,
    },
    {
      testId: "profile-delete",
      icon: DeleteIcon,
      label: t(I18nKey.BUTTON$DELETE),
      onSelect: onDelete,
      isDestructive: true,
    },
  ];

  return (
    <ContextMenu
      ref={ref}
      testId="profile-actions-menu"
      alignment="right"
      position="bottom"
      className="min-w-[180px]"
    >
      {items.map(
        ({
          testId,
          icon: Icon,
          label,
          onSelect,
          isDisabled,
          isDestructive,
        }) => (
          <ContextMenuListItem
            key={testId}
            testId={testId}
            onClick={() => {
              onSelect();
              onClose();
            }}
            isDisabled={isDisabled}
            className="cursor-pointer p-0 h-auto hover:bg-transparent"
          >
            <ConversationNameContextMenuIconText
              icon={<Icon width={16} height={16} />}
              text={label}
              className={isDestructive ? "text-red-400" : undefined}
            />
          </ContextMenuListItem>
        ),
      )}
    </ContextMenu>
  );
}
