import { Download, FileText } from "lucide-react";
import PlayIcon from "#/icons/play.svg?react";
import PowerIcon from "#/icons/power.svg?react";
import TrashIcon from "#/icons/trash.svg?react";
import EditIcon from "#/icons/u-edit.svg?react";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import type { KebabMenuItem } from "./kebab-menu";

interface BuildAutomationMenuItemsOptions {
  automation: Automation;
  t: (key: I18nKey) => string;
  canManage: boolean;
  onRunNow: (id: string) => void;
  isRunPending: boolean;
  onView: () => void;
  onExport: (automation: Automation) => void;
  onEdit?: (id: string) => void;
  onToggle: (id: string, enabled: boolean) => void;
  onDelete: (id: string) => void;
}

export function buildAutomationMenuItems({
  automation,
  t,
  canManage,
  onRunNow,
  isRunPending,
  onView,
  onExport,
  onEdit,
  onToggle,
  onDelete,
}: BuildAutomationMenuItemsOptions): KebabMenuItem[] {
  return [
    ...(canManage
      ? [
          {
            label: t(I18nKey.AUTOMATIONS$RUN_NOW),
            icon: <PlayIcon className="size-4" />,
            onClick: () => onRunNow(automation.id),
            disabled: isRunPending || !automation.enabled,
          },
        ]
      : []),
    {
      label: t(I18nKey.COMMON$VIEW),
      icon: <FileText className="size-4" aria-hidden />,
      onClick: onView,
    },
    {
      label: t(I18nKey.AUTOMATIONS$EXPORT),
      icon: <Download className="size-4" aria-hidden />,
      onClick: () => onExport(automation),
    },
    ...(canManage && onEdit
      ? [
          {
            label: t(I18nKey.AUTOMATIONS$EDIT),
            icon: <EditIcon className="size-4" />,
            onClick: () => onEdit(automation.id),
          },
        ]
      : []),
    ...(canManage
      ? [
          {
            label: automation.enabled
              ? t(I18nKey.AUTOMATIONS$TURN_OFF)
              : t(I18nKey.AUTOMATIONS$TURN_ON),
            icon: <PowerIcon className="size-4" />,
            onClick: () => onToggle(automation.id, automation.enabled),
          },
          {
            label: t(I18nKey.AUTOMATIONS$DELETE),
            icon: <TrashIcon className="size-4" />,
            onClick: () => onDelete(automation.id),
          },
        ]
      : []),
  ];
}
