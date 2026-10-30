import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import { ToggleSwitch } from "#/components/features/automations/toggle-switch";
import { KebabMenu } from "#/components/features/automations/kebab-menu";
import PowerIcon from "#/icons/power.svg?react";
import DownloadIcon from "#/icons/download.svg?react";
import TrashIcon from "#/icons/trash.svg?react";
import EditIcon from "#/icons/u-edit.svg?react";
import PlayIcon from "#/icons/play.svg?react";
import { ArrowRightLeft } from "lucide-react";
import { ActiveStatusBadge } from "./active-status-badge";

interface DetailHeaderProps {
  automation: Automation;
  onToggle: () => void;
  /** When provided (and the user can edit), the kebab menu shows an Edit entry. */
  onEdit?: () => void;
  onDelete: () => void;
  /** When provided, the kebab menu shows a Change Runs As entry. */
  onChangeRunAs?: () => void;
  onExport: () => void;
  onDownloadTarball: () => void;
  onRunNow?: () => void;
  isRunningNow?: boolean;
  /** Whether the caller may mutate this automation (manage perm or owner). */
  canManage?: boolean;
  /** Whether the caller may edit this automation. Defaults to `canManage`. */
  canEdit?: boolean;
  /**
   * Whether the caller may flip the enabled switch. Defaults to `canManage`;
   * pass `false` when the caller may only turn the automation off or not at all.
   */
  canToggle?: boolean;
}

export function DetailHeader({
  automation,
  onToggle,
  onEdit,
  onDelete,
  onChangeRunAs,
  onExport,
  onDownloadTarball,
  onRunNow,
  isRunningNow = false,
  canManage = true,
  canEdit = canManage,
  canToggle = canManage,
}: DetailHeaderProps) {
  const { t } = useTranslation("openhands");

  const kebabItems = [
    // Read-only items are always available to anyone who can view.
    {
      label: t(I18nKey.AUTOMATIONS$EXPORT),
      icon: <DownloadIcon className="size-4" />,
      onClick: onExport,
    },
    {
      label: t(I18nKey.AUTOMATIONS$DOWNLOAD_TARBALL),
      icon: <DownloadIcon className="size-4" />,
      onClick: onDownloadTarball,
    },
    // Write items — only when the user may mutate this automation.
    ...(canEdit && onEdit
      ? [
          {
            label: t(I18nKey.AUTOMATIONS$EDIT),
            icon: <EditIcon className="size-4" />,
            onClick: onEdit,
          },
        ]
      : []),
    ...(canToggle
      ? [
          {
            label: automation.enabled
              ? t(I18nKey.AUTOMATIONS$TURN_OFF)
              : t(I18nKey.AUTOMATIONS$TURN_ON),
            icon: <PowerIcon className="size-4" />,
            onClick: onToggle,
          },
        ]
      : []),
    ...(onChangeRunAs
      ? [
          {
            label: t(I18nKey.AUTOMATIONS$CHANGE_RUN_AS),
            icon: <ArrowRightLeft className="size-4" aria-hidden />,
            onClick: onChangeRunAs,
          },
        ]
      : []),
    ...(canManage
      ? [
          {
            label: t(I18nKey.AUTOMATIONS$DELETE),
            icon: <TrashIcon className="size-4" />,
            onClick: onDelete,
          },
        ]
      : []),
  ];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 flex-wrap items-center gap-3">
          <h1 className="min-w-0 break-words text-xl font-medium text-content">
            {automation.name}
          </h1>
          <ActiveStatusBadge automation={automation} />
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-2">
          {canManage && onRunNow && (
            <button
              type="button"
              className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm font-medium text-content transition-colors hover:bg-surface-raised disabled:cursor-not-allowed disabled:opacity-60"
              disabled={isRunningNow || !automation.enabled}
              onClick={onRunNow}
            >
              <PlayIcon className="size-3.5 shrink-0" aria-hidden />
              {isRunningNow
                ? t(I18nKey.AUTOMATIONS$STARTING)
                : t(I18nKey.AUTOMATIONS$RUN_NOW)}
            </button>
          )}
          {canToggle && (
            <ToggleSwitch
              enabled={automation.enabled}
              label={
                automation.enabled
                  ? t(I18nKey.AUTOMATIONS$TURN_OFF)
                  : t(I18nKey.AUTOMATIONS$TURN_ON)
              }
              onToggle={onToggle}
            />
          )}
          <KebabMenu items={kebabItems} />
        </div>
      </div>
    </div>
  );
}
