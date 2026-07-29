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
import { useHasPermission } from "#/hooks/use-has-permission";
import { ActiveStatusBadge } from "./active-status-badge";

interface DetailHeaderProps {
  automation: Automation;
  onToggle: () => void;
  /**
   * When provided, the kebab menu shows an Edit entry. Omitted for cloud
   * backends where the Edit feature is not supported in MVP.
   */
  onEdit?: () => void;
  onDelete: () => void;
  onExport: () => void;
  onDownloadTarball: () => void;
  onRunNow?: () => void;
  isRunningNow?: boolean;
}

export function DetailHeader({
  automation,
  onToggle,
  onEdit,
  onDelete,
  onExport,
  onDownloadTarball,
  onRunNow,
  isRunningNow = false,
}: DetailHeaderProps) {
  const { t } = useTranslation("openhands");
  const canManage = useHasPermission("manage_automations");

  const kebabItems = [
    ...(onEdit
      ? [
          {
            label: t(I18nKey.AUTOMATIONS$EDIT),
            icon: <EditIcon className="size-4" />,
            onClick: onEdit,
          },
        ]
      : []),
    {
      label: automation.enabled
        ? t(I18nKey.AUTOMATIONS$TURN_OFF)
        : t(I18nKey.AUTOMATIONS$TURN_ON),
      icon: <PowerIcon className="size-4" />,
      onClick: onToggle,
    },
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
    {
      label: t(I18nKey.AUTOMATIONS$DELETE),
      icon: <TrashIcon className="size-4" />,
      onClick: onDelete,
    },
  ];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-medium text-content">
            {automation.name}
          </h1>
          <ActiveStatusBadge active={automation.enabled} />
        </div>
        <div className="flex items-center gap-2">
          {canManage && onRunNow && (
            <button
              type="button"
              className="inline-flex items-center gap-1.5 rounded-md border border-[var(--oh-border)] px-3 py-1.5 text-sm font-medium text-content transition-colors hover:bg-surface-raised disabled:cursor-not-allowed disabled:opacity-60"
              disabled={isRunningNow || !automation.enabled}
              onClick={onRunNow}
            >
              <PlayIcon className="size-3.5 shrink-0" aria-hidden />
              {isRunningNow
                ? t(I18nKey.AUTOMATIONS$STARTING)
                : t(I18nKey.AUTOMATIONS$RUN_NOW)}
            </button>
          )}
          {canManage && (
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
          {canManage && <KebabMenu items={kebabItems} />}
        </div>
      </div>
    </div>
  );
}
