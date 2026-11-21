import { ArrowRightLeft, Server } from "lucide-react";
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
import { automationListPath } from "#/manifests/automation-interface";
import { AutomationPageHeader } from "../automation-page-header";
import { ActiveStatusBadge } from "./active-status-badge";

interface DetailHeaderProps {
  automation: Automation;
  onToggle: () => void;
  /**
   * When provided, the kebab menu offers flipping `keep_alive`. Omit it when
   * the setting does not apply or cannot be changed (local backend, no
   * permission, draft, or forced on).
   */
  onToggleKeepAlive?: () => void;
  isKeepAlivePending?: boolean;
  /** When provided (and the user can edit), the header shows an Edit button. */
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

const headerButtonClassName =
  "inline-flex h-7 items-center gap-1.5 rounded-md border border-border px-2.5 text-xs font-medium text-content transition-colors hover:bg-surface-raised disabled:cursor-not-allowed disabled:opacity-60";

export function DetailHeader({
  automation,
  onToggle,
  onToggleKeepAlive,
  isKeepAlivePending = false,
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
    ...(onToggleKeepAlive
      ? [
          {
            label:
              automation.keep_alive === true
                ? t(I18nKey.AUTOMATIONS$STOP_KEEPING_SANDBOX_ALIVE)
                : t(I18nKey.AUTOMATIONS$KEEP_SANDBOX_ALIVE),
            icon: <Server className="size-4" aria-hidden />,
            onClick: onToggleKeepAlive,
            disabled: isKeepAlivePending,
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
    <AutomationPageHeader
      testId="automation-detail-header"
      title={automation.name}
      backTo={automationListPath()}
      badge={<ActiveStatusBadge automation={automation} />}
    >
      {canManage && onRunNow && (
        <button
          type="button"
          className={headerButtonClassName}
          disabled={isRunningNow || !automation.enabled}
          onClick={onRunNow}
        >
          <PlayIcon className="size-3.5 shrink-0" aria-hidden />
          {isRunningNow
            ? t(I18nKey.AUTOMATIONS$STARTING)
            : t(I18nKey.AUTOMATIONS$RUN_NOW)}
        </button>
      )}
      {canEdit && onEdit && (
        <button
          type="button"
          data-testid="automation-detail-edit"
          className={headerButtonClassName}
          onClick={onEdit}
        >
          <EditIcon className="size-3.5 shrink-0" aria-hidden />
          {t(I18nKey.AUTOMATIONS$EDIT)}
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
    </AutomationPageHeader>
  );
}
