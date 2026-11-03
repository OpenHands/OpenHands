import { Server } from "lucide-react";
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
  /** When provided (and the user can manage), the header shows an Edit button. */
  onEdit?: () => void;
  onDelete: () => void;
  onExport: () => void;
  onDownloadTarball: () => void;
  onRunNow?: () => void;
  isRunningNow?: boolean;
  /** Whether the caller may mutate this automation (manage perm or owner). */
  canManage?: boolean;
  /**
   * Whether the caller may flip the enabled switch. Defaults to `canManage`;
   * pass `false` for a disabled automation the caller did not create, since
   * non-creators may only turn automations off.
   */
  canToggle?: boolean;
  /** Rendered under the name, e.g. the tags editor. */
  children?: React.ReactNode;
}

const headerButtonClassName =
  "inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm font-medium text-content transition-colors hover:bg-surface-raised disabled:cursor-not-allowed disabled:opacity-60";

export function DetailHeader({
  automation,
  onToggle,
  onToggleKeepAlive,
  isKeepAlivePending = false,
  onEdit,
  onDelete,
  onExport,
  onDownloadTarball,
  onRunNow,
  isRunningNow = false,
  canManage = true,
  canToggle = canManage,
  children,
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
        <div className="flex min-w-0 flex-col gap-2">
          <div className="flex min-w-0 flex-wrap items-center gap-3">
            <h1 className="min-w-0 break-words text-xl font-medium text-content">
              {automation.name}
            </h1>
            <ActiveStatusBadge automation={automation} />
          </div>
          {children}
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-2">
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
          {canManage && onEdit && (
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
        </div>
      </div>
    </div>
  );
}
