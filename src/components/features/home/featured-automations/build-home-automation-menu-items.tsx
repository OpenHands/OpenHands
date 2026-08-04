import { FileText, Pin, Square } from "lucide-react";
import type { ReactNode } from "react";
import PlayIcon from "#/icons/play.svg?react";
import PowerIcon from "#/icons/power.svg?react";
import EditIcon from "#/icons/u-edit.svg?react";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";

export type HomeAutomationMenuEntry =
  | {
      kind: "item";
      key: string;
      label: string;
      icon: ReactNode;
      onClick: () => void;
      disabled?: boolean;
      testId: string;
    }
  | { kind: "separator"; key: string };

interface BuildHomeAutomationMenuItemsOptions {
  automation: Automation;
  t: (key: I18nKey) => string;
  canManage: boolean;
  canEdit: boolean;
  isRunPending: boolean;
  isCancelPending: boolean;
  canCancel: boolean;
  testIdPrefix: string;
  pinTestId: string;
  isPinned: boolean;
  onRunNow: () => void;
  onCancelRun: () => void;
  onView: () => void;
  onEdit?: () => void;
  onTurnOff: () => void;
  onTogglePin: () => void;
}

/**
 * Home featured-automation kebab (pinned cards + activity rows):
 * quick actions first, separator, then pin/unpin.
 */
export function buildHomeAutomationMenuItems({
  automation,
  t,
  canManage,
  canEdit,
  isRunPending,
  isCancelPending,
  canCancel,
  testIdPrefix,
  pinTestId,
  isPinned,
  onRunNow,
  onCancelRun,
  onView,
  onEdit,
  onTurnOff,
  onTogglePin,
}: BuildHomeAutomationMenuItemsOptions): HomeAutomationMenuEntry[] {
  const items: HomeAutomationMenuEntry[] = [];

  if (canManage) {
    items.push({
      kind: "item",
      key: "run-now",
      label: t(I18nKey.AUTOMATIONS$RUN_NOW),
      icon: <PlayIcon className="size-4" />,
      onClick: onRunNow,
      disabled: isRunPending || !automation.enabled || canCancel,
      testId: `${testIdPrefix}-run-${automation.id}`,
    });
  }

  if (canManage && canCancel) {
    items.push({
      kind: "item",
      key: "cancel-run",
      label: t(I18nKey.AUTOMATIONS$CANCEL_RUN),
      icon: <Square className="size-3.5 fill-current" aria-hidden />,
      onClick: onCancelRun,
      disabled: isCancelPending,
      testId: `${testIdPrefix}-cancel-${automation.id}`,
    });
  }

  items.push({
    kind: "item",
    key: "view",
    label: t(I18nKey.COMMON$VIEW),
    icon: <FileText className="size-4" aria-hidden />,
    onClick: onView,
    testId: `${testIdPrefix}-view-${automation.id}`,
  });

  if (canManage && canEdit && onEdit) {
    items.push({
      kind: "item",
      key: "edit",
      label: t(I18nKey.AUTOMATIONS$EDIT),
      icon: <EditIcon className="size-4" />,
      onClick: onEdit,
      testId: `${testIdPrefix}-edit-${automation.id}`,
    });
  }

  if (canManage) {
    items.push({
      kind: "item",
      key: "turn-off",
      label: t(I18nKey.AUTOMATIONS$TURN_OFF),
      icon: <PowerIcon className="size-4" />,
      onClick: onTurnOff,
      testId: `${testIdPrefix}-turn-off-${automation.id}`,
    });
  }

  items.push(
    { kind: "separator", key: "pin-separator" },
    {
      kind: "item",
      key: "pin",
      label: t(
        isPinned
          ? I18nKey.FEATURED_AUTOMATIONS$UNPIN
          : I18nKey.FEATURED_AUTOMATIONS$PIN,
      ),
      icon: (
        <Pin
          className={isPinned ? "size-3.5 fill-current" : "size-3.5"}
          aria-hidden="true"
        />
      ),
      onClick: onTogglePin,
      testId: pinTestId,
    },
  );

  return items;
}
