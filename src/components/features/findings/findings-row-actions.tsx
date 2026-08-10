/**
 * Per-row / drawer triage action menu.
 * @spec PROJETOSIN-188 — findings-row-actions
 */

import React from "react";
import { useTranslation } from "react-i18next";
import type { FindingStatus } from "#/api/pentest/findings-types";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

export type FindingsTriageAction =
  | "confirm"
  | "mark_fp"
  | "duplicate"
  | "accept_risk";

interface FindingsRowActionsProps {
  findingTitle: string;
  disabled?: boolean;
  onAction: (action: FindingsTriageAction) => void;
}

const ACTION_TO_STATUS: Record<FindingsTriageAction, FindingStatus> = {
  confirm: "confirmed",
  mark_fp: "false_positive",
  duplicate: "duplicate",
  accept_risk: "risk_accepted",
};

export function triageActionToStatus(
  action: FindingsTriageAction,
): FindingStatus {
  return ACTION_TO_STATUS[action];
}

export function FindingsRowActions({
  findingTitle,
  disabled = false,
  onAction,
}: FindingsRowActionsProps) {
  const { t } = useTranslation("openhands");
  const [open, setOpen] = React.useState(false);
  const rootRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!open) return undefined;
    const onPointerDown = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("mousedown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("mousedown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const menuLabel = t(I18nKey.FINDINGS$ACTIONS_MENU, {
    title: findingTitle,
  });

  return (
    <div
      ref={rootRef}
      data-testid="findings-row-actions"
      className="relative"
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => event.stopPropagation()}
    >
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={menuLabel}
        disabled={disabled}
        className={cn(
          "rounded-md border border-[var(--oh-border)] px-2 py-1 text-xs text-[var(--oh-text-secondary)] hover:bg-[var(--oh-surface-raised)]",
          disabled && "cursor-not-allowed opacity-50",
        )}
        onClick={() => setOpen((value) => !value)}
      >
        {/* eslint-disable-next-line i18next/no-literal-string -- kebab glyph */}
        <span aria-hidden="true">⋯</span>
      </button>
      {open ? (
        <div
          role="menu"
          className="absolute right-0 z-20 mt-1 min-w-[12rem] rounded-md border border-[var(--oh-border)] bg-[var(--oh-surface)] py-1 shadow-lg"
        >
          <ActionItem
            testId="findings-action-confirm"
            label={t(I18nKey.FINDINGS$ACTION_CONFIRM)}
            onSelect={() => {
              setOpen(false);
              onAction("confirm");
            }}
          />
          <ActionItem
            testId="findings-action-mark-fp"
            label={t(I18nKey.FINDINGS$ACTION_MARK_FP)}
            onSelect={() => {
              setOpen(false);
              onAction("mark_fp");
            }}
          />
          <ActionItem
            testId="findings-action-duplicate"
            label={t(I18nKey.FINDINGS$ACTION_DUPLICATE)}
            onSelect={() => {
              setOpen(false);
              onAction("duplicate");
            }}
          />
          <ActionItem
            testId="findings-action-accept-risk"
            label={t(I18nKey.FINDINGS$ACTION_ACCEPT_RISK)}
            onSelect={() => {
              setOpen(false);
              onAction("accept_risk");
            }}
          />
        </div>
      ) : null}
    </div>
  );
}

function ActionItem({
  testId,
  label,
  onSelect,
}: {
  testId: string;
  label: string;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      data-testid={testId}
      className="block w-full px-3 py-1.5 text-left text-sm text-white hover:bg-[var(--oh-surface-raised)]"
      onClick={onSelect}
    >
      {label}
    </button>
  );
}
