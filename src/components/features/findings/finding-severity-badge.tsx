/**
 * Severity badge for findings table / drawer.
 * @spec PROJETOSIN-188 — finding-severity-badge
 */

import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

const SEVERITY_I18N: Record<string, I18nKey> = {
  critical: I18nKey.FINDINGS$SEVERITY_CRITICAL,
  high: I18nKey.FINDINGS$SEVERITY_HIGH,
  medium: I18nKey.FINDINGS$SEVERITY_MEDIUM,
  low: I18nKey.FINDINGS$SEVERITY_LOW,
  info: I18nKey.FINDINGS$SEVERITY_INFO,
};

const SEVERITY_CLASS: Record<string, string> = {
  critical: "border-transparent bg-[var(--oh-color-danger)] text-white",
  high: "border-[var(--oh-color-danger)] bg-[rgba(220,38,38,0.25)] text-[color:var(--oh-color-danger)]",
  medium:
    "border-[var(--oh-color-primary)] bg-[rgba(255,200,80,0.12)] text-[color:var(--oh-color-primary)]",
  low: "border-[var(--oh-border)] bg-[rgba(255,255,255,0.04)] text-[var(--oh-text-secondary)]",
  info: "border-[var(--oh-border)] bg-transparent text-[var(--oh-text-tertiary)]",
};

export function FindingSeverityBadge({ severity }: { severity: string }) {
  const { t } = useTranslation("openhands");
  const key = SEVERITY_I18N[severity] ?? I18nKey.FINDINGS$SEVERITY_INFO;
  const label = t(key);

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium",
        SEVERITY_CLASS[severity] ?? SEVERITY_CLASS.info,
      )}
      aria-label={label}
    >
      {label}
    </span>
  );
}

const STATUS_I18N: Record<string, I18nKey> = {
  new: I18nKey.FINDINGS$STATUS_NEW,
  triaging: I18nKey.FINDINGS$STATUS_TRIAGING,
  confirmed: I18nKey.FINDINGS$STATUS_CONFIRMED,
  false_positive: I18nKey.FINDINGS$STATUS_FALSE_POSITIVE,
  duplicate: I18nKey.FINDINGS$STATUS_DUPLICATE,
  risk_accepted: I18nKey.FINDINGS$STATUS_RISK_ACCEPTED,
};

const STATUS_CLASS: Record<string, string> = {
  new: "border-[var(--oh-color-primary)] text-[color:var(--oh-color-primary)]",
  triaging: "border-[var(--oh-border)] text-[var(--oh-text-secondary)]",
  confirmed:
    "border-[var(--oh-color-success)] text-[color:var(--oh-color-success)]",
  false_positive: "border-[var(--oh-border)] text-[var(--oh-text-tertiary)]",
  duplicate: "border-[var(--oh-border)] text-[var(--oh-text-tertiary)]",
  risk_accepted:
    "border-[var(--oh-color-primary)] text-[var(--oh-text-secondary)]",
};

export function FindingStatusBadge({ status }: { status: string }) {
  const { t } = useTranslation("openhands");
  const key = STATUS_I18N[status] ?? I18nKey.FINDINGS$STATUS_NEW;
  const label = t(key);

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border bg-transparent px-2 py-0.5 text-xs",
        STATUS_CLASS[status] ?? STATUS_CLASS.triaging,
      )}
      aria-label={label}
    >
      {label}
    </span>
  );
}
