import SuccessIcon from "#/icons/success.svg?react";
import { cn } from "#/utils/utils";

interface KeyStatusIconProps {
  testId?: string;
  isSet: boolean;
  /**
   * Accessible/visible-on-hover text. Defaults to a plain "a key string is
   * stored" reading, which is all this icon actually checks — it never pings
   * the provider. Override with a context-specific label anywhere the icon
   * sits apart from the key field itself (e.g. a connection row), where
   * "stored" on its own could read as "verified working."
   */
  label?: string;
}

export function KeyStatusIcon({ testId, isSet, label }: KeyStatusIconProps) {
  const resolvedLabel = label ?? (isSet ? "API key set" : "API key not set");
  return (
    <span
      data-testid={testId || (isSet ? "set-indicator" : "unset-indicator")}
      title={resolvedLabel}
      aria-label={resolvedLabel}
    >
      <SuccessIcon className={cn(isSet ? "text-success" : "text-danger")} />
    </span>
  );
}
