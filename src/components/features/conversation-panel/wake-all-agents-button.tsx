import { useTranslation } from "react-i18next";
import { Zap } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

/**
 * "Wake all agents" control for the conversation panel header: resumes every
 * sleeping conversation (paused cloud sandbox or paused execution) in one
 * click, so returning to the app does not mean opening each conversation just
 * to wake it. Rendered only while something is actually asleep.
 */
export function WakeAllAgentsButton({
  count,
  disabled,
  onClick,
}: {
  count: number;
  disabled?: boolean;
  onClick: () => void;
}) {
  const { t } = useTranslation("openhands");

  return (
    <button
      type="button"
      data-testid="wake-all-agents"
      aria-label={t(I18nKey.CONVERSATION_PANEL$WAKE_ALL_AGENTS, { count })}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md",
        "text-muted transition-colors",
        "hover:bg-surface-raised hover:text-contrast",
        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-border",
        "disabled:cursor-not-allowed disabled:opacity-50",
      )}
    >
      <Zap className="h-4 w-4 shrink-0" aria-hidden strokeWidth={2} />
    </button>
  );
}
