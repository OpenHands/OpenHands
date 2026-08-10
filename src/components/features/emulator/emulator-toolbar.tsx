import { useTranslation } from "react-i18next";
import { RefreshCw } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

type EmulatorToolbarProps = {
  onRefresh: () => void;
};

export function EmulatorToolbar({ onRefresh }: EmulatorToolbarProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      className="flex h-9 shrink-0 items-center justify-end border-b border-[var(--oh-border)] px-2"
      data-testid="emulator-toolbar"
    >
      <button
        type="button"
        data-testid="emulator-refresh-button"
        onClick={onRefresh}
        aria-label={t(I18nKey.EMULATOR$REFRESH)}
        className={cn(
          "flex h-8 w-8 items-center justify-center rounded text-[var(--oh-muted)]",
          "cursor-pointer hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--foreground)]",
          "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--foreground)]",
        )}
      >
        <RefreshCw className="h-4 w-4" aria-hidden />
      </button>
    </div>
  );
}
