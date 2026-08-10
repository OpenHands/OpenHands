import { useTranslation } from "react-i18next";
import { Smartphone } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { LoadingSpinner } from "#/components/shared/loading-spinner";

export type EmulatorEmptyKind =
  | "loading"
  | "idle"
  | "unavailable"
  | "starting"
  | "error";

type EmulatorEmptyStateProps = {
  kind: EmulatorEmptyKind;
  message?: string;
  onStart?: () => void;
};

export function EmulatorEmptyState({
  kind,
  message,
  onStart,
}: EmulatorEmptyStateProps) {
  const { t } = useTranslation("openhands");
  const canStart =
    (kind === "idle" || kind === "error") && typeof onStart === "function";

  return (
    <div
      className="flex h-full min-h-0 flex-col items-center justify-center gap-4 p-6 text-center"
      data-testid={
        kind === "unavailable" ? "emulator-unavailable" : "emulator-empty-state"
      }
    >
      <Smartphone className="h-10 w-10 text-[var(--oh-muted)]" aria-hidden />
      {kind === "loading" && (
        <LoadingSpinner size="small" data-testid="emulator-status-spinner" />
      )}
      {kind === "starting" && (
        <>
          <LoadingSpinner size="small" data-testid="emulator-start-spinner" />
          <p className="text-sm text-[var(--oh-muted)]">
            {t(I18nKey.EMULATOR$STARTING)}
          </p>
        </>
      )}
      {(kind === "idle" || kind === "error" || kind === "unavailable") && (
        <>
          <p
            className="max-w-sm text-sm text-[var(--oh-muted)]"
            data-testid="emulator-status-message"
          >
            {kind === "error"
              ? (message ?? t(I18nKey.EMULATOR$FAILED))
              : kind === "unavailable"
                ? t(I18nKey.EMULATOR$UNAVAILABLE)
                : t(I18nKey.EMULATOR$OPEN)}
          </p>
          {canStart && (
            <button
              type="button"
              data-testid="emulator-start-button"
              onClick={onStart}
              className={cn(
                "flex h-11 min-w-11 items-center justify-center rounded bg-white px-4 text-sm font-medium text-black",
                "cursor-pointer transition-opacity hover:opacity-90",
              )}
            >
              {t(I18nKey.EMULATOR$OPEN)}
            </button>
          )}
        </>
      )}
    </div>
  );
}
