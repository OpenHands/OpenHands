import React from "react";
import { useTranslation } from "react-i18next";
import { Minus, PictureInPicture2, Plus, X } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { mobileTopBarIconButtonClassName } from "#/utils/mobile-top-bar-icon-button-classes";
import { cn } from "#/utils/utils";
import type { PopoutMode } from "#/stores/popout-store";

/**
 * Popouts dock into a ~380px column, so their controls run one step
 * smaller than the shared `mobileTopBarIconClassName` (`size-5`) used by the
 * full-width mobile top bars.
 */
const iconClassName = "size-3.5 shrink-0";

interface PopoutChromeProps {
  title: string;
  /**
   * Agent status indicator rendered left of the title. Taken as a slot so this
   * shell stays presentational — the status query belongs to the caller.
   */
  statusIndicator?: React.ReactNode;
  mode: PopoutMode;
  onMaximize: () => void;
  onToggleMinimized: () => void;
  onClose: () => void;
  children?: React.ReactNode;
}

/**
 * Title bar + body shell for a floating popout. The compact chrome keeps
 * the conversation title on the left and window controls on the right.
 */
export function PopoutChrome({
  title,
  statusIndicator,
  mode,
  onMaximize,
  onToggleMinimized,
  onClose,
  children,
}: PopoutChromeProps) {
  const { t } = useTranslation("openhands");
  const titleId = React.useId();
  const isMinimized = mode === "minimized";

  return (
    <section
      data-testid="popout"
      data-mode={mode}
      aria-labelledby={titleId}
      className={cn(
        "flex flex-col overflow-hidden rounded-xl border border-[var(--oh-border)] bg-[var(--oh-background)] text-[var(--oh-foreground)] shadow-lg",
        isMinimized ? "h-11" : "h-[min(560px,70vh)]",
      )}
    >
      <header
        className={cn(
          "flex shrink-0 items-center gap-2 border-b border-[var(--oh-border-subtle)] bg-[var(--oh-surface)] px-3",
          isMinimized ? "h-full cursor-pointer border-b-0" : "h-12",
        )}
        onClick={isMinimized ? onToggleMinimized : undefined}
        onKeyDown={
          isMinimized
            ? (event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onToggleMinimized();
                }
              }
            : undefined
        }
        role={isMinimized ? "button" : undefined}
        tabIndex={isMinimized ? 0 : undefined}
      >
        {statusIndicator}
        <h2
          id={titleId}
          className="min-w-0 flex-1 truncate text-sm font-medium text-[var(--oh-foreground)]"
          title={title}
        >
          {title}
        </h2>
        <div
          className="flex shrink-0 items-center gap-0.5"
          onClick={(event) => event.stopPropagation()}
          onKeyDown={(event) => event.stopPropagation()}
        >
          <button
            type="button"
            data-testid="popout-minimize"
            className={mobileTopBarIconButtonClassName}
            aria-label={
              isMinimized
                ? t(I18nKey.POPOUT$EXPAND)
                : t(I18nKey.POPOUT$MINIMIZE)
            }
            onClick={onToggleMinimized}
          >
            {isMinimized ? (
              <Plus className={iconClassName} aria-hidden />
            ) : (
              <Minus className={iconClassName} aria-hidden />
            )}
          </button>
          <button
            type="button"
            data-testid="popout-maximize"
            className={mobileTopBarIconButtonClassName}
            aria-label={t(I18nKey.POPOUT$MAXIMIZE)}
            onClick={onMaximize}
          >
            {/* Lucide's picture-in-picture-2 insets its sub-window bottom-right,
                which reads as "enter PiP". Mirroring it points the inset back
                out toward the page — i.e. "leave the window, open full page". */}
            <PictureInPicture2
              className={cn(iconClassName, "-scale-x-100")}
              aria-hidden
            />
          </button>
          <button
            type="button"
            data-testid="popout-close"
            className={mobileTopBarIconButtonClassName}
            aria-label={t(I18nKey.POPOUT$CLOSE)}
            onClick={onClose}
          >
            <X className={iconClassName} aria-hidden />
          </button>
        </div>
      </header>
      {/* Keep children mounted while minimized so the live conversation (and
          its WebSocket) survive a collapse/expand cycle. The section's height
          collapses to the title bar, so this body is visually clipped. */}
      <div
        className={isMinimized ? "hidden" : "min-h-0 flex-1 overflow-hidden"}
        aria-hidden={isMinimized}
      >
        {children}
      </div>
    </section>
  );
}
