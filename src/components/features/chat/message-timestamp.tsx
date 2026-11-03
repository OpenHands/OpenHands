import React from "react";
import { useTranslation } from "react-i18next";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  formatEventTimestamp,
  formatRelativeEventTimestamp,
} from "#/utils/format-event-timestamp";

const TIMESTAMP_REFRESH_INTERVAL_MS = 1_000;

export function MessageTimestamp({
  timestamp,
  active,
  align,
}: {
  timestamp: string;
  active: boolean;
  align: "left" | "right";
}) {
  const { t, i18n } = useTranslation("openhands");
  const [focused, setFocused] = React.useState(false);
  const [, refresh] = React.useReducer((value: number) => value + 1, 0);
  const exact = formatEventTimestamp(timestamp, i18n?.language);
  const visible = active || focused;

  React.useEffect(() => {
    if (!visible || !exact) return undefined;
    const timer = setInterval(refresh, TIMESTAMP_REFRESH_INTERVAL_MS);
    document.addEventListener("visibilitychange", refresh);
    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", refresh);
    };
  }, [visible, exact]);

  if (!exact) return null;
  const relative = formatRelativeEventTimestamp(
    timestamp,
    i18n?.language,
    t(I18nKey.CHAT_INTERFACE$TIMESTAMP_JUST_NOW),
  );

  return (
    <div
      data-testid="message-timestamp"
      onFocusCapture={() => setFocused(true)}
      onBlurCapture={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget))
          setFocused(false);
      }}
      className={cn(
        // Padding bridges the pointer path from the bubble to its timestamp.
        "absolute top-full z-10 w-max pt-1 text-xs leading-4 text-muted",
        align === "right" ? "right-0 text-right" : "left-0 text-left",
        visible ? "opacity-100" : "pointer-events-none opacity-0",
      )}
    >
      <StyledTooltip
        content={<time dateTime={timestamp}>{exact}</time>}
        placement="top"
      >
        <button
          type="button"
          aria-label={exact}
          className="cursor-default whitespace-nowrap rounded-sm focus-visible:outline-2 focus-visible:outline-focus"
        >
          <time dateTime={timestamp}>{relative}</time>
        </button>
      </StyledTooltip>
    </div>
  );
}
