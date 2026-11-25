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
  align,
  suffix,
}: {
  timestamp: string;
  align: "left" | "right";
  suffix?: string;
}) {
  const { t, i18n } = useTranslation("openhands");
  const [, refresh] = React.useReducer((value: number) => value + 1, 0);
  const exact = formatEventTimestamp(timestamp, i18n?.language);

  React.useEffect(() => {
    if (!exact) return undefined;
    const timer = setInterval(refresh, TIMESTAMP_REFRESH_INTERVAL_MS);
    document.addEventListener("visibilitychange", refresh);
    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", refresh);
    };
  }, [exact]);

  if (!exact) return null;
  const relative = formatRelativeEventTimestamp(
    timestamp,
    i18n?.language,
    t(I18nKey.CHAT_INTERFACE$TIMESTAMP_JUST_NOW),
  );

  return (
    <div
      data-testid="message-timestamp"
      className={cn(
        "absolute top-full z-10 flex w-max items-center gap-1.5 pt-0.5 text-xs leading-4 text-muted",
        align === "right" ? "right-0" : "left-0",
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
      {suffix && <span className="whitespace-nowrap">· {suffix}</span>}
    </div>
  );
}
