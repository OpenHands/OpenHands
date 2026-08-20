import React from "react";
import { cn } from "#/utils/utils";

const MENU_SECTION_HEADING_PADDING = "px-2 pb-1 pt-1";
const MENU_SECTION_HEADING_TEXT =
  "text-[11px] font-semibold uppercase tracking-wide text-[var(--oh-muted)]";

export function MenuHeading({
  children,
  suffix,
}: {
  children: React.ReactNode;
  suffix?: React.ReactNode;
}) {
  if (suffix != null) {
    return (
      <div
        role="presentation"
        className={cn(
          "flex items-baseline justify-between gap-2",
          MENU_SECTION_HEADING_PADDING,
        )}
      >
        <span
          className={cn(
            "min-w-0 truncate text-left",
            MENU_SECTION_HEADING_TEXT,
          )}
        >
          {children}
        </span>
        {suffix}
      </div>
    );
  }

  return (
    <div
      role="presentation"
      className={cn(MENU_SECTION_HEADING_PADDING, MENU_SECTION_HEADING_TEXT)}
    >
      {children}
    </div>
  );
}
