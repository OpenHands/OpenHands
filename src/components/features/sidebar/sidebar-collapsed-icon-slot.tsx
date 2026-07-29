import React from "react";
import { cn } from "#/utils/utils";
import {
  SIDEBAR_COLLAPSED_ICON_SLOT_CLASS,
  sidebarCollapsedIconBgClassName,
  sidebarCollapsedIconGlyphClassName,
} from "./sidebar-layout";

interface SidebarCollapsedIconSlotProps {
  active: boolean;
  className?: string;
  children: React.ReactNode;
}

/** Collapsed rail: 36px-tall hover/active target aligned with the expanded icon column. */
export function SidebarCollapsedIconSlot({
  active,
  className,
  children,
}: SidebarCollapsedIconSlotProps) {
  return (
    <span className={cn(SIDEBAR_COLLAPSED_ICON_SLOT_CLASS, className)}>
      <span aria-hidden className={sidebarCollapsedIconBgClassName(active)} />
      <span className={sidebarCollapsedIconGlyphClassName(active)}>
        {children}
      </span>
    </span>
  );
}
