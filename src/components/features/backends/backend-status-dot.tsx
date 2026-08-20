import React from "react";
import { cn } from "#/utils/utils";

interface BackendStatusDotProps {
  /** `null` while the first probe is in flight. */
  isConnected: boolean | null | "unavailable";
  className?: string;
}

/**
 * Small colored dot that reflects backend reachability:
 *   - green when connected
 *   - red when disconnected
 *   - dim gray while the first probe is in flight
 */
export function BackendStatusDot({
  isConnected,
  className,
}: BackendStatusDotProps) {
  let color: string;
  let label: string;
  let status: string;
  if (isConnected === "unavailable") {
    color = "bg-[var(--oh-text-tertiary)]";
    label = "No Backend Available";
    status = "unavailable";
  } else if (isConnected === true) {
    color = "bg-[var(--oh-status-success)]";
    label = "Connected";
    status = "connected";
  } else if (isConnected === false) {
    color = "bg-red-500";
    label = "Disconnected";
    status = "disconnected";
  } else {
    color = "bg-[var(--oh-interactive-selected)]";
    label = "Checking connection";
    status = "checking";
  }

  return (
    <span
      data-testid="backend-status-dot"
      data-status={status}
      aria-label={label}
      title={label}
      role="status"
      className={cn(
        "inline-block w-2 h-2 rounded-full shrink-0",
        color,
        className,
      )}
    />
  );
}
