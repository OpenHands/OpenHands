import React from "react";
import { Pin } from "lucide-react";
import { NavigationLink } from "#/components/shared/navigation-link";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { useNavigation } from "#/context/navigation-context";
import { hoverRevealActionClassName } from "#/utils/hover-reveal-classes";
import { cn } from "#/utils/utils";
import { SidebarCollapsedIconSlot } from "./sidebar-collapsed-icon-slot";
import {
  SIDEBAR_ICON_SLOT_CLASS,
  SIDEBAR_ROW_INTERACTIVE_CLASS,
  sidebarNavLabelClassName,
  sidebarNavRowClassName,
} from "./sidebar-layout";

function isPathActive(currentPath: string, to: string, end: boolean) {
  if (to === "/") {
    return currentPath === to;
  }

  if (end) {
    return currentPath === to;
  }

  return currentPath === to || currentPath.startsWith(`${to}/`);
}

export interface SidebarNavLinkPinAction {
  pinned: boolean;
  onToggle: () => void;
  /** Localized aria-label; pin vs unpin variants resolved by the caller. */
  label: string;
  testId: string;
}

interface SidebarNavLinkProps {
  to: string;
  label: string;
  end?: boolean;
  indent?: boolean;
  testId?: string;
  disabled?: boolean;
  icon?: React.ReactElement;
  collapsed?: boolean;
  hoverContent?: React.ReactNode;
  /**
   * When true, forces the active style regardless of the current path.
   * Useful for links that should appear active for multiple related routes
   * (e.g. the Extensions link being active on /mcp and /plugins too).
   */
  forceActive?: boolean;
  /** Pin-as-home toggle; rendered only when the sidebar is expanded. */
  pinAction?: SidebarNavLinkPinAction;
}

export function SidebarNavLink({
  to,
  label,
  end = false,
  indent = false,
  testId,
  disabled = false,
  icon,
  collapsed = false,
  hoverContent,
  forceActive = false,
  pinAction,
}: SidebarNavLinkProps) {
  const { currentPath } = useNavigation();
  const active = forceActive || isPathActive(currentPath, to, end);
  const showPinAction = !collapsed && pinAction != null;

  const link = (
    <NavigationLink
      to={to}
      end={end}
      data-testid={testId}
      tabIndex={disabled ? -1 : 0}
      aria-label={collapsed ? label : undefined}
      // Announce the disabled state to assistive tech. The visual disabled
      // styling plus tabIndex=-1 + preventDefault gives sighted/keyboard users
      // the right behaviour already; this closes the screen-reader gap so the
      // link doesn't sound "actionable."
      aria-disabled={disabled || undefined}
      onClick={(e) => {
        if (disabled) {
          e.preventDefault();
        }
      }}
      className={cn(
        sidebarNavRowClassName({ indent, collapsed }),
        !collapsed &&
          (active
            ? SIDEBAR_ROW_INTERACTIVE_CLASS.active
            : SIDEBAR_ROW_INTERACTIVE_CLASS.idle),
        disabled && "opacity-50",
        disabled && "pointer-events-none",
        // Constant right reserve so label truncation doesn't reflow on hover.
        showPinAction && "pr-9",
      )}
    >
      {icon ? (
        collapsed ? (
          <SidebarCollapsedIconSlot active={active}>
            {icon}
          </SidebarCollapsedIconSlot>
        ) : (
          <span className={SIDEBAR_ICON_SLOT_CLASS}>{icon}</span>
        )
      ) : null}
      <span className={sidebarNavLabelClassName(collapsed)}>{label}</span>
    </NavigationLink>
  );

  if (!collapsed) {
    if (!pinAction) return link;

    // NavigationLink renders an <a>, so the pin toggle is an absolutely
    // positioned sibling rather than a nested button. Bare `group` is safe
    // here: only collapsed rows use it, and this branch is expanded-only.
    return (
      <div className="group relative">
        {link}
        <button
          type="button"
          data-testid={pinAction.testId}
          aria-pressed={pinAction.pinned}
          aria-label={pinAction.label}
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            pinAction.onToggle();
          }}
          className={cn(
            "absolute right-1.5 top-1/2 -translate-y-1/2",
            "flex shrink-0 cursor-pointer items-center justify-center rounded-md p-1",
            "text-[var(--oh-muted)] hover:bg-white/10 hover:text-white",
            hoverRevealActionClassName(pinAction.pinned),
          )}
        >
          <Pin
            className={cn("h-3.5 w-3.5", pinAction.pinned && "fill-current")}
            aria-hidden
          />
        </button>
      </div>
    );
  }

  return (
    <StyledTooltip
      content={hoverContent ?? label}
      placement="right"
      tooltipClassName={hoverContent ? "p-0 bg-tertiary text-white" : undefined}
    >
      {link}
    </StyledTooltip>
  );
}
